"""Small supervised utility models with explicit inference-feature allowlists."""
import backend as B
from backend import ROOT,CFG,E,np,read,save,sha,ident
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.dummy import DummyClassifier
import joblib,time,csv,json


def records(split):
    out=[]
    for row in B.rows(split):
        r=read(ROOT/'evaluation'/split/(row['id']+'.json'))
        assert r['complete'] and r['manifest_sha256']==ident();out.append(r)
    return out


def cases(rr,conditions):
    return [dict(v,id=r['id'],condition=c) for r in rr for c,v in r['conditions'].items() if c in conditions]


def keys_for(sample,groups):
    return [(g,k) for g in groups for k in sorted(sample['feature_groups'][g])]


def matrix(samples,keys):
    # Never flatten a result object: all model inputs come from these exact feature groups.
    assert all(g in CFG['feature_groups'] for g,k in keys)
    x=np.asarray([[s['feature_groups'][g][k] for g,k in keys] for s in samples],dtype=float)
    assert np.isfinite(x).all();return x


def classifier(x,y,c):
    learner=LogisticRegression(C=c,max_iter=2000,random_state=CFG['seed']) if len(np.unique(y))>1 else DummyClassifier(strategy='prior')
    return make_pipeline(StandardScaler(),learner).fit(x,y)


def prob_correct(model,x):
    classes=list(model.classes_)
    return model.predict_proba(x)[:,classes.index(1)] if 1 in classes else np.zeros(len(x))


def predict(entry,samples):
    if entry['kind']=='constant':return np.full(len(samples),entry['value'])
    if entry['kind']=='feature':return matrix(samples,entry['keys'])[:,0]
    if entry['kind']=='branch_correctness':
        return prob_correct(entry['accept_model'],matrix(samples,entry['accept_keys']))-prob_correct(entry['reject_model'],matrix(samples,entry['reject_keys']))
    x=matrix(samples,entry['keys']);model=entry['model']
    if entry['kind']=='utility_em':return model.predict_proba(x)@np.asarray(model.classes_,dtype=float)
    return np.clip(model.predict(x),-1.,1.)


def decision_metrics(samples,values,threshold):
    a=np.asarray(values)>threshold
    return dict(f1=float(np.mean([s['accept' if take else 'reject']['f1'] for s,take in zip(samples,a)])),
        em=float(np.mean([s['accept' if take else 'reject']['em'] for s,take in zip(samples,a)])),accept_count=int(a.sum()))


def calibrate(samples,values,target):
    thresholds=sorted(set([-1e30,0.,1e30]+np.quantile(values,CFG['fit_threshold_quantiles']).tolist()))
    trials=[]
    for t in thresholds:
        score=decision_metrics(samples,values,t)
        key=(score[target],score['em' if target=='f1' else 'f1'],-score['accept_count'],t)
        trials.append((key,float(t),score))
    _,threshold,score=max(trials,key=lambda x:x[0]);return threshold,score


def fit():
    B.verify();assert read(ROOT/'development_completed.json')['manifest_sha256']==ident()
    assert not list((ROOT/'evaluation/test').glob('hotpot_*.json')),'Must lock before test inference'
    train=cases(records('train'),CFG['conditions']);cal=cases(records('calibration'),CFG['conditions'])
    sample=train[0];entries={};meta={}
    for name,value in [('always_accept',1.),('always_reject',-1.)]:
        entries[name]=dict(kind='constant',value=value,threshold=0.);meta[name]=dict(threshold=0.,calibration=decision_metrics(cal,predict(entries[name],cal),0.))
    for feature in ['question_gain','entropy_gain']:
        e=dict(kind='feature',keys=[('output',feature)]);values=predict(e,cal)
        entries[feature+'_zero']=dict(e,threshold=0.);meta[feature+'_zero']=dict(threshold=0.,calibration=decision_metrics(cal,values,0.))
        threshold,score=calibrate(cal,values,'f1');entries[feature+'_calibrated']=dict(e,threshold=threshold)
        meta[feature+'_calibrated']=dict(threshold=threshold,calibration=score)
    ak=[('output',k) for k in sorted(sample['feature_groups']['output']) if k.startswith('a_')]
    rk=[('output',k) for k in sorted(sample['feature_groups']['output']) if k.startswith('r_')]
    confidence_trials=[]
    for c in CFG['fit_regularization']:
        e=dict(kind='branch_correctness',accept_keys=ak,reject_keys=rk,
            accept_model=classifier(matrix(train,ak),np.array([s['accept']['em'] for s in train]),c),
            reject_model=classifier(matrix(train,rk),np.array([s['reject']['em'] for s in train]),c))
        threshold,score=calibrate(cal,predict(e,cal),'f1');e['threshold']=threshold
        confidence_trials.append(((score['f1'],score['em'],-score['accept_count'],-c),e,dict(C=c,threshold=threshold,calibration=score)))
    _,entries['calibrated_confidence'],meta['calibrated_confidence']=max(confidence_trials,key=lambda x:x[0])
    families={'utility_output':['output'],'utility_cross':['output','cross']}
    families.update({f'utility_stability{n}':['output','cross',f'stability{n}'] for n in [2,4,8]})
    for family,groups in families.items():
        keys=keys_for(sample,groups);x=matrix(train,keys)
        for target in ['em','f1']:
            y=np.array([s['utility_'+target] for s in train]);trials=[]
            for c in CFG['fit_regularization']:
                model=classifier(x,y.astype(int),c) if target=='em' else make_pipeline(StandardScaler(),Ridge(alpha=1/c)).fit(x,y)
                e=dict(kind='utility_'+target,model=model,keys=keys)
                threshold,score=calibrate(cal,predict(e,cal),target);e['threshold']=threshold
                key=(score[target],score['f1' if target=='em' else 'em'],-score['accept_count'],-c)
                trials.append((key,e,dict(C=c,threshold=threshold,calibration=score,groups=groups,feature_count=len(keys),target=target)))
            name=family+'_'+target;_,entries[name],meta[name]=max(trials,key=lambda x:x[0]);E.log('FIT '+name+' '+json.dumps(meta[name]))
    joblib.dump(entries,ROOT/'estimators.joblib')
    # JSON parameter export makes the tiny models reviewable without deserializing pickle.
    parameters={}
    for name,e in entries.items():
        parameters[name]={k:v for k,v in e.items() if not k.endswith('model')}
        for key in ['model','accept_model','reject_model']:
            if key not in e:continue
            pipeline=e[key];scaler=pipeline.steps[0][1];learner=pipeline.steps[-1][1]
            parameters[name][key]=dict(type=type(learner).__name__,mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),
                coef=getattr(learner,'coef_',np.array([])).tolist(),intercept=np.asarray(getattr(learner,'intercept_',[])).tolist(),
                classes=np.asarray(getattr(learner,'classes_',[])).tolist(),class_prior=np.asarray(getattr(learner,'class_prior_',[])).tolist())
    save(ROOT/'estimator_parameters.json',parameters)
    lock=dict(manifest_sha256=ident(),locked_at=time.strftime('%F %T'),estimator_sha256=sha(ROOT/'estimators.joblib'),
        primary_model='utility_stability8_f1',primary_metric='f1',metadata=meta,train_questions=CFG['counts']['train'],
        calibration_questions=CFG['counts']['calibration'],test_not_used=True,
        result_inputs={split:{p.name:sha(p) for p in sorted((ROOT/'evaluation'/split).glob('hotpot_*.json'))} for split in ['train','calibration']})
    save(ROOT/'selector.json',lock);E.log('ALL ESTIMATORS LOCKED BEFORE TEST')


if __name__=='__main__':fit()
