"""Streamed atlas figures and question-cluster, layer/head-controlled associations."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def ranks(a):
    order=np.argsort(a,axis=-1,kind='stable');v=np.take_along_axis(a,order,axis=-1);n=a.shape[-1];p=np.arange(n)
    starts=np.concatenate([np.ones(a.shape[:-1]+(1,),bool),np.diff(v,axis=-1)!=0],-1)
    ends=np.concatenate([np.diff(v,axis=-1)!=0,np.ones(a.shape[:-1]+(1,),bool)],-1)
    lo=np.maximum.accumulate(np.where(starts,p,0),-1)
    hi=np.minimum.accumulate(np.where(ends,p,n-1)[...,::-1],-1)[...,::-1]
    out=np.empty_like(a);np.put_along_axis(out,order,(lo+hi)/2+1,-1);return out


def cell_rank(a):
    r=ranks(np.swapaxes(a,-1,-2));r-=r.mean(-1,keepdims=True)
    return r.reshape(r.shape[:-2]+(-1,))


def rho(a,b):
    den=np.sqrt((a*a).sum(-1)*(b*b).sum(-1))
    return np.divide((a*b).sum(-1),den,out=np.full_like(den,np.nan),where=den>1e-20)


def number(v):return float(v) if np.isfinite(v) else None
def ci(values):
    good=values[np.isfinite(values)]
    return np.quantile(good,[.025,.975]).tolist() if len(good)>=.95*len(values) else None


def association(records,context,mode,cfg):
    predictors=['relative_l2','normalized_rmse','mean_abs_coordinate_contribution','logit_rmse','native_weighted_logit_variance','cancellation_ratio']
    targets=['route_js','wo_tt_nmse'];features=predictors+targets
    arrays={}
    for name in features:
        rows=[]
        for r in records:
            selected=sorted([v for v in r['functional_features'] if v['context']==context and v['query_mode']==mode],key=lambda v:(v['layer'],v['kv_head']))
            assert len(selected)==288
            rows.append([v[name] for v in selected])
        arrays[name]=np.asarray(rows,dtype=float);assert np.isfinite(arrays[name]).all()
    draws=np.random.default_rng(cfg['seed']).integers(0,32,size=(cfg['bootstrap_samples'],32))
    point={k:cell_rank(v) for k,v in arrays.items()}
    boot={k:cell_rank(v[draws]) for k,v in arrays.items()};table={};bootstrap={}
    for x in predictors:
        table[x]={};bootstrap[x]={}
        for y in targets:
            bs=rho(boot[x],boot[y]);bootstrap[x][y]=bs
            table[x][y]=dict(within_layer_head_spearman=number(rho(point[x],point[y])),question_cluster_ci95=ci(bs),n_questions=32)
    comparisons={}
    for x in ['mean_abs_coordinate_contribution','logit_rmse','native_weighted_logit_variance']:
        comparisons[x]={}
        for y in targets:
            a=table[x][y]['within_layer_head_spearman'];b=table['relative_l2'][y]['within_layer_head_spearman']
            comparisons[x][y]=dict(delta_rho=None if a is None or b is None else a-b,
                paired_question_cluster_ci95=ci(bootstrap[x][y]-bootstrap['relative_l2'][y]))
    return dict(associations=table,vs_relative_l2=comparisons,
        definition='Rank32 questions separately within each layer/KV-head cell; center ranks in each cell; correlate pooled residual ranks. Bootstrap whole32-question clusters and rerank. Layers/heads are not independent samples.')


def sample_plot(root,record,human_head):
    z=np.load(root/record['stats_file'],allow_pickle=False);layers=[3,21,27]
    fig,axes=plt.subplots(3,4,figsize=(17,11),layout='constrained')
    titles=['|K translated-native|','Native-scale normalized |K error|','|Native-Q coordinate contribution|','|Translated-Q coordinate contribution|']
    matrices=[];ticks=[]
    for layer in layers:
        key=f'L{layer:02d}_H{human_head:02d}';meta=record['representatives'][key]
        indices=np.array(meta['token_positions']);take=np.array([list(indices).index(p) for p in meta['fixed_plot_positions']])
        error=np.abs(z['example__'+key+'__k_post_delta'])[take]
        normal=error/z['example__'+key+'__k_normalizer'][None]
        nq=np.abs(z['example__'+key+'__native_q__first_probe_signed_contribution']).mean(0)[take]
        tq=np.abs(z['example__'+key+'__translated_q__first_probe_signed_contribution']).mean(0)[take]
        matrices.append([error,normal,nq,tq]);ticks.append(indices[take])
    for col,title in enumerate(titles):
        maximum=max(float(m[col].max()) for m in matrices)
        for row,layer in enumerate(layers):
            ax=axes[row,col];im=ax.imshow(matrices[row][col],aspect='auto',origin='lower',cmap='magma',vmin=0,vmax=max(maximum,1e-12))
            ax.set_title(f'L{layer} | {title}',fontsize=9);ax.set_xlabel('Post-RoPE coordinate dimension')
            positions=np.linspace(0,len(ticks[row])-1,min(6,len(ticks[row]))).round().astype(int)
            ax.set_yticks(positions,[str(int(ticks[row][i])) for i in positions]);ax.set_ylabel('Body-token position (fixed stride)')
        fig.colorbar(im,ax=axes[:,col].tolist(),fraction=.015)
    fig.suptitle(f'{record["task"]} | {record["id"]} | KV head{human_head} | first answer probe\nContribution averaged over its4 associated Query heads; common per-column scales')
    p=root/'figures'/f'{record["id"]}_H{human_head:02d}_token_dimension.png'
    fig.savefig(p,dpi=170);fig.savefig(p.with_suffix('.svg'));plt.close(fig)


def generate(C,root,cfg,manifest):
    figures=root/'figures';figures.mkdir(exist_ok=True);tasks={};all_records={}
    for task in cfg['test_counts']:
        assert C.read(root/f'evaluation/{task}/completed.json')['manifest_sha256']==manifest()
        records=[C.read(p) for p in sorted((root/f'evaluation/{task}/per_sample').glob('*.json'))]
        assert len(records)==32 and all(r['manifest_sha256']==manifest() for r in records)
        all_records[task]=records;associations={}
        for context in cfg['contexts']:
            associations[context]={}
            for mode in ['native_q','translated_q']:
                result=association(records,context,mode,cfg);associations[context][mode]=result
                C.save(root/f'analysis/{task}_{context}_{mode}.json',dict(manifest_sha256=manifest(),**result))
        atlas={}
        for role in ['k_post','k_pre','v']:
            for metric in ['relative_l2','raw_rmse','normalized_rmse','top1pct_token_energy','top10pct_token_energy']:
                arr=np.array([[[h[metric] for h in layer['heads']] for layer in r['error_summary'][role]] for r in records])
                atlas[role+'__'+metric]=arr.mean(0).tolist()
        dimension={}
        for role in ['k_post','k_pre','v']:
            values=[]
            for r in records:
                with np.load(root/r['stats_file'],allow_pickle=False) as z:values.append(z[role+'__dimension__normalized_rmse'].mean(1))
            dimension[role]=np.mean(values,0).tolist()
        tasks[task]=dict(n=32,atlas_means=atlas,dimension_profiles=dimension,statistics=associations)
        for r in records:
            if r['representatives']:
                for head in [1,5]:sample_plot(root,r,head)
    fig,axes=plt.subplots(2,3,figsize=(14,9),layout='constrained')
    for i,(task,d) in enumerate(tasks.items()):
        for j,role in enumerate(['k_post','k_pre','v']):
            data=np.asarray(d['atlas_means'][role+'__normalized_rmse'])
            im=axes[i,j].imshow(data,origin='lower',aspect='auto',cmap='magma')
            axes[i,j].set_title(task+' | '+role+' normalized RMSE');axes[i,j].set_xlabel('KV head');axes[i,j].set_ylabel('Receiver layer')
            axes[i,j].set_xticks(range(8),range(1,9));axes[i,j].set_yticks([0,5,11,17,23,29,35],[1,6,12,18,24,30,36]);fig.colorbar(im,ax=axes[i,j])
    fig.savefig(figures/'layer_head_error_atlas.png',dpi=180);fig.savefig(figures/'layer_head_error_atlas.svg');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,9),layout='constrained')
    for i,(task,d) in enumerate(tasks.items()):
        for j,role in enumerate(['k_post','k_pre','v']):
            data=np.asarray(d['dimension_profiles'][role]);im=axes[i,j].imshow(data,origin='lower',aspect='auto',cmap='magma')
            axes[i,j].set_title(task+' | '+role+' normalized dimension RMSE');axes[i,j].set_xlabel('Dimension');axes[i,j].set_ylabel('Receiver layer');fig.colorbar(im,ax=axes[i,j])
    fig.savefig(figures/'layer_dimension_atlas.png',dpi=180);fig.savefig(figures/'layer_dimension_atlas.svg');plt.close(fig)
    C.save(root/'summary.json',dict(manifest_sha256=manifest(),status='completed',tasks=tasks,config=cfg,
        conclusion='Experiments1/2 only: error distribution and functional associations. No dimension repair or causal validation has been run.'))
    lines=['# Fine-grained KV error and functional sensitivity: C, ARC32+Hotpot32','',
        '| Task | Context | Q | Predictor | Correlation with Attention JS | Correlation with Head post-Wo error |',
        '|---|---|---|---|---:|---:|']
    for task,d in tasks.items():
        for context,refs in d['statistics'].items():
            for mode,result in refs.items():
                for predictor in ['relative_l2','mean_abs_coordinate_contribution','logit_rmse','native_weighted_logit_variance']:
                    vals=[result['associations'][predictor][y]['within_layer_head_spearman'] for y in ['route_js','wo_tt_nmse']]
                    lines.append('|'+task+'|'+context+'|'+mode+'|'+predictor+'|'+'|'.join('undefined' if v is None else f'{v:.3f}' for v in vals)+'|')
    lines+=['','![Layer/head atlas](figures/layer_head_error_atlas.png)','![Coordinate profiles](figures/layer_dimension_atlas.png)','',
        'All token/head/coordinate error marginals and contribution summaries are in error_stats/*.npz (derived statistics only). Selected critical-token signed error/contribution vectors are included; full native/translated KV and full dense contribution tensors are not stored.','',
        *['- '+v for v in cfg['caveats']]]
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
