"""Preregistered curve rules, expressed in score fractions (8pp =0.08)."""
from statistics import median


def classify(values, cuts):
    assert len(values)==len(cuts)==7 and cuts==[0,6,12,18,24,30,36]
    differences=[b-a for a,b in zip(values,values[1:])]
    middle=median(differences)
    candidates=[i+1 for i,v in enumerate(differences) if v>=.08-1e-12 and v>=2*middle-1e-12]
    xbar=sum(cuts)/len(cuts);ybar=sum(values)/len(values)
    slope=sum((x-xbar)*(y-ybar) for x,y in zip(cuts,values))/sum((x-xbar)**2 for x in cuts)
    fitted=[ybar+slope*(x-xbar) for x in cuts]
    total=sum((v-ybar)**2 for v in values)
    r2=1-sum((y-yhat)**2 for y,yhat in zip(values,fitted))/total if total>1e-20 else None
    localized=bool(candidates)
    distributed=all(-.03-1e-12<=v<=.08+1e-12 for v in differences) and r2 is not None and r2>=.9-1e-12
    label='localized' if localized and not distributed else 'distributed' if distributed and not localized else 'ambiguous'
    return dict(label=label,localized_rule=localized,distributed_rule=distributed,
        qualifying_groups=candidates,median_marginal=middle,marginal_deltas=differences,
        linear_r2=r2,linear_slope=slope,monotone=all(v>=-1e-9 for v in differences),
        native_minus_translated=values[-1]-values[0],
        recovery_interpretation_applicable=values[-1]>values[0])


def overall(decisions):
    applicable=[v['label'] for v in decisions if v['recovery_interpretation_applicable']]
    return applicable[0] if applicable and len(set(applicable))==1 else 'ambiguous'


def check_rules():
    cuts=[0,6,12,18,24,30,36]
    assert classify([0,.02,.04,.06,.26,.28,.30],cuts)['label']=='localized'
    assert classify([0,.05,.10,.15,.20,.25,.30],cuts)['label']=='distributed'
    assert classify([0,0,0,0,0,0,0],cuts)['label']=='ambiguous'
    assert classify([.4,.3,.4,.3,.4,.3,.4],cuts)['monotone'] is False
    # Exact boundary, conflicting rules and zero-variance behavior are fixed.
    d=classify([0,.04,.08,.12,.16,.20,.28],cuts)
    assert d['localized_rule'] and d['distributed_rule'] and d['label']=='ambiguous'
    assert overall([dict(label='localized',recovery_interpretation_applicable=True),
                    dict(label='distributed',recovery_interpretation_applicable=True)])=='ambiguous'
    assert overall([dict(label='localized',recovery_interpretation_applicable=False)])=='ambiguous'
    return dict(status='completed',thresholds_fixed=True,pp_units_checked=True,
        overlap_and_zero_variance_ambiguous=True,overall_mixed_labels_ambiguous=True)


if __name__=='__main__':
    import json
    print(json.dumps(check_rules()))
