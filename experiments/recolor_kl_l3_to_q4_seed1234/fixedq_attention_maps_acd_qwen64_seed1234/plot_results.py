"""Question-level summaries and comparable scientific attention-map figures."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
METHODS=['native','A','C','D'];COLORS={'A':'#3478b8','C':'#29945d','D':'#d57a25'}


def example_figure(record,path,binned=False):
    layers=sorted(record['example_maps'],key=int)
    if not layers:return
    fig,axes=plt.subplots(len(layers),4,figsize=(19,4.2*len(layers)),layout='constrained',squeeze=False)
    for i,layer in enumerate(layers):
        value=record['example_maps'][layer]
        maps=value['body_bin_mass'] if binned else value['first_query_token_mass']
        vmax=max(float(np.max(maps[a])) for a in METHODS);vmax=max(vmax,1e-12)
        images=[]
        for j,a in enumerate(METHODS):
            ax=axes[i,j];im=ax.imshow(maps[a],aspect='auto',origin='lower',vmin=0,vmax=vmax,cmap='magma');images.append(im)
            ax.set_title(f'{a} | Layer {layer}',fontsize=11)
            ax.set_yticks([0,7,15,23,31],['1','8','16','24','32']);ax.set_ylabel('Query head')
            if binned:ax.set_xlabel('Body position bin (64 bins; raw mass)')
            else:
                labels=[v if len(v)<19 else v[:16]+'...' for v in value['token_labels']]
                ax.set_xticks(range(len(labels)),labels,rotation=90,fontsize=5.5)
                ax.set_xlabel('Shared body-token columns: index:text')
        fig.colorbar(images[-1],ax=axes[i,:],fraction=.016,pad=.01,label='Original probability mass')
    context='all gold probes averaged; body bins' if binned else 'first answer probe; union of top8 body tokens per method'
    fig.suptitle(f'{record["task"]} | {record["id"]}\nFixed native Query, shared nonbody keys | {context}',fontsize=13)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=180);fig.savefig(path.with_suffix('.svg'));plt.close(fig)


def bootstrap_mean(values,seed=1234,b=1000):
    values=np.asarray(values,dtype=float);n=len(values)
    rng=np.random.default_rng(seed);samples=values[rng.integers(0,n,size=(b,n))].mean(1)
    return dict(mean=values.mean(0).tolist(),ci95=np.quantile(samples,[.025,.975],axis=0).tolist())


def generate(C,root,cfg,manifest):
    root=Path(root);figures=root/'figures';figures.mkdir(exist_ok=True)
    summaries={};records={}
    for task in cfg['test_counts']:
        assert C.read(root/f'evaluation/{task}/completed.json')['manifest_sha256']==manifest()
        rr=[C.read(p) for p in sorted((root/f'evaluation/{task}/per_sample').glob('*.json'))]
        assert len(rr)==64 and all(r['manifest_sha256']==manifest() for r in rr)
        records[task]=rr;summary={}
        for a in ['A','C','D']:
            summary[a]={}
            for metric in ['kl','js','tv','cosine','body_top10_overlap','top1_agreement']:
                v=np.array([[r['layers'][l]['to_native'][a]['mean'][metric] for l in range(36)] for r in rr])
                summary[a][metric]=dict(by_layer=bootstrap_mean(v,cfg['seed'],cfg['bootstrap_samples']),
                    all_layers=bootstrap_mean(v.mean(1),cfg['seed'],cfg['bootstrap_samples']))
            summary[a]['head_kl_mean']=np.array([[r['layers'][l]['to_native'][a]['per_head']['kl'] for l in range(36)] for r in rr]).mean(0).tolist()
            primary='correct' if task=='arc' else 'f1'
            summary[a]['historical_task_score']=sum(r['historical_task_results'][a][primary] for r in rr)/64
            summary[a]['question_attention_kl']={r['id']:float(np.mean([layer['to_native'][a]['mean']['kl'] for layer in r['layers']])) for r in rr}
        pairwise={};residual={}
        for pair in rr[0]['layers'][0]['pairwise']:
            pairwise[pair]={}
            for metric in ['js','cosine','tv']:
                v=[np.mean([layer['pairwise'][pair][metric] for layer in r['layers']]) for r in rr]
                pairwise[pair][metric]=bootstrap_mean(v,cfg['seed'],cfg['bootstrap_samples'])
        for pair in rr[0]['layers'][0]['residual_alignment']:
            v=[]
            for r in rr:
                good=[l['residual_alignment'][pair]['cosine'] for l in r['layers'] if l['residual_alignment'][pair]['cosine'] is not None]
                if good:v.append(float(np.mean(good)))
            residual[pair]=bootstrap_mean(v,cfg['seed'],cfg['bootstrap_samples']) if v else None
        summaries[task]=dict(methods=summary,pairwise=pairwise,residual_alignment=residual,
            native_task_score=sum(r['historical_task_results']['native']['correct' if task=='arc' else 'f1'] for r in rr)/64,
            unit='64 questions; average heads/probes within question, bootstrap complete questions',
            phase1_max_route_kl_difference=max(r['phase1_max_route_kl_difference'] for r in rr))
        for r in rr:
            if r['example_maps']:
                example_figure(r,figures/(r['id']+'_tokens.png'))
                example_figure(r,figures/(r['id']+'_bodybins.png'),binned=True)
        C.save(root/f'evaluation/{task}/summary.json',dict(manifest_sha256=manifest(),status='completed',n=64,**summaries[task]))
    # Layer curves: all methods share axes and uncertainty is over questions.
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for i,(task,d) in enumerate(summaries.items()):
        for j,metric in enumerate(['kl','tv']):
            ax=axes[i,j]
            for a in ['A','C','D']:
                v=d['methods'][a][metric]['by_layer'];mean=np.array(v['mean']);lo,hi=np.array(v['ci95'])
                ax.plot(np.arange(1,37),mean,label=a,color=COLORS[a]);ax.fill_between(np.arange(1,37),lo,hi,alpha=.13,color=COLORS[a])
            ax.axhline(0,color='gray',linestyle='--',linewidth=.7,label='Native self')
            ax.set_title(task+' | '+('KL(Native || method)' if metric=='kl' else 'Total variation'))
            ax.set_xlabel('Receiver layer');ax.set_ylabel(metric);ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Fixed native Q attention differences; same gold contexts; question-bootstrap95% intervals')
    fig.savefig(figures/'layer_curves.png',dpi=180);fig.savefig(figures/'layer_curves.svg');plt.close(fig)
    max_kl=max(np.max(d['methods'][a]['head_kl_mean']) for d in summaries.values() for a in ['A','C','D'])
    fig,axes=plt.subplots(2,3,figsize=(14,10),layout='constrained')
    for i,(task,d) in enumerate(summaries.items()):
        for j,a in enumerate(['A','C','D']):
            ax=axes[i,j];data=np.asarray(d['methods'][a]['head_kl_mean'])
            im=ax.imshow(np.maximum(data,1e-5),origin='lower',aspect='auto',cmap='magma',norm=LogNorm(vmin=1e-5,vmax=max(max_kl,1e-4)))
            ax.set_title(task+' | '+a);ax.set_xlabel('Query head');ax.set_ylabel('Receiver layer')
            ax.set_xticks([0,7,15,23,31],['1','8','16','24','32']);ax.set_yticks([0,5,11,17,23,29,35],['1','6','12','18','24','30','36'])
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Mean KL(Native || method), shared log color scale',fraction=.02)
    fig.suptitle('All36 layers x32 Query heads; equal question weights')
    fig.savefig(figures/'layer_head_kl.png',dpi=180);fig.savefig(figures/'layer_head_kl.svg');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,9),layout='constrained')
    for i,(task,d) in enumerate(summaries.items()):
        for j,metric in enumerate(['js','cosine']):
            arr=np.eye(4) if metric=='cosine' else np.zeros((4,4))
            for x,a in enumerate(METHODS):
                for y,b in enumerate(METHODS[x+1:],x+1):arr[x,y]=arr[y,x]=d['pairwise'][a+'__'+b][metric]['mean']
            ax=axes[i,j];im=ax.imshow(arr,cmap='viridis',vmin=0,vmax=1 if metric=='cosine' else np.log(2))
            ax.set_xticks(range(4),METHODS);ax.set_yticks(range(4),METHODS);ax.set_title(task+' | '+metric)
            for x in range(4):
                for y in range(4):ax.text(y,x,f'{arr[x,y]:.3f}',ha='center',va='center',color='white' if arr[x,y]<(0.5 if metric=='cosine' else .35) else 'black')
            fig.colorbar(im,ax=ax,fraction=.045)
    fig.suptitle('Pairwise attention relations, shared fixed native Query')
    fig.savefig(figures/'pairwise_relations.png',dpi=180);fig.savefig(figures/'pairwise_relations.svg');plt.close(fig)
    C.save(root/'summary.json',dict(manifest_sha256=manifest(),status='completed',tasks=summaries,config=cfg,
        interpretation='Attention probability uses Q and K only. Native is a reference, not proof of correct evidence or better answers. No actual-Q comparison in this follow-up.'))
    lines=['# A/C/D vs Qwen attention maps: fixed native Query','',
        '| Task | Method | Mean KL(Native || method) | JS | TV | Cosine | Body top10 overlap | Historical task score |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for task,d in summaries.items():
        for a in ['A','C','D']:
            v=d['methods'][a];numbers=[v[m]['all_layers']['mean'] for m in ['kl','js','tv','cosine','body_top10_overlap']]
            lines.append('|'+task+'|'+a+'|'+'|'.join(f'{x:.4f}' for x in numbers)+f'|{v["historical_task_score"]*100:.2f}%|')
    lines+=['','Historical task score = ARC accuracy or Hotpot answer F1 on the same64 questions; not a new evaluation.','',
        '![Layer curves](figures/layer_curves.png)','![Head-layer differences](figures/layer_head_kl.png)',
        '![Pairwise attention relationships](figures/pairwise_relations.png)','',
        'Examples use fixed row indices0 and32, layers3/21/27. Token plots use the same union-of-top8 columns and shared probability scale per layer across all4 methods. Body-bin plots sum original probability mass, never renormalize selected tokens.','',
        *['- '+v for v in cfg['caveats']]]
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
