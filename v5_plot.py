from pathlib import Path
import json,math,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('/home/xiaolongwu/vla_workspace/outputs/base_single_task_v5');out=Path('/mnt/c/VLATest/v5_results');summary=json.loads((root/'assessment_summary.json').read_text()) if (root/'assessment_summary.json').exists() else {}
fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained');colors={'A':'#2563eb','B':'#d88713','C':'#168572','D':'#8656b8'};records={}
for g in 'ABCD':
 runs=sorted(root.glob(f'{g}_formal2000_bs24_w2_*'))
 if not runs or not (runs[-1]/'steps.jsonl').exists():continue
 p=runs[-1];s=[json.loads(l) for l in (p/'steps.jsonl').read_text().splitlines()];params=json.loads((p/'trainable.json').read_text())['trainable'];n=min(100,len(s));loss=np.array([x['loss'] for x in s]);sm=np.convolve(loss,np.ones(n)/n,'valid')
 axs[0,0].plot(np.arange(len(sm))+n/2,sm,color=colors[g],label=('A-lite' if g=='A' else g)+(' (running)' if not (p/'result.json').exists() else ''),lw=1.7)
 records[g]=dict(allocated=max(x['peak_allocated_gib'] for x in s),reserved=max(x['peak_reserved_gib'] for x in s),params=params)
axs[0,0].set(title='Training flow loss (100-step mean)',xlabel='Optimizer step',ylabel='Loss',xlim=(0,2000));axs[0,0].legend(frameon=False)
x=np.arange(4)
for i,g in enumerate('ABCD'):
 if g in records:
  r=records[g];axs[0,1].bar(i-.15,r['allocated'],width=.3,color=colors[g]);axs[0,1].bar(i+.15,r['reserved'],width=.3,color=colors[g],alpha=.35)
  axs[1,0].bar(i,r['params']/1e6,color=colors[g],width=.55);axs[1,0].text(i,r['params']/1e6*1.2,f"{r['params']/1e6:.3g}M",ha='center',fontsize=9)
 else:
  axs[0,1].text(i,1,'pending',ha='center',rotation=90,color='gray');axs[1,0].text(i,.2,'pending',ha='center',rotation=90,color='gray')
 if g in summary:
  s=summary[g]['main20']['successes'];n=len(s);k=sum(s);p=k/n;z=1.95996398454;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
  axs[1,1].bar(i,p*100,width=.55,color=colors[g],alpha=.8);axs[1,1].errorbar(i,p*100,yerr=[[max(0,100*(p-(c-h)))],[max(0,100*((c+h)-p))]],fmt='none',color='black',capsize=4);axs[1,1].text(i,p*100+4,f'{k}/{n}',ha='center')
 else:axs[1,1].text(i,15,'pending',ha='center',rotation=90,color='gray')
axs[0,1].set(title='Training CUDA peak: allocated / reserved',ylabel='GiB',ylim=(0,8));axs[0,1].axhline(8,color='gray',ls='--',lw=1)
axs[1,0].set(title='Trainable parameters (log scale)',ylabel='Millions',yscale='log',ylim=(.1,200))
axs[1,1].set(title='Matched 20-episode main evaluation',ylabel='Success % (Wilson 95% CI)',ylim=(0,110))
for ax in [axs[0,1],axs[1,0],axs[1,1]]:ax.set_xticks(x,['A-lite','B','C','D'])
for ax in axs.flat:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.savefig(out/'v5_comparison.png',dpi=160);plt.close(fig);print('PLOT_UPDATED',list(summary))
