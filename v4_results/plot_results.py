from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('/home/xiaolongwu/vla_workspace/vla_interview_project/vla_training_ablation_v4');out=Path('/mnt/c/VLATest/v4_results')
summary=json.loads((out/'summary.json').read_text());fig,axes=plt.subplots(1,3,figsize=(13,3.6),layout='constrained');colors=['#2563eb','#e59719','#168572']
for r,c in zip(summary,colors):
 run=Path(r['run']);s=[json.loads(l)['loss'] for l in (run/'steps.jsonl').read_text().splitlines()]
 axes[0].plot(np.arange(25,476),np.convolve(s,np.ones(50)/50,mode='valid'),label=r['group'],color=c,lw=1.7)
axes[0].set(title='Training loss (50-step mean)',xlabel='Optimizer step',ylabel='Flow matching loss');axes[0].legend(frameon=False)
x=np.arange(3);axes[1].bar(x-.18,[r['peak_allocated'] for r in summary],width=.36,color=colors,label='Allocated');axes[1].bar(x+.18,[r['peak_reserved'] for r in summary],width=.36,color=colors,alpha=.35,label='Reserved');axes[1].set(xticks=x,xticklabels=list('BCD'),title='Measured training CUDA peak',ylabel='GiB',ylim=(0,1.3));axes[1].legend(frameon=False,fontsize=8)
axes[2].bar(x,[r['trainable']/1000 for r in summary],color=colors,width=.55);axes[2].set(xticks=x,xticklabels=list('BCD'),title='Trainable LoRA parameters',ylabel='Thousands',ylim=(0,480))
for i,r in enumerate(summary):axes[2].text(i,r['trainable']/1000+8,f"{r['trainable']/1000:.1f}k",ha='center',fontsize=9)
for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
fig.savefig(out/'training_comparison.png',dpi=180);plt.close(fig)
print(out/'training_comparison.png')
