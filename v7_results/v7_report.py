from pathlib import Path
import json,csv,shutil,statistics,ast
import torch
from safetensors.torch import load_file
torch.set_num_threads(4)
R=Path('/home/xiaolongwu/vla_workspace/outputs/one_vlm_layer_v7')
D=Path('/mnt/c/VLATest/v7_results');D.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text())
s=read(R/'assessment_summary.json');run=Path(s['run']);ckpt=Path(s['checkpoint']);source=Path(s['source'])
old=Path('/home/xiaolongwu/vla_workspace/outputs/base_single_task_v5')
a=read(old/'assessment_summary.json')['A']
a10=read(Path('/home/xiaolongwu/vla_workspace/outputs/expert_continuation_v6/assessment_summary.json'))['A_confirmation10']
assert str(source)==a['checkpoint']
assert all((ckpt.parent.parent/f'{i:06d}'/'pretrained_model/model.safetensors').exists() and (ckpt.parent.parent/f'{i:06d}'/'vlm14_fp32_master.safetensors').exists() for i in [250,500])
rows=[json.loads(l) for l in (run/'steps.jsonl').read_text().splitlines()]
assert len(rows)==500 and read(run/'result.json')['reload_exact']
targets=[n for n in read(run/'trainable.json')['parameters'] if '.vlm.model.text_model.layers.14.' in n]
assert len(targets)==9
assert all(x['parameters'][n]['grad_finite'] for x in rows for n in targets)
grad_ever=[n for n in targets if any(x['parameters'][n]['grad_norm']>0 for x in rows)]
update_ever=[n for n in targets if any(x['parameters'][n]['max_update']>0 for x in rows)]
assert len(grad_ever)==9 and len(update_ever)>=7
before=load_file(str(source/'model.safetensors'));after=load_file(str(ckpt/'model.safetensors'))
masters=load_file(str(ckpt.parent/'vlm14_fp32_master.safetensors'))
assert set(masters)==set(targets)
assert all(torch.equal(masters[n].to(after[n].dtype),after[n]) for n in targets)
delta=[]
for n in read(run/'trainable.json')['parameters']:
    delta.append(dict(name=n,numel=after[n].numel(),changed_elements=int(torch.count_nonzero(before[n]!=after[n]))))
audit=dict(steps=500,targets=targets,nonzero_gradient_ever=grad_ever,native_weight_update_ever=update_ever,master_to_native_rounding_exact=True,master_tensors_changed=sum(not torch.equal(masters[n],before[n].float()) for n in targets),native_trainable_tensors=len(delta),native_changed_tensors=sum(x['changed_elements']>0 for x in delta),native_changed_elements=sum(x['changed_elements'] for x in delta),per_tensor=delta)
(R/'weight_update_audit.json').write_text(json.dumps(audit,indent=2))
del before,after,masters
for stage in ['check1','check3','formal500']:
    p=sorted(R.glob(f'F14_{stage}_bs{s["batch"]}_w2_*'))[-1]
    assert read(p/'result.json')['status']=='passed'
    assert read(p/'initial_weights_verified.json')['all_saved_tensors_exact']
    assert read(p/'frozen_weights_verified.json')['exact']
    dest=D/'evidence'/stage;dest.mkdir(parents=True,exist_ok=True)
    for f in p.glob('*.json*'):shutil.copy2(f,dest/f.name)
    if (p/'train/eval').exists():shutil.copytree(p/'train/eval',dest/'native_functional_checks',dirs_exist_ok=True)
    assert (Path(read(p/'result.json')['checkpoint']).parent/'vlm14_fp32_master.safetensors').exists()
for label,ev,start,n in [('main20',s['main20'],20,20),('second10',s['second10'],40,10)]:
    p=Path(ev['path']);states=[json.loads(l)['init_state_index'] for l in (p/'initial_states.jsonl').read_text().splitlines()]
    assert states==list(range(start,start+n)) and len(ev['successes'])==n and read(p/'dense_verified.json')['exact']
    shutil.copytree(p,D/'evidence'/label,dirs_exist_ok=True)
pairs=[]
for label,start,seed,f_eval,a_eval in [('main20',20,6000,s['main20'],a['main20']),('second10',40,7000,s['second10'],a10)]:
    for i,(f_ok,a_ok) in enumerate(zip(f_eval['successes'],a_eval['successes'])):
        pairs.append(dict(split=label,init_state_index=start+i,seed=seed+i,A_success=int(a_ok),F14_success=int(f_ok)))
with (D/'paired_results.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
table=[dict(group='A-lite',source='base',additional_steps=2000,main20=sum(a['main20']['successes']),second10=sum(a10['successes']),checkpoint=str(source)),dict(group='F14',source='A-lite2000',additional_steps=500,main20=sum(s['main20']['successes']),second10=sum(s['second10']['successes']),checkpoint=str(ckpt))]
with (D/'results_v7.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
better=table[1]['main20']>=table[0]['main20'] and table[1]['second10']>=table[0]['second10'] and (table[1]['main20']>table[0]['main20'] or table[1]['second10']>table[0]['second10'])
recommended='F14' if better else 'A-lite'
index=dict(recommended_model=recommended,recommended_checkpoint=str(ckpt if better else source),F14=str(ckpt),F14_fp32_master=str(ckpt.parent/'vlm14_fp32_master.safetensors'),A_reference=str(source))
(D/'RUN_INDEX.json').write_text(json.dumps(index,indent=2))
for folder in ['metadata','validation']:shutil.copytree(R/folder,D/folder,dirs_exist_ok=True)
for p in R.glob('*.json'):shutil.copy2(p,D/p.name)
for p in R.glob('*.log'):shutil.copy2(p,D/p.name)
for p in Path('/mnt/c/VLATest').glob('v7_*.py'):shutil.copy2(p,D/p.name)
val=read(sorted((R/'validation').glob('F14_plus500_*.json'))[-1])
t=read(run/'trainable.json');alloc=max(x['peak_allocated_gib'] for x in rows);reserve=max(x['peak_reserved_gib'] for x in rows)
native=[]
for line in (R/f'F14_formal500_bs{s["batch"]}.log').read_text().splitlines():
    if 'Suite overall aggregated:' in line:native.append(ast.literal_eval(line.split('Suite overall aggregated:',1)[1].strip()))
lines=['# v7：A-lite checkpoint解冻VLM第14层','', '状态：1步→3步→追加500步全部通过，验证及固定30回合比较完成；未从base重训，未重复训练/评测A基线。','',f'当前推荐：**{recommended}**。本轮只检验一个VLM层和一个短预算，不自动延长训练。','', '|模型|原20初态20–39|另10初态40–49|累计正式步数|','|---|---:|---:|---:|',f"|A-lite|{table[0]['main20']}/20|{table[0]['second10']}/10|2000|",f"|F14：A+完整VLM14|{table[1]['main20']}/20|{table[1]['second10']}/10|2500|",'', '## 改动','', '从已有A-lite2000步完整模型checkpoint直接继续。保留Expert与5个状态/动作投影可训练，额外解冻VLM text_model.layers.14整层（从0计数，倒数第二层），包括q/k/v/o、MLP和归一化，共9个参数张量。视觉、VLM其余层、词嵌入与输出头冻结。不是原C组只加q/v LoRA，也没有同时解冻第13层。', '', f"batch{s['batch']}/workers2，追加500步、共{500*s['batch']}起始帧抽样≈{500*s['batch']/5472:.2f}等效轮；动作端LR1e-5，VLM14 LR1e-6；warmup25/decay500、seed42、AMP关闭。可训练参数{t['trainable']:,}；峰值allocated/reserved {alloc:.3f}/{reserve:.3f} GiB。末100步训练Loss {statistics.mean(x['loss'] for x in rows[-100:]):.4f}，固定72帧验证Loss {val['mean_loss']:.4f}。", '', '每个1/3步门槛都从同一A checkpoint开始，正式500步也从A开始，不把门槛训练串入正式预算。250/500步均保存checkpoint。改变参数集合后重建优化器/调度器，是模型权重续训而非原样恢复A的优化器状态。', '', '## 小学习率与精度','', '模型保持原生BF16/FP32格式，VLM14用FP32 master参数和AdamW状态累计小更新，逐步舍入复制回原生权重，避免每个小步都完全丢失。原生全局梯度裁剪后将VLM14梯度复制到master，再step，再复制回模型并清理原始VLM梯度，防止跨步意外累积。Expert/投影仍为原生优化方式。', '', '每个checkpoint旁保存vlm14_fp32_master.safetensors；v7_train.py可用V7_START指定已有checkpoint，若旁边有该文件会核对舍入后一致并恢复master，优化器仍重建。原生模型权重可用标准make_policy加载，评测无需特殊精度补丁。', '', f"实际审计：9个VLM参数张量均有非零梯度，{len(update_ever)}个原生张量曾数值改变，FP32 master中{audit['master_tensors_changed']}个改变；master舍入后与磁盘原生权重全部一致。原生未改变的归一化权重仍受BF16舍入限制，不把requires_grad数量解释为所有元素实际改变。", '', '三阶段均核验起始权重逐项一致、实际梯度/更新、冻结参数前后哈希一致、所有可训练张量原生恢复精确一致。独立评测再次逐项核对全部dense权重。原始每步审计及全量张量差分见evidence与weight_update_audit.json。', '', '## 比较边界','', '与A相同Object0、fixed10、chunk50、双相机、8维状态、7维相对动作、20Hz配置。主20实际初态20–39/seed6000，另10为40–49/seed7000。A结果复用已完成记录，不重复评测。两组初态在前轮均已被查看，本轮属于固定开发基准，不称新holdout或训练外泛化；示范和模拟初态对应未知。', '', '35训练轨迹5472帧、9验证轨迹1395帧不变；全局归一化覆盖验证集合，存在统计重叠。单任务、单训练种子与30回合规模有限，不能保证稳定差异。F14同时增加训练预算、解冻层及该层优化器精度，未另做等预算A续训对照，不能把变化唯一归因于解冻层。', '', '## 文件','', '- RUN_INDEX.json：推荐模型、A来源、F14及FP32 master路径；大模型留WSL，不上传。','- results_v7.csv、paired_results.csv：组间和逐初态比较。','- evidence：三阶段权重/梯度/显存记录及30回合评测、实际初态和视频；主20默认最多保存10视频。','- metadata、validation、脚本及日志可复核。','', '![固定基准对比](C:/VLATest/v7_results/comparison.png)']
if native:lines+=['',f"原生500步单回合功能检查：{native[-1]['pc_success']}%，n={native[-1]['n_episodes']}，仅诊断，不混入正式结果。"]
text='\n'.join(lines)+'\n';(D/'FINAL_REPORT.md').write_text(text,encoding='utf8');(R/'FINAL_REPORT.md').write_text(text,encoding='utf8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,2,figsize=(10,4.5))
for axis,key,n,title in [(ax[0],'main20',20,'Fixed development states 20-39'),(ax[1],'second10',10,'Reused development states 40-49')]:
    for row,color in zip(table,['#3975e8','#e47635']):
        k=row[key];axis.bar(row['group'],k/n*100,color=color);axis.text(row['group'],k/n*100+2,f'{k}/{n}',ha='center')
    axis.set_ylim(0,110);axis.set_ylabel('Success (%)');axis.set_title(title)
fig.tight_layout();fig.savefig(D/'comparison.png',dpi=160);plt.close(fig)
print('V7_FINAL_AUDIT_PASS',recommended,table[1]['main20'],table[1]['second10'],flush=True)
