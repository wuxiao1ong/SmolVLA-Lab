from pathlib import Path
import json,csv,shutil,statistics,math,ast
import torch
from safetensors.torch import load_file
torch.set_num_threads(4)
R=Path('/home/xiaolongwu/vla_workspace/outputs/two_vlm_layers_v8')
D=Path('/mnt/c/VLATest/v8_results');D.mkdir(exist_ok=True)
V5=Path('/home/xiaolongwu/vla_workspace/outputs/base_single_task_v5')
V7=Path('/home/xiaolongwu/vla_workspace/outputs/one_vlm_layer_v7')
read=lambda p:json.loads(p.read_text())
s=read(R/'assessment_summary.json');assert set(s)=={'G13_14','H14'}
a=read(V5/'assessment_summary.json')['A'];f=read(V7/'assessment_summary.json')
a10=read(Path('/home/xiaolongwu/vla_workspace/outputs/expert_continuation_v6/assessment_summary.json'))['A_confirmation10']
budget=read(R/'budget_eval_summary.json');data=read(R/'data_sufficiency_audit.json')
score=lambda ev:f"{sum(ev['successes'])}/{len(ev['successes'])}"
def wilson(k,n):
    z=1.95996398454;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [100*(c-h),100*(c+h)]
def copy_eval(label,ev,offset,n):
    p=Path(ev['path']);states=[json.loads(l)['init_state_index'] for l in (p/'initial_states.jsonl').read_text().splitlines()]
    assert states==list(range(offset,offset+n)) and len(ev['successes'])==n
    assert read(p/'dense_verified.json')['exact']
    shutil.copytree(p,D/'evidence'/label,dirs_exist_ok=True)
table=[dict(group='A2000',layers=[],total_steps=2000,main20=sum(a['main20']['successes']),second10=sum(a10['successes']),checkpoint=a['checkpoint']),dict(group='F14',layers=[14],total_steps=2500,main20=sum(f['main20']['successes']),second10=sum(f['second10']['successes']),checkpoint=f['checkpoint'])]
stats={};audits={};curves={}
for group in ['H14','G13_14']:
    item=s[group];run=Path(item['run']);ckpt=Path(item['checkpoint']);manifest=read(run/'manifest.json');params=read(run/'trainable.json')
    assert item['source']==f['checkpoint'] and manifest['base']==f['checkpoint']
    rows=[json.loads(l) for l in (run/'steps.jsonl').read_text().splitlines()];assert len(rows)==500
    layers=manifest['vlm_layers'];targets=[n for n in params['parameters'] if any(f'.vlm.model.text_model.layers.{i}.' in n for i in layers)]
    assert len(targets)==9*len(layers)
    assert all(x['parameters'][n]['grad_finite'] for x in rows for n in targets)
    update_ever=[n for n in targets if any(x['parameters'][n]['max_update']>0 for x in rows)]
    grad_ever=[n for n in targets if any(x['parameters'][n]['grad_norm']>0 for x in rows)]
    assert len(grad_ever)==len(targets) and len(update_ever)>=7*len(layers)
    masters=load_file(str(ckpt.parent/'vlm13_14_fp32_master.safetensors'));actual=load_file(str(ckpt/'model.safetensors'))
    assert set(masters)==set(targets) and all(torch.equal(v.to(actual[n].dtype),actual[n]) for n,v in masters.items())
    start=load_file(str(Path(item['source'])/'model.safetensors'))
    source_masters=load_file(str(Path(item['source']).parent/'vlm14_fp32_master.safetensors'))
    delta=[dict(name=n,numel=actual[n].numel(),changed_elements=int(torch.count_nonzero(start[n]!=actual[n]))) for n in params['parameters']]
    audits[group]=dict(layers=layers,steps=500,gradients_finite_all=True,nonzero_gradient_ever=grad_ever,native_update_ever=update_ever,master_rounding_exact=True,source_master_names=list(source_masters),master_changed_tensors=sum(not torch.equal(v,source_masters[n] if n in source_masters else start[n].float()) for n,v in masters.items()),native_trainable_tensors=len(delta),native_changed_tensors=sum(x['changed_elements']>0 for x in delta),native_changed_elements=sum(x['changed_elements'] for x in delta),per_tensor=delta)
    del masters,actual,start,source_masters
    for stage in ['check1','check3','formal500']:
        p=sorted(R.glob(f'{group}_{stage}_bs{item["batch"]}_w2_*'))[-1]
        assert read(p/'result.json')['status']=='passed' and read(p/'result.json')['reload_exact']
        assert read(p/'initial_weights_verified.json')['all_saved_tensors_exact'] and read(p/'frozen_weights_verified.json')['exact']
        mr=read(p/'master_restore_verified.json');assert mr['rounded_source_model_exact'] and len(mr['restored_names'])==9
        dest=D/'evidence'/group/stage;dest.mkdir(parents=True,exist_ok=True)
        for q in p.glob('*.json*'):shutil.copy2(q,dest/q.name)
        if (p/'train/eval').exists():shutil.copytree(p/'train/eval',dest/'native_functional_checks',dirs_exist_ok=True)
    assert all((ckpt.parent.parent/f'{n:06d}/pretrained_model/model.safetensors').exists() and (ckpt.parent.parent/f'{n:06d}/vlm13_14_fp32_master.safetensors').exists() for n in [250,500])
    for label,n,offset in [('main20',20,20),('second10',10,40)]:copy_eval(group+'_'+label,item[label],offset,n)
    v250=read(sorted((R/'validation').glob(group+'_250_*.json'))[-1]);v500=read(sorted((R/'validation').glob(group+'_500_*.json'))[-1])
    stats[group]=dict(trainable=params['trainable'],batch=item['batch'],loss_last100=statistics.mean(x['loss'] for x in rows[-100:]),allocated=max(x['peak_allocated_gib'] for x in rows),reserved=max(x['peak_reserved_gib'] for x in rows),first_loss=rows[0]['loss'],validation250=v250['mean_loss'],validation500=v500['mean_loss'],native_checks=[])
    for line in (R/f'{group}_formal500_bs{item["batch"]}.log').read_text().splitlines():
        if 'Suite overall aggregated:' in line:
            z=ast.literal_eval(line.split('Suite overall aggregated:',1)[1].strip());stats[group]['native_checks'].append(dict(pc_success=z['pc_success'],n=z['n_episodes']))
    table.append(dict(group=group,layers=layers,total_steps=3000,main20=sum(item['main20']['successes']),second10=sum(item['second10']['successes']),checkpoint=item['checkpoint']))
    curves[group]=[dict(additional_step=250,loss=v250['mean_loss']),dict(additional_step=500,loss=v500['mean_loss'])]
copy_eval('A_budget1000',budget['A1000'],20,20)
a_rows=[json.loads(l) for l in (Path(a['run'])/'steps.jsonl').read_text().splitlines()]
curves['A']=[]
for step in [500,1000,1500,2000]:
    val=read(sorted((R/'validation').glob(f'A_curve_{step}_*.json'))[-1]) if step<2000 else read(sorted((V5/'validation').glob('A_bs24_2000_*.json'))[-1])
    curves['A'].append(dict(step=step,train_last100=statistics.mean(x['loss'] for x in a_rows[step-100:step]),validation=val['mean_loss'],equivalent_draw_passes=step*24/5472))
epochs=dict(A2000=48000/5472,F14=60000/5472,H14=(60000+500*s['H14']['batch'])/5472,G13_14=(60000+500*s['G13_14']['batch'])/5472,new_layer13_only=500*s['G13_14']['batch']/5472,layer14_since_unfreeze=(12000+500*s['G13_14']['batch'])/5472)
pair=[]
for label,offset,seed,a_ev,f_ev in [('main20',20,6000,a['main20'],f['main20']),('second10',40,7000,a10,f['second10'])]:
    for i in range(20 if label=='main20' else 10):
        pair.append(dict(split=label,init_state_index=offset+i,seed=seed+i,A=int(a_ev['successes'][i]),F14=int(f_ev['successes'][i]),H14=int(s['H14'][label]['successes'][i]),G13_14=int(s['G13_14'][label]['successes'][i])))
discord=dict(G_recovers_H=sum(x['G13_14']==1 and x['H14']==0 for x in pair),G_introduces_H_failures=sum(x['G13_14']==0 and x['H14']==1 for x in pair))
with (D/'paired_results.csv').open('w',newline='') as out:
    writer=csv.DictWriter(out,fieldnames=list(pair[0]));writer.writeheader();writer.writerows(pair)
with (D/'results_v8.csv').open('w',newline='') as out:
    writer=csv.DictWriter(out,fieldnames=['group','layers','total_steps','main20','second10','checkpoint']);writer.writeheader();writer.writerows(table)
review=dict(data=data,epochs=epochs,curves=curves,stats=stats,layer_pairing=discord,A1000_main20=sum(budget['A1000']['successes']),A2000_main20=19,matched_control='same F14 model and restored layer14 masters, same500 additionalsteps/batch/seed/LR; only layer13 added',limits=['single training seed','reused development30, not independent holdout','no controlled data-size experiment','global normalization covers validation episodes','source demonstration scene identity unknown','no claim min/max robot state proves scene diversity'])
(R/'adequacy_review.json').write_text(json.dumps(review,indent=2));(D/'weight_update_audit.json').write_text(json.dumps(audits,indent=2));(D/'adequacy_review.json').write_text(json.dumps(review,indent=2))
best=table[0]
for row in table[2:]:
    if row['main20']>=best['main20'] and row['second10']>=best['second10'] and (row['main20']>best['main20'] or row['second10']>best['second10']):best=row
index=dict(recommended_model=best['group'],recommended_checkpoint=best['checkpoint'],A_reference=a['checkpoint'],F14_source=f['checkpoint'],**{g:s[g]['checkpoint'] for g in s},**{g+'_fp32_master':str(Path(s[g]['checkpoint']).parent/'vlm13_14_fp32_master.safetensors') for g in s})
(D/'RUN_INDEX.json').write_text(json.dumps(index,indent=2))
for folder in ['metadata','validation']:shutil.copytree(R/folder,D/folder,dirs_exist_ok=True)
for p in R.glob('*.json'):shutil.copy2(p,D/p.name)
for p in R.glob('*.log'):shutil.copy2(p,D/p.name)
for p in Path('/mnt/c/VLATest').glob('v8_*.py'):shutil.copy2(p,D/p.name)
lines=['# v8：两层VLM与同预算单层对照','', '已完成两组各1步→3步→500步、固定验证与各30回合比较；复用旧模型检查学习曲线，没有从base重训。','',f"当前推荐：**{best['group']}**。推荐仅在两组固定基准都不差、至少一组更好时更新，否则保留A。这个规则不是统计显著性证明。",'', '|模型|解冻VLM层（从0计数）|主20|另10|累计正式步数|','|---|---|---:|---:|---:|']
for row in table:lines.append(f"|{row['group']}|{row['layers']}|{row['main20']}/20|{row['second10']}/10|{row['total_steps']}|")
lines+=['',f"两层对照的逐初态差异：G修复H的{discord['G_recovers_H']}个失败，同时引入{discord['G_introduces_H_failures']}个新失败。",'', '## 对照设计','', 'G13_14与H14都独立从同一F14正式500步checkpoint及其第14层FP32 master开始；G额外解冻完整第13层，H仍只有第14层可训练。Expert和5个投影均继续训练，其余VLM与视觉冻结。各新增500步，同batch、seed42、动作端LR1e-5、VLM LR1e-6、warmup25/decay500；两组优化器/调度器都重建。对照用于区分多解冻一层与仅多训练500步。', '', '1/3步门槛独立从F14开始，正式500步也从F14开始，不承接门槛权重；保存250/500 checkpoint。第14层FP32累积权重恢复并核对舍入一致，第13层从原生权重建立master。每步全局梯度裁剪后复制到master，step后写回原生BF16并清除原始VLM梯度。原生模型可标准make_policy加载。', '', '|组别|可训练参数|batch|末100步Loss|验证250→500|峰值allocated/reserved GiB|','|---|---:|---:|---:|---:|---:|']
for group,t in stats.items():lines.append(f"|{group}|{t['trainable']:,}|{t['batch']}|{t['loss_last100']:.4f}|{t['validation250']:.4f}→{t['validation500']:.4f}|{t['allocated']:.3f}/{t['reserved']:.3f}|")
lines+=['', '## 验收与解释边界','', '六个训练阶段起始权重逐项一致，原第14层master恢复一致；冻结参数哈希不变，全部可训练张量原生保存恢复一致。两层18张量/单层9张量均审计梯度有限、逐层非零梯度与实际更新；原生BF16归一化可能未跨舍入精度，master的细小变化仍保存。完整记录见weight_update_audit.json与evidence。', '', 'A/F14复用已完成19/20+10/10、18/20+10/10对照，不重复训练/评测。两组新模型主20为实际初态20–39、seed6000，另10为40–49、seed7000，Object0、fixed10、chunk50、20Hz相对动作。均是已查看开发基准，不是新盲测。单种子且没有跨任务验证；不强称逐步随机噪声完全相同。小样本的一两个回合差异不能证明稳定层数优势。', '', '数据和轮数的详细判断见 [SUFFICIENCY_REVIEW.md](C:/VLATest/v8_results/SUFFICIENCY_REVIEW.md)。','', '![对照与曲线](C:/VLATest/v8_results/comparison.png)','', '## 产物','', '- RUN_INDEX.json：推荐模型、F14来源、G/H及master路径。大模型仍在WSL，原模型均保留，不上传。','- results_v8.csv、paired_results.csv：组间与逐初态结果。','- evidence：六阶段和两组60回合、旧A1000的20回合结果、实际初态和视频。主20默认最多保存10视频。','- validation：旧A500/1000/1500与G/H250/500固定72帧；A2000参照旧记录。','- adequacy_review.json、data_sufficiency_audit.json、weight_update_audit.json、所有脚本/日志。']
(D/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
adequacy=['# 数据、训练轮数与论证充分性','', '**结论分用途：当前数据已足以让模型在这个单任务的固定开发基准上学会执行；尚不能确认数据足以覆盖新场景，也不能证明训练已达到全局最优。**','', '## 数据是否足够','', f"任务24与Object0语言精确一致，完整44条示范共6867帧，固定35条训练5472帧、9条验证1395帧，episode无交叉；所有数值有限。轨迹长度{min(data['episode_lengths'].values())}–{max(data['episode_lengths'].values())}帧。验证第0维状态/动作约1.0%/1.3%超出训练min/max，其余维没有超界；这只能说明数值范围接近，不能证明物体场景覆盖充分。", '', '5472帧和高度重叠的50步动作窗不是5472个独立摆放场景。数据没有物体初始位姿数值标注，示范与模拟初态对应未知，不能核实覆盖了多少不同摆放/干扰物/恢复情况。35个训练episode只是当前可用训练集合，不是经数据规模实验得到的最小充分数量。', '', 'A已有19/20和10/10说明数据可学；反过来不能把它当作独立泛化证明，因为初态已反复用于开发，留出统计也用了全局归一化。要证明数据量充分，需要从同一初始模型用不同数据规模、按场景分组留出并重复种子；不能从已看过35条的checkpoint改用10条子集，就宣称只用10条也足够。本轮遵循减少训练要求，没有启动这些重训。', '', '## 已经训练了多少轮','', '|范围|等效抽样轮数|','|---|---:|']
for name,value in epochs.items():adequacy.append(f'|{name}|{value:.2f}|')
adequacy+=['', '等效轮数按实际batch×steps/5472累加，不是去重epoch。A Expert已抽样约8.77轮；F14 Expert累计约10.96轮，新第14层只有约2.19轮；G/H再增加约2.19轮。新第13层不能按Expert累计约13轮来描述，它只训练了本轮500步。旧metadata/data_audit.json的epochs_2000=0.365按batch1计算，不能套用当前batch24。', '', '## 现有学习曲线','', '|A checkpoint步数|训练末100步Loss|固定72帧验证Loss|','|---:|---:|---:|']
for row in curves['A']:adequacy.append(f"|{row['step']}|{row['train_last100']:.4f}|{row['validation']:.4f}|")
adequacy+=['',f"仅评测已有A1000 checkpoint：主20为{score(budget['A1000'])}；已有A2000为19/20。本项没有新增训练。",'', '|续训组|250步验证Loss|500步验证Loss|','|---|---:|---:|']
for group in ['H14','G13_14']:adequacy.append(f"|{group}|{stats[group]['validation250']:.4f}|{stats[group]['validation500']:.4f}|")
adequacy+=['', '训练Loss、离线验证和闭环成功率需要分别判断。末尾验证若持平或变差、闭环也无增益，不支持盲目增加轮数；若验证仍改善，也只能说明当前配置可能未稳定，不能保证再训就能提高成功率。单个学习率、单个种子和有限checkpoint不能证明充分收敛或全局最优。', '', '## 实验论述能写到什么程度','', f"本轮同源、同预算对照：H14={score(s['H14']['main20'])}+{score(s['H14']['second10'])}，G13_14={score(s['G13_14']['main20'])}+{score(s['G13_14']['second10'])}。可写本任务、本训练预算、本开发初态下的实测差异；不能写两层VLM普遍优于单层、35条数据已经充足、或数据外泛化已得到证明。", '', '两层与单层控制了相同起点和额外训练预算，比直接比较不同累计步数更能解释层数影响，但只有一个训练种子、固定30个已查看初态。举例19/20的Wilson95%区间约76.4%–99.1%；该区间还以独立二项抽样为前提，本开发基准反复选模不满足盲测条件，不能把它当作可靠总体置信度。', '', '下一阶段若要加强论证，优先增加未参与调参的场景/物体摆放与独立评测、重复训练种子，再做控制数据规模的实验。若只追求当前任务执行，已有高成功A模型仍应保留，不以参数更多或轮数更多为升级理由。', '', '## 额外时间尺度限制','', '数据metadata标10Hz，当前LiberoEnv通过fps参数将control_freq设为20Hz，源码已核查；原始动作降采样与每步物理时间语义尚未重建。这个已观测差异不等于任务选错，也不能仅凭它判定训练管线错误，A在相同设置下已有成功。所有本轮组保持同样20Hz，没有中途改变。若要论证时间对齐无影响，仍需核对原始转换和做单变量时间尺度检查。']
adequacy+=['', '## 中间checkpoint的调度限制', '', f"A1000和A2000来自同一2000步训练日程，1000步附近记录学习率为{a_rows[999]['lr']:.6g}，末步为{a_rows[1999]['lr']:.6g}。A1000不是独立优化过学习率日程的1000步训练实验；G/H250也只是500步计划的中间点。因此可用它们判断这条训练路径的趋势，不能据此确定最省算力的最优步数。"]
best_offline=min(curves['A'],key=lambda x:x['validation'])
adequacy+=['', '## 本次实测判断', '', f"A四个已存checkpoint中，固定72帧验证Loss最低的是{best_offline['step']}步（{best_offline['validation']:.4f}），2000步为{curves['A'][-1]['validation']:.4f}。训练末100步均值从500步的{curves['A'][0]['train_last100']:.4f}降到2000步的{curves['A'][-1]['train_last100']:.4f}；不能只用训练Loss下降判断还需要更多轮数。"]
for group in ['H14','G13_14']:
    z=stats[group];trend='未见进一步降低' if z['validation500']>=z['validation250'] else '仍有降低，但两点曲线不足以确认收敛'
    adequacy.append(f"{group}验证250→500为{z['validation250']:.4f}→{z['validation500']:.4f}，后半段{trend}。")
kb=sum(budget['A1000']['successes'])
if kb>=18:adequacy.append(f"A1000主20为{kb}/20，A2000为19/20；这一小样本比较未显示1000步之后有很大的闭环增益，不能据此宣称1000步是最优预算。")
else:adequacy.append(f"A1000主20为{kb}/20，A2000为19/20，说明这条训练路径1000步之后仍有学习收益；它不能证明2000步之后仍需增加训练。")
adequacy+=['不能从增加VLM层无收益直接推断数据不足：已有视觉表示足够、当前优化设置不合适、场景覆盖有限或新层训练尚不充分都可能产生类似现象。本轮没有单独隔离这些原因。数据量充分性仍缺干净的数据规模实验和独立场景测试；当前没有证据支持盲目增加训练轮数。']
if all(sum(s['H14'][k]['successes'])>=sum(f[k]['successes']) for k in ['main20','second10']) and any(sum(s['H14'][k]['successes'])>sum(f[k]['successes']) for k in ['main20','second10']):
    adequacy.append(f"本轮H14相对旧F14出现开发基准改善：F14为{score(f['main20'])}+{score(f['second10'])}，继续500步的H14为{score(s['H14']['main20'])}+{score(s['H14']['second10'])}。因此不能笼统说所有模块都已训够：第14层从500步继续到累计1000步在这条路径上有收益，而新增第13层只训练500步。该结论仍限于本日程、单种子、已查看初态，且优化器重建。")
else:adequacy.append('同一第14层继续500步没有在两组基准上同时显现优势，不支持把加轮数作为默认升级方向。')
(D/'SUFFICIENCY_REVIEW.md').write_text('\n'.join(adequacy)+'\n',encoding='utf8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(2,2,figsize=(12,9))
for ax,key,n,title in [(axs[0,0],'main20',20,'Fixed main20'),(axs[0,1],'second10',10,'Reused second10')]:
    vals=[x[key]/n*100 for x in table];ax.bar([x['group'] for x in table],vals,color=['#3975e8','#a8883a','#60a897','#e47635'])
    for i,x in enumerate(table):ax.text(i,vals[i]+2,f"{x[key]}/{n}",ha='center')
    ax.set_ylim(0,110);ax.set_title(title);ax.set_ylabel('Success (%)')
axs[1,0].plot([x['step'] for x in curves['A']],[x['train_last100'] for x in curves['A']],marker='o',label='train last100')
axs[1,0].plot([x['step'] for x in curves['A']],[x['validation'] for x in curves['A']],marker='o',label='fixed72 validation');axs[1,0].set_title('Existing A checkpoints: no retraining');axs[1,0].set_xlabel('Training step');axs[1,0].legend()
for group,color in [('H14','#60a897'),('G13_14','#e47635')]:axs[1,1].plot([x['additional_step'] for x in curves[group]],[x['loss'] for x in curves[group]],marker='o',label=group,color=color)
axs[1,1].set_title('Matched continuation validation');axs[1,1].set_xlabel('Additional step');axs[1,1].legend()
fig.tight_layout();fig.savefig(D/'comparison.png',dpi=160);plt.close(fig)
for name in ['FINAL_REPORT.md','SUFFICIENCY_REVIEW.md']:shutil.copy2(D/name,R/name)
print('V8_FINAL_AUDIT_PASS',best['group'],{g:[sum(s[g]['main20']['successes']),sum(s[g]['second10']['successes'])] for g in s},'A1000',sum(budget['A1000']['successes']),flush=True)
