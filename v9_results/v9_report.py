import json,csv,shutil,collections
from pathlib import Path
import torch
from safetensors.torch import load_file
root=Path('/home/xiaolongwu/vla_workspace/outputs/alite_vlm_lora_v9')
out=Path('/mnt/c/VLATest/v9_results');out.mkdir(exist_ok=True)
labels=['A','C_A14','D_A13_14']
training=json.loads((root/'training_summary.json').read_text())
evaluations={l:json.loads((root/'eval'/l/'result.json').read_text()) for l in labels}
initial={l:[json.loads(x) for x in (root/'eval'/l/'initial_states.jsonl').read_text().splitlines()] for l in labels}
noise={l:[json.loads(x) for x in (root/'eval'/l/'noise_audit.jsonl').read_text().splitlines()] for l in labels}
plan=json.loads((root/'PLAN.json').read_text())
for label in labels:
    assert len(evaluations[label]['cases'])==len(initial[label])==len(noise[label])==40
    assert evaluations[label]['cases']==[dict(state=s,schedule=n,success=evaluations[label]['cases'][i]['success']) for i,(s,n) in enumerate(( (s,n) for n in plan['eval_schedules'] for s in plan['eval_states']))]
    assert initial[label]==initial['A'], ('initial observations differ',label)
    assert noise[label]==noise['A'], ('noise differs',label)
    audit=json.loads((root/'eval'/label/'weight_audit.json').read_text());assert audit['all_A_weights_exact']
    shutil.copytree(root/'eval'/label,out/'evidence'/'eval'/label,dirs_exist_ok=True)
counts={l:sum(c['success'] for c in evaluations[l]['cases']) for l in labels}
by_schedule={l:{str(n):sum(c['success'] for c in evaluations[l]['cases'] if c['schedule']==n) for n in plan['eval_schedules']} for l in labels}
paired={}
for label in labels[1:]:
    paired[label]={kind:[dict(state=a['state'],schedule=a['schedule']) for a,b in zip(evaluations['A']['cases'],evaluations[label]['cases']) if (not a['success'] and b['success'] if kind=='recovered_A_failures' else a['success'] and not b['success'])] for kind in ['recovered_A_failures','lost_A_successes']}
with (out/'paired_results.csv').open('w',newline='') as f:
    writer=csv.writer(f);writer.writerow(['init_state','noise_schedule']+labels)
    for i,case in enumerate(evaluations['A']['cases']):writer.writerow([case['state'],case['schedule']]+[int(evaluations[l]['cases'][i]['success']) for l in labels])
validation={}
for label in labels[1:]:
    p=sorted((root/'validation').glob(label+'_500_*.json'))[-1];v=json.loads(p.read_text());validation[label]=v['mean_loss']
    (out/'validation').mkdir(exist_ok=True);shutil.copy2(p,out/'validation'/p.name)
validation['A']=json.loads(Path('/mnt/c/VLATest/v5_results/evidence/A/validation.json').read_text())['mean_loss']
index={'A_reference':plan['base'],'recommended_checkpoint':plan['base'],'recommendation':'retain A as default; this is a small repeated development benchmark, no automatic deployment or global improvement claim'}
best=max(labels,key=lambda label:counts[label])
index['best_development_model']=best
index['best_development_checkpoint']=plan['base'] if best=='A' else training[best]['checkpoint']
for label in labels[1:]:
    run=Path(training[label]['run']);ckpt=Path(training[label]['checkpoint'])
    assert json.loads((run/'frozen_weights_verified.json').read_text())['all_original_weights_unchanged']
    assert json.loads((run/'zero_adapter_equivalence.json').read_text())['exact_loss']
    for name in ['policy_preprocessor_step_5_normalizer_processor.safetensors','policy_postprocessor_step_0_unnormalizer_processor.safetensors']:
        original_stats=load_file(str(Path(plan['base'])/name));actual_stats=load_file(str(ckpt/name))
        assert set(original_stats)==set(actual_stats) and all(torch.equal(value,actual_stats[key]) for key,value in original_stats.items()), (label,name)
    index[label]=str(ckpt)
    shutil.copytree(ckpt,out/'models'/label,dirs_exist_ok=True)
(out/'processor_audit.json').write_text(json.dumps(dict(normalizer_and_unnormalizer_exact_A=True,groups=labels[1:]),indent=2))
for p in root.iterdir():
    if p.is_dir() and (p/'manifest.json').exists():shutil.copytree(p,out/'evidence'/'training'/p.name,dirs_exist_ok=True)
    elif p.is_file():
        (out/'logs').mkdir(exist_ok=True);shutil.copy2(p,out/'logs'/p.name)
for name in ['v9_train.py','v9_pipeline.py','v9_validate.py','v9_eval.py','v9_status.py','v9_report.py','v9_action_delta.py']:
    shutil.copy2(Path('/mnt/c/VLATest')/name,out/name)
(out/'RUN_INDEX.json').write_text(json.dumps(index,indent=2))
action_delta=json.loads((root/'action_delta.json').read_text())
(out/'assessment_summary.json').write_text(json.dumps(dict(counts=counts,by_schedule=by_schedule,paired_to_A=paired,validation=validation,training=training,action_delta=action_delta,initial_observations_exact_across_models=True,first_chunk_noise_exact_across_models=True),indent=2))
with (out/'results_v9.csv').open('w',newline='') as f:
    writer=csv.writer(f);writer.writerow(['label','source','extra_steps','trainable_params','seed','batch','lr','noise12300','noise12400','success','episodes','val_loss','peak_allocated_gib'])
    for label in labels:
        train=training.get(label,{})
        writer.writerow([label,plan['base'],0 if label=='A' else 500,0 if label=='A' else (25600 if label=='C_A14' else 51200),42,24,0 if label=='A' else .0001,by_schedule[label]['12300'],by_schedule[label]['12400'],counts[label],40,validation[label],train.get('peak_allocated_gib','')])
rows='\n'.join(f"|{l}|{0 if l=='A' else 500}|{0 if l=='A' else (25600 if l=='C_A14' else 51200):,}|{by_schedule[l]['12300']}/20|{by_schedule[l]['12400']}/20|{counts[l]}/40|{validation[l]:.6f}|" for l in labels)
changes='\n'.join(f"- {label}：恢复A失败 {len(paired[label]['recovered_A_failures'])} 回合，新增失败 {len(paired[label]['lost_A_successes'])} 回合；明细见assessment_summary.json。" for label in labels[1:])
report=f'''# v9：从训练好的A-lite添加VLM LoRA

全部完成：C单层→D双层，每组仅1步检查→正式500步；用户明确跳过3步，本轮没有运行3步检查。原A不重训，官方smolvla_libero不重跑，旧checkpoint均保留。

## 结果

|模型|新增步数|新增可训练参数|噪声12300|噪声12400|合计|固定72帧验证Loss|
|---|---:|---:|---:|---:|---:|---:|
{rows}

本轮相对A的净成功数变化：单层 {counts['C_A14']-counts['A']:+d}/40，双层 {counts['D_A13_14']-counts['A']:+d}/40。开发评测最高成功数模型为 {best}（并列时优先保留A）。这是本次固定条件下的实测排序，不能据此认定稳定的总体提升。

本次各模型是相同20初态×2套噪声，共40回合；不是40个独立场景，也不是新的盲测集合。不能把这里的成功率直接与旧A19/20、H29/30等不同噪声协议的分数相减，声称训练提升。

{changes}

## 本次真正检验的变化

共同基座是已训练2000步的A-lite：`{plan['base']}`。不是从smolvla_base重新开始，也不是原v5 Expert LoRA C/D。C_A14仅在VLM text_model.layers.14的q/v添加LoRA；D_A13_14仅在13/14层q/v添加LoRA，层号从0开始。完整A动作专家、5投影、视觉编码器、所有VLM原权重都冻结。

两组独立从同一A开始，新LoRA rank8、alpha16、dropout0，batch24/workers2，LR1e-4，warmup25/decay500，seed42，保存250/500。正式训练不继承1步检查的权重。只训练VLM低秩适配器，不能把结果解释为完整Expert继续训练或整层VLM解冻的效果。新建优化器/调度器，不声称恢复A的优化器状态。

## 权重及训练验收

四个成功训练阶段均验证：包装LoRA前全部原权重与A逐张量相等，adapter_config基座准确指向A；LoRA初始B为零，固定随机数下启用/禁用适配器的损失完全一致；优化器只包含选定VLM LoRA；梯度有限、每个目标层有非零梯度和更新；训练前后全部原权重hash不变；保存重载的adapter及冻结权重精确一致。第一步LoRA A矩阵梯度为零符合B零初始化，不能误判未训练；B矩阵已非零更新。

已有固定验证中的18个观察点采用相同显式动作噪声，C/D各18个首动作均与A出现数值差异，确认更新影响了实际动作输出。每维变化保存在logs/action_delta.json；它只说明输出发生变化，不直接说明动作质量提高。

两个早期C检查在训练0步时退出：当前CLI不支持lora_dropout参数；默认get_optim_params包含冻结参数，触发了本包装的严格优化器集合断言。已移除不支持参数并核验默认dropout0，显式只传LoRA参数给优化器。修正后的1步检查通过才启动正式训练。错误日志及0步结果保留，没有把失败尝试算作完成训练。

## 对照如何保证一致

本次A/C/D均重新评测，相同Object0任务、初态20–39、每初态种子6000+state-20、relative、20Hz、chunk50、fixed10。两套噪声schedule12300/12400在训练结果出来前写入PLAN.json；每个动作块显式噪声seed为schedule×100000+state×100+chunk_index，与前面回合长短无关。对PEFT模型在实际底层SmolVLA的_get_action_chunk注入噪声，避免只改包装器而实际未使用。

各模型40次初始观察hash完全一致，40条首次动作块噪声hash完全一致，且逐块都按同一公式生成。评测再次验证所有原A权重不变、适配器逐张量精确加载；没有做BF16合并，保留未合并FP32 LoRA分支。逐回合结果、实际初态、噪声及位置轨迹均保存。

训练的episode采样顺序由LeRobot EpisodeAwareSampler(seed42)控制；两组LoRA参数初始化数量不同，训练flow噪声未逐步强制对齐。本轮只有一个训练种子，仍不能把所有差异唯一归因于多出的第13层。推理噪声是逐块强制配对的。

## 数据与结论边界

同一task24的35条训练5472帧、9条验证1395帧，episode不交叉。每组额外12000帧抽样约2.19等效轮，重叠动作窗不是独立示范。沿用A的全局归一化统计，验证统计重叠；训练场景与模拟初态对应未知。数据metadata10Hz与评测20Hz的时间语义限制沿用并披露，本轮没有改变协议。

本实验能回答“在当前A权重和500步配置下，冻结动作端、只给一/两层VLM加q/v LoRA的开发评测表现”；不能证明数据规模充分、训练已最优、LoRA普遍优劣或未见场景泛化。原A继续作为默认推荐，适配器作为候选保留；扩大独立场景和训练种子后再决定替换。

## 文件

- results_v9.csv、paired_results.csv、assessment_summary.json：汇总及配对明细。
- RUN_INDEX.json：原A和两组正式adapter的WSL路径。加载adapter必须保留它引用的A-lite基座，不能改回通用base。
- models/C_A14、models/D_A13_14：正式适配器及处理器配置的小文件副本；没有复制巨大A权重，没有上传。
- evidence/training、evidence/eval：检查/训练审计、全部120回合原始结果与轨迹。
- WSL根：{root}。训练和评估均已结束，不应重复启动pipeline。
'''
(out/'FINAL_REPORT.md').write_text(report,encoding='utf-8')
handoff=Path('/mnt/c/VLATest/PROJECT_HANDOFF_20261021.md')
handoff_text=handoff.read_text(encoding='utf-8')
start=handoff_text.index('## v9 在A-lite基础上做VLM LoRA')
end=handoff_text.index('## 2026-10-22 C/D落后原因诊断已完成',start)
completed=f'''## v9 在A-lite基础上做VLM LoRA已全部完成（最高优先）

用户明确跳过3步检查。本轮C_A14、D_A13_14分别从原A2000独立新增VLM14或13/14的q/v LoRA，rank8/alpha16/LR1e-4，全部A原权重（含Expert/投影）冻结。batch24/workers2/seed42，各1步→独立正式500步全部完成，保存250/500。无3步、无重训A、无官方对照。正式训练不继承1步权重。

共同噪声评测已完成120回合：初态20–39×schedule12300/12400，每组40回合，A {counts['A']}/40，C_A14 {counts['C_A14']}/40，D_A13_14 {counts['D_A13_14']}/40。全部40个初始观察hash和首块噪声hash跨模型完全一致；每动作块显式按同一公式生成噪声。旧开发初态、单训练种子，不声称新holdout或训练外泛化，不与旧A19/20等不同噪声协议直接比较。

全部初始A权重、零适配器等价、有限梯度/真实更新、原权重冻结hash、保存恢复、评测再次加载校验通过。C两次早期检查在0步退出（CLI参数/严格优化器断言），修正后才完成门槛，日志保留。默认仍保留A推荐，候选adapter不合并、不覆盖原模型。

最终C:/VLATest/v9_results/FINAL_REPORT.md、results_v9.csv、paired_results.csv、assessment_summary.json、RUN_INDEX.json；models含适配器小文件，evidence含全部训练和评测。WSL根/home/xiaolongwu/vla_workspace/outputs/alite_vlm_lora_v9。C:/VLATest/v9_*.py可追溯。训练和评测均已结束，不要重复启动pipeline。

'''
handoff.write_text(handoff_text[:start]+completed+handoff_text[end:],encoding='utf-8')
print('V9_REPORT_COMPLETE',json.dumps(counts),out/'FINAL_REPORT.md',flush=True)
