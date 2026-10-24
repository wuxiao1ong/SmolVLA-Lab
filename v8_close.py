from pathlib import Path
import json,subprocess
R=Path('/home/xiaolongwu/vla_workspace/outputs/two_vlm_layers_v8');D=Path('/mnt/c/VLATest/v8_results')
s=json.loads((R/'assessment_summary.json').read_text());index=json.loads((D/'RUN_INDEX.json').read_text());review=json.loads((D/'adequacy_review.json').read_text())
assert 'V8_PIPELINE_COMPLETE' in (R/'pipeline.log').read_text() and (D/'SUFFICIENCY_REVIEW.md').exists()
procs=subprocess.check_output(['ps','-eo','args'],text=True).splitlines()
assert not [p for p in procs if p.startswith('/home/xiaolongwu/vla_workspace/env/bin/python /mnt/c/VLATest/') and any(n in p for n in ['v8_train.py','v8_eval.py','v8_validate.py','v8_pipeline.py','v8_budget_eval.py'])]
count=lambda ev:f"{sum(ev['successes'])}/{len(ev['successes'])}"
state=f'''## v8 两层VLM与数据/轮数检查已全部完成（最高优先）

用户要求再解冻一层，确认数据/轮数/论述充分性，优先已有checkpoint。G13_14与H14均从同一已有F14正式500 checkpoint及原第14层FP32 master独立开始，各1步→3步→新增500步均通过。G完整解冻13/14层，H仅14层继续；Expert+投影均继续训练，其余VLM/视觉冻结。相同batch{s['G13_14']['batch']}/workers2、Expert LR1e-5、VLM LR1e-6、seed42、warmup25/decay500。重建优化器/调度器，模型和第14层master继续，非原样optimizer恢复。250/500均保存。

结果：G13_14主20 {count(s['G13_14']['main20'])}、另10 {count(s['G13_14']['second10'])}；H14同预算单层对照主20 {count(s['H14']['main20'])}、另10 {count(s['H14']['second10'])}。A参考19/20+10/10、旧F14为18/20+10/10，未重训/重评这些基线。主20实际初态20–39/seed6000，另10为40–49/seed7000，均为反复查看开发基准，不声称新holdout或训练外泛化。

另仅验证旧A500/1000/1500 checkpoint与G/H250 checkpoint；旧A2000固定验证结果复用。仅评测旧A1000主20为{review['A1000_main20']}/20，对比A2000的19/20，没有新增这些训练。曲线在SUFFICIENCY_REVIEW.md及adequacy_review.json。

数据审计：35训练5472帧/9验证1395帧，同task24，episode无交叉、全部数值有限、轨迹136–195帧；约1%状态/动作第0维验证超train min/max，没有物体位姿数值标注，场景覆盖未知。数据足以演示单任务可学，不能证明35条充分泛化或全局最优训练量；单种子、反复开发评测和全局归一化统计重叠限制均披露。数据metadata10Hz、评测control_freq20Hz观察到但原始降采样/动作时间语义未重建，不能直接判错，中途未改协议。旧epochs_2000=0.365字段按batch1算，不用于当前24；实际等效轮数已重算。

当前推荐 {index['recommended_model']}：{index['recommended_checkpoint']}。原A/F14及G/H全部保留。最终C:/VLATest/v8_results/FINAL_REPORT.md、SUFFICIENCY_REVIEW.md、RUN_INDEX.json、results_v8.csv、paired_results.csv；evidence含六阶段训练审计、两组60回合和旧A1000的20回合。完整大模型与FP32 masters保留WSL，不上传。

WSL根 /home/xiaolongwu/vla_workspace/outputs/two_vlm_layers_v8，pipeline.log以V8_PIPELINE_COMPLETE结束，无活动训练/评测。不要重复训练。v8_report.py可重建报告，v8_close.py更新交接。旧v5/v6/v7完整保留。

'''
(R/'CURRENT_STATE.md').write_text(state,encoding='utf8');(D/'LIVE_STATUS.md').write_text(state,encoding='utf8')
p=Path('/mnt/c/VLATest/PROJECT_HANDOFF_20261021.md');text=p.read_text(encoding='utf-8-sig');a=text.index('## v8 ');b=text.index('## v7 ',a)
p.write_text(text[:a]+state+text[b:],encoding='utf8')
print('V8_CLOSED',index['recommended_model'],flush=True)
