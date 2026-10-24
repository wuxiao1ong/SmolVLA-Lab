from pathlib import Path
import json,subprocess
R=Path('/home/xiaolongwu/vla_workspace/outputs/one_vlm_layer_v7')
D=Path('/mnt/c/VLATest/v7_results')
s=json.loads((R/'assessment_summary.json').read_text());index=json.loads((D/'RUN_INDEX.json').read_text())
assert 'V7_PIPELINE_COMPLETE' in (R/'pipeline.log').read_text() and (D/'FINAL_REPORT.md').exists()
procs=subprocess.check_output(['ps','-eo','args'],text=True).splitlines()
assert not [p for p in procs if p.startswith('/home/xiaolongwu/vla_workspace/env/bin/python /mnt/c/VLATest/') and any(n in p for n in ['v7_pipeline.py','v7_train.py','v7_eval.py'])]
count=lambda e:f"{sum(e['successes'])}/{len(e['successes'])}"
state=f'''## v7 解冻单层VLM已全部完成（最高优先）

用户要求尽量利用已有checkpoint、减少训练量。已从A-lite正式2000步checkpoint直接继续，完整Expert+5投影可训练，额外解冻完整VLM text_model.layers.14（从0计数，倒数第二层），其余VLM/视觉冻结；不是仅q/v LoRA。1步→3步→正式追加500步均通过，250/500步保存，未追加其他训练。

F14主20初态20–39/seed6000：{count(s['main20'])}；另10初态40–49/seed7000：{count(s['second10'])}。复用A-lite已完成对照19/20、10/10，避免重复训练/评测A。两组初态前轮均已查看，是复用开发基准，不称新holdout或训练外泛化。

batch{s['batch']}/workers2，Expert LR1e-5、VLM14 LR1e-6、warmup25/decay500、seed42。原生模型BF16/FP32，只有VLM14用FP32 master累计小更新后舍入回原生权重。9个VLM张量有非零梯度、7个注意力/MLP原生张量实际更新；两个归一化权重未跨BF16舍入，master舍入与模型保存结果逐项一致。每个checkpoint旁保存vlm14_fp32_master.safetensors，v7_train.py支持V7_START并自动核验恢复旁边master；优化器/调度器重建，非原样optimizer恢复。本轮三个阶段，每阶段均核验初始权重、冻结哈希及最终可训练张量原生恢复；评测dense加载精确一致。

当前推荐 {index['recommended_model']}，路径 {index['recommended_checkpoint']}。A原权重与本轮F14均完整保留；本轮checkpoint {s['checkpoint']}。

最终报告 C:/VLATest/v7_results/FINAL_REPORT.md；结果results_v7.csv、paired_results.csv，模型RUN_INDEX.json；evidence含三阶段审计和30回合结果/真实初态/视频。大模型保留WSL，不上传。WSL根 /home/xiaolongwu/vla_workspace/outputs/one_vlm_layer_v7，pipeline.log以V7_PIPELINE_COMPLETE结束，无活动训练/评测。不要重跑已完成训练；v7_report.py可重建报告，v7_close.py更新交接。旧v5/v6保留。

'''

(R/'CURRENT_STATE.md').write_text(state,encoding='utf8');(D/'LIVE_STATUS.md').write_text(state,encoding='utf8')
p=Path('/mnt/c/VLATest/PROJECT_HANDOFF_20261021.md');text=p.read_text(encoding='utf-8-sig')
a=text.index('## v7 ');b=text.index('## v6 ',a)
p.write_text(text[:a]+state+text[b:],encoding='utf8')
print('V7_CLOSED',index['recommended_model'],count(s['main20']),count(s['second10']),flush=True)

