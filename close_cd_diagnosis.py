import json, csv, shutil
from pathlib import Path
src=Path('/home/xiaolongwu/vla_workspace/outputs/cd_diagnosis_20261022')
dst=Path('/mnt/c/VLATest/diagnosis_20261022')
dst.mkdir(exist_ok=True)
labels=['A','H','G','G_reset13']
results={}
for label in labels:
    result=json.loads((src/label/'result.json').read_text())
    states=[json.loads(line) for line in (src/label/'initial_states.jsonl').read_text().splitlines()]
    audit=json.loads((src/label/'weight_audit.json').read_text())
    assert len(result['cases'])==len(states)==8
    assert [(r['state'],r['schedule']) for r in result['cases']]==[(r['state'],r['schedule']) for r in states]
    assert audit['original_checkpoint_exact'] and audit['all_other_weights_exact']
    results[label]=result
    shutil.copytree(src/label,dst/'evidence'/label,dirs_exist_ok=True)
    shutil.copy2(src/f'{label}.log',dst/'evidence'/f'{label}.log')
with (dst/'controlled_cases.csv').open('w',newline='') as f:
    writer=csv.writer(f);writer.writerow(['init_state','noise_schedule']+labels)
    for i,case in enumerate(results['A']['cases']):
        writer.writerow([case['state'],case['schedule']]+[int(results[l]['cases'][i]['success']) for l in labels])
summary={l:sum(c['success'] for c in r['cases']) for l,r in results.items()}
(dst/'summary.json').write_text(json.dumps(dict(date='2026-10-22',no_training=True,episodes_per_variant=8,selected_failure_cases=True,successes=summary,results=results),indent=2))
rows='\n'.join(f"|{r['state']}|{r['schedule']}|"+'|'.join(str(int(results[l]['cases'][i]['success'])) for l in labels)+'|' for i,r in enumerate(results['A']['cases']))
report='''# C/D 与 A-lite 差异诊断（2026-10-22）

## 首要结论：原始 C/D 并没有继承 A-lite

v5 A/C/D 的 manifest 均明确记录 `independent restart from same base; not resume`，base 指向同一个 smolvla_base revision `5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6`。原始 C/D 不是“A-lite 已训好的动作专家 + 新增 VLM 训练”。此前没有清楚区分独立 LoRA 实验和后来真正的 A 权重续训，容易造成误解。

|v5组|训练起点|实际训练范围|名义可训练参数|末100步训练Loss|主20|
|---|---|---|---:|---:|---:|
|A-lite|通用base|完整Expert + 5个状态/动作投影，VLM冻结|99,880,992|0.2098|19/20|
|B|同一通用base|Expert q/v及5投影的rank8 LoRA|371,328|0.4276|6/20|
|C|同一通用base|B范围 + VLM14 q/v LoRA|396,928|0.4323|8/20|
|D|同一通用base|B范围 + VLM13/14 q/v LoRA|422,528|0.4347|7/20|

四组都是2000步、batch24、seed42、LR1e-4、相同35训练episode。C/D的Expert MLP、k/o等大部分原权重被冻结，5个投影也是低秩更新；新增少量VLM q/v适配不能等价替代完整动作专家训练。名义参数量相差约236–252倍，但参数数目不是有效更新元素数或泛化能力的直接度量。

因此，原始差距最有证据支持的解释是动作端适配范围/当前LoRA优化配置不够有效，且没有继承A已有能力；不是已证明“在A上多训VLM导致崩溃”。C/D的训练拟合明显弱于A，且C合并后扩大Expert训练范围的E1曾从8/20升至17/20，支持动作端适配方向，但E1同时增加预算并改变学习率、合并及优化器，不能把改善唯一归因于参数量。C/D相对正确LoRA对照B反而多成功1–2回合，20回合、单训练种子不足以证明VLM有稳定增益或损害。

证据：[A manifest](../v5_results/evidence/A/manifest.json)、[C manifest](../v5_results/evidence/C/manifest.json)、[D manifest](../v5_results/evidence/D/manifest.json)、[C实际参数](../v5_results/evidence/C/trainable.json)、[v5结果](../v5_results/results_v5.csv)、[v6报告](../v6_results/FINAL_REPORT.md)。用户提供的指导文件D:/Downloads/SmolVLA_Base_v5.md第142–145行，也把A列为Expert训练、B/C/D列为LoRA配置，并非要求C继承A。

## 真正从A继续的实验，与原始C/D要分开看

|模型|权重来源与新增训练|原固定主20 + 另10|合计|
|---|---|---|---:|
|A|base → 完整Expert训练2000步|19/20 + 10/10|29/30|
|F14|A → Expert继续，完整VLM14解冻500步|18/20 + 10/10|28/30|
|H14|F14 → 相同范围再500步|20/20 + 9/10|29/30|
|G13_14|同一F14 → Expert及完整VLM13/14再500步|19/20 + 8/10|27/30|

H与G同起点、同额外步数和学习率。H并不比A总成功数更差。G比H少2回合，但G成功了H失败的45，另在39/42/49失败；模型表现是失败初态发生迁移，不是所有状态能力均下降。F失败36/38，A失败31，也体现这种差异。它们是同一批反复查看的开发初态，只有一个训练种子，不能称独立泛化结论。

已有初始权重、冻结哈希、非零有限梯度、真实更新、保存恢复和评估加载审计均通过，没有发现“VLM未训练”或“评测错加载base”的证据。模型权重续训时优化器/调度器重建，H/G处理一致；这不是原优化器状态原样恢复。

证据：[逐初态对照](../v8_results/paired_results.csv)、[真实checkpoint索引](../v8_results/RUN_INDEX.json)、[权重更新审计](../v8_results/weight_update_audit.json)。

## 本次新增：固定推理噪声与第13层恢复消融

没有新增训练，没有覆盖任何checkpoint。选取原G失败39/42/49和原H失败45，共4个初态、2套预设噪声，每个版本8回合，总32回合。A/H/G使用原checkpoint；G_reset13仅在内存将G完整第13层的9个张量恢复到A值，所有其他权重逐项保持G值。保存了加载及恢复审计。这是保持G其他部分固定的局部干预，不是重新构造H。

环境种子保持原对应值（39→6019，42/45/49→7002/7005/7009），任务Object0、relative、20Hz、chunk50、fixed10均保持。每个动作块显式生成共同高斯噪声，seed=`schedule*100000+state*100+chunk_index`，schedule为8100/9100。这样不同轨迹长短不会改变后续回合的噪声序列。原评测只设置整体seed，未做到每个初态、每个动作块的噪声配对；不能把原来少数回合差异完全归于训练层数。

|初态|噪声schedule|A|H14|G13_14|G恢复13|
|---:|---:|---:|---:|---:|---:|
'''+rows+f'''

合计：A {summary['A']}/8，H {summary['H']}/8，G {summary['G']}/8，G恢复13 {summary['G_reset13']}/8。这些初态是根据已有失败选出的诊断样本，不能作为新总体成功率，不能与原30回合直接合并。

原G在39/42/49的失败，在本次两套噪声下均未复现。这个结果说明原少数失败具有条件依赖性；不能凭原27/30就确定“多一层一定退步”，也不能把过拟合、表征漂移或灾难性遗忘写成已经定位的原因。第13层恢复结果只检验当前选定状态/噪声中的局部影响；恢复不能抵消联合训练时其他层已形成的适应，阴性结果也不能证明第13层普遍无影响。

原始日志、每回合实际初态、权重审计在[evidence](evidence)，逐例结果在[controlled_cases.csv](controlled_cases.csv)，脚本C:/VLATest/diagnosis_eval.py和run_cd_diagnosis.sh。

## 数据、训练量与下一步判断

训练task24文本与Object0一致，35条训练5472帧、9条验证1395帧按episode分开，动作/相机映射已核查；A/H已有29/30，现有证据不支持“训练数据选成了另一个任务”作为组间差距主因。35条能让当前单任务可学，但场景覆盖、训练与测试初态的对应未知，不能保证泛化充分。数据metadata10Hz与当前模拟control_freq20Hz差异已披露，原转换时间语义尚未完全重建；各组共用协议，不能单独解释C/D相对A的差距。

A训练Loss下降时固定验证Loss并未同步下降；E1离线验证优于A但新初态闭环更差。不能按训练Loss或参数更多就认定应继续增加层数、轮数。新增VLM可能改变动作专家依赖的表示，少量示范上的训练也可能使策略偏向特定轨迹，但这些目前是待隔离的机制假设。

当前保留A为推荐、H为同成功数候选，G保留为实验版本。若继续改进，先用未参与调参的初态和每初态多套固定噪声验证A/H/G的稳定差异，再针对真实失败阶段补充恢复轨迹；只有出现可复现的VLM相关退步，才值得用更低VLM学习率、限制更新范围或保持A输出的约束做同源对照。当前不支持为了参数更多而盲目重训C/D或再解冻更多层。
'''
(dst/'DIAGNOSIS_REPORT.md').write_text(report,encoding='utf-8')
handoff=Path('/mnt/c/VLATest/PROJECT_HANDOFF_20261021.md')
text=handoff.read_text(encoding='utf-8')
note='''## 2026-10-22 C/D落后原因诊断已完成（最高优先）

原v5 C/D不是从A-lite继续，A/B/C/D均从相同smolvla_base独立开始；A完整Expert+5投影99.88m，C/D为Expert q/v与投影及VLM q/v的rank8 LoRA仅0.397m/0.423m。原主20 A19、B6、C8、D7；不能解释为在A上多训VLM把A破坏。真正A续训为F14→H/G：A29/30，F28/30，H29/30，G27/30，仅开发小样本和单种子。

本轮无新增训练，32回合固定初态/逐动作块共同噪声诊断（A/H/G/G仅内存恢复第13层，各8回合）。原G失败39/42/49在两套新固定噪声下均成功，不支持已确定第13层导致原少数失败。结果与边界见C:/VLATest/diagnosis_20261022/DIAGNOSIS_REPORT.md、controlled_cases.csv及evidence；脚本diagnosis_eval.py、run_cd_diagnosis.sh、close_cd_diagnosis.py。WSL根outputs/cd_diagnosis_20261022。各组真实加载和恢复精确核验，原checkpoint均保留，推荐A不变。不要重复已完成训练或诊断。

'''
marker='## v8 两层VLM与数据/轮数检查已全部完成（最高优先）'
assert marker in text
handoff.write_text(text.replace(marker,note+marker,1),encoding='utf-8')
print(json.dumps(summary));print(dst/'DIAGNOSIS_REPORT.md')
