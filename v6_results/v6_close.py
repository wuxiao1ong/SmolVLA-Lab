from pathlib import Path
import json,subprocess
r=Path('/home/xiaolongwu/vla_workspace/outputs/expert_continuation_v6')
d=Path('/mnt/c/VLATest/v6_results')
s=json.loads((r/'assessment_summary.json').read_text())
first=json.loads((r/'E1_assessment_summary.json').read_text());second=json.loads((r/'E2_assessment_summary.json').read_text())
assert 'V6_EXTENSION_COMPLETE' in (r/'extension.log').read_text()
assert (d/'FINAL_REPORT.md').exists() and 'confirmation10' in s
procs=subprocess.check_output(['ps','-eo','args'],text=True).splitlines()
active=[p for p in procs if p.startswith('/home/xiaolongwu/vla_workspace/env/bin/python /mnt/c/VLATest/') and any(n in p for n in ['v6_train.py','v6_eval.py','v6_extend.py'])]
assert not active,active
name='E1' if s['checkpoint']==first['checkpoint'] else 'E2'
count=lambda e:f"{sum(e['successes'])}/{len(e['successes'])}"
state=f'''## v6 改进实验已全部完成（优先）

用户授权从已有checkpoint改进成功率。已从v5 C2000（8/20）安全合并后，完整训练Expert+5投影、冻结视觉/VLM；不再是纯LoRA。E1追加1000步主20为{count(first['main20'])}，E2从E1再追加1000步主20为{count(second['main20'])}。每阶段独立1步→3步→正式1000步均通过。batch24/workers2，LR5e-5，warmup50/decay1000，seed42，峰值allocated/reserved约6.10/6.29GiB。原模型及所有checkpoint保留。

按开发主20成功数选定{name}；选定后在预先固定的新初态40–49/seed7000上评测{count(s['confirmation10'])}，同样新初态A-lite为{count(s['A_confirmation10'])}。原20初态20–39/seed6000已经用于训练方案开发和选模，有选择偏差；新10未用于后续调参，示范与模拟初态对应未知，不声称训练外泛化。E1累计C2000+1000，E2累计C2000+2000，不能与BCD声称相同预算。优化器重建、模型权重续训，非原样优化器恢复；BF16合并舍入及名义参数与实际更新限制已披露。

六训练阶段初始权重/冻结哈希/可训练张量恢复与四次正式评测dense权重均核验通过。extension.log以V6_EXTENSION_COMPLETE结束，没有活动训练/评测。不要重复启动v6_pipeline.py或v6_extend.py。

最终报告 C:/VLATest/v6_results/FINAL_REPORT.md，结果results_v6.csv，逐回合paired_main20.csv，模型RUN_INDEX.json，六阶段及60新评测回合证据在evidence。选定checkpoint：{s['checkpoint']}。

WSL根 /home/xiaolongwu/vla_workspace/outputs/expert_continuation_v6；全部大权重仍保留WSL，不上传。v6_finalize_report.py（v6_report.py转发）可重建报告；v6_close.py核对无活动进程后更新交接。旧v5/v4与recovery保留。

'''
state=state.replace('按开发主20成功数选定', '当前推荐A-lite：原主20为19/20，新10为10/10，均高于续训模型。RUN_INDEX.json的recommended_checkpoint为A-lite，selected_continuation为本轮E1；不替换最佳模型。\n\n按开发主20成功数选定')
(r/'CURRENT_STATE.md').write_text('# v6 最终交接\n\n'+state,encoding='utf8')
(d/'LIVE_STATUS.md').write_text(state,encoding='utf8')
p=Path('/mnt/c/VLATest/PROJECT_HANDOFF_20261021.md');text=p.read_text(encoding='utf-8-sig')
a=text.index('## v6 改进实验');b=text.index('## v5 已全部完成',a)
p.write_text(text[:a]+state+text[b:],encoding='utf8')
print('V6_CLOSED',name,count(s['main20']),count(s['confirmation10']),count(s['A_confirmation10']),flush=True)
