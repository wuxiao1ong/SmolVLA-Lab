from pathlib import Path
import json, shutil

root = Path('/home/xiaolongwu/vla_workspace/outputs/base_single_task_v5')
out = Path('/mnt/c/VLATest/v5_results')
summary = json.loads((root/'assessment_summary.json').read_text())
assert set(summary) == set('ABCD')
assert 'V5_PIPELINE_COMPLETE' in (root/'pipeline_attempt2.log').read_text()
for group, expected in zip('ABCD', [19, 6, 8, 7]):
    ev = summary[group]['main20']
    assert len(ev['successes']) == 20 and sum(ev['successes']) == expected
    for stage in ['check1', 'check3', 'formal2000']:
        runs = sorted(root.glob(f'{group}_{stage}_bs24_w2_*'))
        assert runs, (group, stage)
        assert json.loads((runs[-1]/'result.json').read_text())['status'] == 'passed'
shutil.copy2(root/'paired_main20.csv', out/'paired_main20.csv')
shutil.copy2(out/'results_v5.csv', root/'results_v5.csv')
shutil.copy2(out/'FINAL_REPORT.md', root/'FINAL_REPORT.md')
state = '''# v5 最终状态：全部完成

A-lite基线完成后，严格B→C→D，每组1步→3步→2000步通过。统一batch24/workers2、LR1e-4、训练seed42，从smolvla_base出发，跳过官方模型重复验证。主比较同一实际初态20–39、seed6000、fixed10。

最终：A-lite19/20（95%），B6/20（30%），C8/20（40%），D7/20（35%）。A峰值allocated6.10/reserved6.29GiB。B/C/D全部目标模块有非零梯度和权重更新，独立评测adapter精确一致。A名义155可训练张量中122改变，33个原生BF16 RMSNorm未改变，已在报告披露。

pipeline_attempt2.log以V5_PIPELINE_COMPLETE结束，无待执行训练。不要重复运行。最高实测成功率优先A-lite；单种子20回合不能断言VLM增层稳定获益。数据任务匹配；旧500步batch1预算与适配范式是需关注因素，但未隔离唯一原因。

Windows最终报告 C:/VLATest/v5_results/FINAL_REPORT.md，表results_v5.csv，权重路径RUN_INDEX.json，逐回合paired_main20.csv。evidence四组完整记录及视频，models/B|C|D最终adapter。A完整权重仍在WSL。旧v4/recovery/batch1训练保留。主评测每组20结果完整，默认最多10视频。验证统计量、初态对应未知及单种子限制见最终报告。
'''
(root/'CURRENT_STATE.md').write_text(state, encoding='utf8')
(out/'LIVE_STATUS.md').write_text(state, encoding='utf8')
(out/'PROGRESS_REPORT.md').write_text('# 阶段报告已收尾\n\n全部完成，最终权威报告见 [FINAL_REPORT.md](C:/VLATest/v5_results/FINAL_REPORT.md)。\n', encoding='utf8')
print('FINAL_CHECK_PASS: 12 training stages, 80 main episodes, all artifacts finalized')
