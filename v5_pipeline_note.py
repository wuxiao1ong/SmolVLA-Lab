from pathlib import Path
p=Path('/home/xiaolongwu/vla_workspace/outputs/base_single_task_v5/CURRENT_STATE.md')
p.write_text(p.read_text()+'''\n## 新增：串行后续流程已启动\n\n/mnt/c/VLATest/v5_pipeline.py 正在运行，unified exec session76788，日志本目录pipeline.log。当前WAIT_A，不占GPU；A完成且恢复验收后自动验证72帧、1回合排障、10回合discovery；若A discovery为0/10会停止并写pipeline_needs_diagnosis.json，不盲目开始BCD。若非零，会做main20(init20–39 seed6000)，然后严格B→C→D各1/3/2000步batch24/workers2，同样验证、1回合排障和main20。同协议20回合是主要公平比较，A额外10回合是诊断用途。assessment_summary.json持续保存。新用户更改计划时务必先检查A进程和pipeline进程，不能重复启动。1500步原生功能检查也成功1/1。\n''',encoding='utf8')
print('PIPELINE_STATE_SAVED')
