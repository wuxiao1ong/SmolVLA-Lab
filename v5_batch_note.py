from pathlib import Path
import json
root=Path('/home/xiaolongwu/vla_workspace/outputs/base_single_task_v5');run=sorted(root.glob('A_check1_bs24_w2_*'))[-1]
print((run/'result.json').read_text());print((run/'steps.jsonl').read_text()[:300])
(root/'metadata/visual_audit.md').write_text('''# 数据视觉核查\n\n已阅读两个44条示范末帧拼图，所有44条轨迹末帧均与目标物放入左侧篮子的结果一致；该检查是视觉一致性，不替代模拟器success标志。ep813首/中/末帧双相机视图与评测图像方向一致，未见相机交换或180度翻转差异。task24状态范围与模拟器同场景相符，首帧末端z约0.26，对照环境约0.25；没有根据跨任务全局均值误判坐标系偏移。\n\n任务文本通过数据集task24与LIBERO Object0精确匹配。归一化张量与已成功官方模型一致。此前v4失败视频主要出现偏离目标、抬升与侧向漂移；这些证据支持优先检查训练适配程度而不是修改测试任务。\n''',encoding='utf8')
