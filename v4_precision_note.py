from pathlib import Path
root=Path('/home/xiaolongwu/vla_workspace/vla_interview_project/vla_training_ablation_v4')
p=root/'PROTOCOL.md';s=p.read_text();s=s.replace('float32','AMP关闭、原生混合dtype（VLM以BF16加载，PEFT适配器默认FP32；实际计数见audit/dtypes_*.json）');p.write_text(s)
(root/'audit/PRECISION_ERRATUM.md').write_text('''# 精度元数据更正\n\n原始训练 manifest 的 precision=float32 标注有误。实验代码仅设 use_amp=false，未执行 model.float()。LeRobot 原生 SmolVLMWithExpertModel 在构建时使用 torch_dtype=bfloat16，PEFT 默认提升适配器精度。所有组配置始终相同，未改变任何训练张量或重跑覆盖结果。准确的运行时 dtype 计数由同一原生工厂恢复 checkpoint 后记录到 dtypes_base/B/C/D.json。原始 manifest 保留，防止篡改原始记录；报告以此更正为准。\n''')
print('PRECISION_ANNOTATION_CORRECTED')
