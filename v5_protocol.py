from pathlib import Path
import json
root=Path('/home/xiaolongwu/vla_workspace/outputs/base_single_task_v5')
(root/'PROTOCOL.md').write_text('''# v5 Base单任务训练协议\n\n遵循用户提供的SmolVLA_Base_v5.md。按用户指示跳过官方任务专用权重验证；已存在同入口、同初态10–19的8/10结果。训练始终从固定smolvla_base出发，不改用smolvla_libero。原v4与recovery输出保留。\n\n数据：task24/Object0，44条示范，固定35训练（5472帧）/9验证（1395帧）；完整episode映射和索引在metadata。图像键和状态/动作语义沿用通过的原生处理。全局已有归一化统计覆盖验证集合，该限制如实记录。\n\n顺序：A-lite全专家与状态/动作投影参数训练，先1步、3步验证后2000步；根据真实验证与闭环决定扩展5000/10000，未建立可用A-lite前不把BCD差异当作性能结论。之后B→C→D，保持相同数据/起点/有效batch/正式步数；每组先1/3步再正式。\n\n用户追加授权增大batch：已有batch1 A2000保留作对照，另做batch4/8/16容量与吞吐检查，按实测结果选择统一batch；如果16仍有大量余量可进一步测试，但以吞吐与安全余量决定，不为占满显存盲目加参数。改变batch时从同一base独立重启，不冒充续训；通过批量与帧覆盖记录解释预算变化。LoRA rank8/alpha16固定。\n\n精度为原生混合dtype、AMP关闭，不声称全FP32。全部真实训练日志包含LR、实际更新、CUDA allocated/reserved。所有新目录唯一，push_to_hub=false。\n\n评测固定Object0/fixed10、chunk50、相机映射、实际初态；先功能检查及10回合。发现非零后用20–50回合进一步确认，记录实际初态和Wilson区间；模拟初态与示范的对应未知，不宣称训练外泛化。\n''',encoding='utf8')
print('BATCH_PROTOCOL_SAVED')
