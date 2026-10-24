import os,json,shutil
from pathlib import Path
root=Path('/home/xiaolongwu/vla_workspace/vla_interview_project/vla_training_ablation_v4')
run=sorted((root/'runs').glob('B_formal*'))[-1];ckpt=Path(json.loads((run/'result.json').read_text())['checkpoint'])
base=root/'base_eval';base.mkdir(exist_ok=False)
cfg=json.loads((ckpt/'config.json').read_text());cfg['use_peft']=False
(base/'config.json').write_text(json.dumps(cfg,indent=2))
for p in ckpt.glob('policy_*'):shutil.copy2(p,base/p.name)
src=Path(json.loads((run/'manifest.json').read_text())['base'])/'model.safetensors'
(base/'model.safetensors').symlink_to(src)
(root/'PROTOCOL.md').write_text('''# v4 已锁定协议\n\n执行顺序：B 1步 → B 3步 → B 500步 → C 1步 → C 3步 → C 500步 → D 1步 → D 3步 → D 500步。每阶段独立从相同 base revision 开始，检查失败即停，不将检查权重续训到正式实验。\n\n基座 smolvla_base revision 5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6。LoRA rank8/alpha16，float32，batch1，workers0，seed42，LR1e-4，正式 warmup25/decay500，双相机512 padding，chunk50。保持手册 B/C/D 模块定义。训练入口只增加审计与数据特征适配，未改上游源码。\n\n全部数据为 task24/Object0，44 episodes/6867 frames。用 Python Random(42) shuffle，前9条留出，后35条正式训练；1/3步用训练集合前8条。完整划分见每次运行 manifest.json，数据 revision 与 SHA256 见 audit/dataset.json。使用数据集已有全局归一化统计（包含留出数据），离线验证不是完全隔离的泛化评估。\n\n训练三组种子相同，但注入模块数不同会消耗不同 RNG，不能声称 minibatch 顺序、LoRA 初值和 flow noise 完全逐项配对。三组起始预训练权重、数据集合、步数、超参数相同。\n\n训练完成后统一原生 fixed10 评测，无推理 rerank/adaptive 控制器。Object0，seed5000，每组1回合 smoke(init0)，再10回合(init10–19)；记录真实 init_state_id。base 对照用相同输入特征与训练所用归一化处理器、未注入 LoRA 的原始基座权重。演示与模拟初态对应关系未知，因此不能把 init10–19 称为训练外独立初态。单任务单训练种子仅作有限预算消融。旧 A 是不同起点/预算条件，只作历史参考。\n\n每步检查所有目标梯度有限，每类别非零梯度且参数更新；初步 LoRA A 梯度为零符合 B 初始化为零的预期。训练显存记录覆盖 backward 和 optimizer.step。保存后通过 make_policy 恢复，适配器权重逐项完全相等；评测进程额外核对实际加载的 adapter state_dict。所有失败尝试保留。\n''',encoding='utf8')
print(base)
