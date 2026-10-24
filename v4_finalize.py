from pathlib import Path
import json,math,statistics,csv,shutil
root=Path('/home/xiaolongwu/vla_workspace/vla_interview_project/vla_training_ablation_v4');out=Path('/mnt/c/VLATest/v4_results')
rows=json.loads((out/'summary.json').read_text());base=sorted((root/'eval').glob('base_formal*'))[-1];bresult=json.loads((base/'result.json').read_text());bval=json.loads((root/'validation_base.json').read_text())
lines=['# v4 LoRA 消融实测报告','', '执行日期：2026-10-21（香港时间）。已严格按 B→C→D，各组 1步→3步→500步完成；全部通过，无 OOM。闭环与离线验证也已完成。','', '## 正式训练','', '|组别|目标模块|可训练参数|峰值 allocated / reserved GiB|训练循环秒|首50步均值 Loss|末50步均值 Loss|', '|---|---|---:|---:|---:|---:|---:|']
for r in rows:
 target={'B':'Expert q/v + 5个状态/动作投影','C':'B + VLM14 q/v','D':'B + VLM13/14 q/v'}[r['group']]
 lines.append(f"|{r['group']}|{target}|{r['trainable']:,}|{r['peak_allocated']:.4f} / {r['peak_reserved']:.4f}|{r['train_seconds']:.0f}|{r['first50']:.4f}|{r['last50']:.4f}|")
lines+=['','精度标注更正：原始 manifest 的 `precision=float32` 是将 `use_amp=false` 误解为全模型 FP32 的元数据错误；训练代码没有执行全模型 FP32 转换。三组均沿用原生加载与 PEFT 默认精度。实际 dtype 计数在 audit/dtypes_*.json。\n\n训练循环时间从官方日志的开始/结束时间计算，包含该循环内 checkpoint 保存；完整运行（含模型构建、保存与恢复核验）耗时另见 summary.json。单次测量受温度、频率和缓存影响，不能根据这次时间排序判断 D 比 C 更快。显存为 PyTorch 训练步峰值，覆盖 forward/backward/optimizer.step，不是整卡 nvidia-smi 占用。','', '## 统一评测','', '原生 fixed10，chunk50，Object0，seed5000。每个模型先 init0 运行1回合排障，随后 init10–19 正式10回合。10回合不会支持稳定提升或显著性结论。','', '|模型|成功/回合|成功率|Wilson 95% 区间|留出轨迹 Loss（72帧）|','|---|---:|---:|---:|---:|']
def wilson(k,n):
 z=1.95996398454;p=k/n;den=1+z*z/n;c=(p+z*z/(2*n))/den;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return f'{100*(c-h):.1f}%–{100*(c+h):.1f}%'
lines.append(f"|未训练 base|{sum(bresult['successes'])}/10|{100*statistics.mean(bresult['successes']):.0f}%|{wilson(sum(bresult['successes']),10)}|{bval['mean_loss']:.4f}|")
for r in rows:lines.append(f"|{r['group']}|{r['successes']}/{r['episodes']}|{100*r['successes']/r['episodes']:.0f}%|{wilson(r['successes'],r['episodes'])}|{r['validation']:.4f}|")
lines+=['', '![训练曲线与资源对比](C:/VLATest/v4_results/training_comparison.png)','', '## 可以支持的结论','', 'B/C/D 在这台 RTX 5060 Laptop 8GB 的固定配置下都能训练。C、D 的指定 VLM 层并非只匹配到模块名：1步、3步及正式训练每一步均记录真实有限梯度、非零梯度和权重更新；保存后通过原生 make_policy 工厂逐项恢复一致，评测进程还将实际加载的全部适配器张量与磁盘 safetensors 逐项比较。','', '模块作用：Expert q/v LoRA 适配动作速度场的注意力计算，状态/动作投影 LoRA 适配低维状态与动作接口；VLM 指定层 q/v LoRA 调整上下文信息的选择与传递。视觉编码器基座保持冻结，C/D 也不是整层全参数解冻。\n\nB→C、C→D 每次增加 25,600 个可训练参数，本次显存增量很小；这只适用于选定的靠后文本层和现有计算图，不能推广到全部 VLM 层。训练 Loss 下降不直接代表闭环策略成功率提升。三组正式 rollout 结果见上表，任何未显示优势的结果均保留，不通过更换任务或初态修饰结论。','', '## 数据、控制条件与限制','', '- 基座固定 smolvla_base revision `5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6`；LeRobot commit `7e241bd630a3719a56157a497ce5d08f244784f1`，PEFT 0.18.0。未使用历史 smolvla_libero 权重作为训练起点。','- 数据 revision `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`，task24 共44 episodes/6867帧；固定35训练/9留出，1/3步检查使用训练集前8条。详细列表、数据文件 SHA256 和任务逐条核验见 evidence 与 audit/dataset.json。','- rank8/alpha16、AMP关闭且采用原生混合 dtype（VLM BF16，LoRA FP32；详见 audit/dtypes_*.json）、batch1/workers0、seed42、LR1e-4、正式 warmup25/decay500、双相机512 padding。每阶段都从同一预训练基座独立开始，检查阶段不加入正式500步预算。','- 同一训练 seed 不等于逐样本严格配对：增加适配器会改变 RNG 消耗，初始 LoRA、minibatch 顺序及 flow noise 未锁定为完全一致。','- 离线验证固定9条留出轨迹，每条均匀8帧，三组及 base 使用相同采样位置/噪声种子；训练沿用数据集已有全局归一化统计，因此存在统计量覆盖留出集合的限制。','- 评测记录实际 init_state_id；轨迹演示与模拟初态对应关系未知，不能将这10个初态声称为训练外独立初态。单任务/单训练种子/500步与10回合规模，只支持有限预算比较，不能宣称泛化提升。','- 原 A 全参数 Expert 训练起点及条件不同，只作历史参考，不直接计算 A/B/C/D 的因果收益。','- MuJoCo EGL 仍回退软件渲染；模型计算使用 CUDA。评测墙钟时间不代表机器人20Hz在线性能。','- 首次 B 检查因实验包装脚本导入路径错误在训练前退出，修正后重新完成 B 的1步门槛，原始记录保留；全部正式训练无 OOM。','', '## 产物与复现','', '- `RESULTS.csv`：原始模板列名的完整实测表；`summary.json`：汇总指标。','- `evidence/B|C|D/check1|check3|formal/`：参数清单、每步梯度/更新、显存、保存恢复、数据列表。','- `evidence/B|C|D/eval/`：10回合 eval_info、实际初态、加载验证与完整视频；`evidence/base/`：未训练对照。','- `train_verified.py`、`eval_verified.py`、`validate_loss.py`、`PROTOCOL.md`：本次实验实现及协议。WSL原始根目录：`'+str(root)+'`。','- 原始 checkpoint 位于每组 `evidence/.../formal/result.json` 的 checkpoint 路径；Windows 的 models/B、C、D 也复制了最终适配器 checkpoint（不包含大型基座权重；adapter_config 中仍引用 WSL 原始基座）。']
(out/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
# Copy complete logs and baseline evidence, including smoke and initial failed attempts.
for p in root.glob('*.log'):shutil.copy2(p,out/p.name)
for group in ['base','B','C','D']:
 dest=out/'evidence'/group;dest.mkdir(parents=True,exist_ok=True)
 for ev in sorted((root/'eval').glob(f'{group}_*')):
  shutil.copytree(ev,dest/ev.name,dirs_exist_ok=True)
 if (root/f'validation_{group}.json').exists():shutil.copy2(root/f'validation_{group}.json',dest/f'validation_{group}.json')
# Verify real initial states, loaded adapters, and video counts.
for group in ['base','B','C','D']:
 for stage,n,offset in [('smoke',1,0),('formal',10,10)]:
  ev=sorted((root/'eval').glob(f'{group}_{stage}*'))[-1]
  state=[json.loads(x)['init_state_index'] for x in (ev/'initial_states.jsonl').read_text().splitlines()]
  assert state==list(range(offset,offset+n)),(group,stage,state)
  assert len(list((ev/'videos').rglob('*.mp4')))==n
  if group!='base':assert json.loads((ev/'adapter_verified.json').read_text())['exact']
for r in rows:
 ckpt=Path(json.loads((Path(r['run'])/'result.json').read_text())['checkpoint'])
 shutil.copytree(ckpt,out/'models'/r['group'],dirs_exist_ok=True)
print('FINAL_REPORT_PASS',out)


