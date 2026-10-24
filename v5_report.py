from pathlib import Path
import json,statistics,math,shutil,csv
ROOT=Path('/home/xiaolongwu/vla_workspace/outputs/base_single_task_v5');OUT=Path('/mnt/c/VLATest/v5_results');OUT.mkdir(exist_ok=True)
summary=json.loads((ROOT/'assessment_summary.json').read_text()) if (ROOT/'assessment_summary.json').exists() else {}
complete=len(summary)==4
lines=['# v5 SmolVLA Base 单任务学习与 LoRA 对照','', '状态：'+('A/B/C/D 训练、恢复、验证和固定20回合评测全部完成。' if complete else '进行中；只将已结束的主评测写为结果，未完成项保持空白。'),'', '已按用户要求跳过官方权重重复验证，沿用此前同入口 Object0 初态10–19 的8/10参考结果。本轮所有新模型均从固定 smolvla_base 出发。','', '## 核心实测结果','', '|组别|名义requires_grad参数|batch / 步数|峰值 allocated / reserved GiB|训练末100步Loss|验证Loss|主评测成功|Wilson 95%区间|状态|','|---|---:|---:|---:|---:|---:|---:|---:|---|']
records=[]
def wilson(k,n):
 z=1.95996398454;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
 return f'{100*(c-h):.1f}%–{100*(c+h):.1f}%'
for group in 'ABCD':
 runs=sorted(ROOT.glob(f'{group}_formal2000_bs24_w2_*'))
 if not runs:
  lines.append(f'|{group}|—|24 / 2000|—|—|—|—|—|未执行|');continue
 run=runs[-1];m=json.loads((run/'manifest.json').read_text());t=json.loads((run/'trainable.json').read_text()) if (run/'trainable.json').exists() else {};r=json.loads((run/'result.json').read_text()) if (run/'result.json').exists() else {};steps=[json.loads(l) for l in (run/'steps.jsonl').read_text().splitlines()] if (run/'steps.jsonl').exists() else []
 vals=sorted((ROOT/'validation').glob(f'{group}_bs24_2000_*.json')) if (ROOT/'validation').exists() else [];v=json.loads(vals[-1].read_text()) if vals else None
 ev=summary[group]['main20'] if group in summary else None;k=sum(ev['successes']) if ev else None;n=len(ev['successes']) if ev else None
 loss=statistics.mean(x['loss'] for x in steps[-100:]) if steps else None;alloc=max((x['peak_allocated_gib'] for x in steps),default=0);reserve=max((x['peak_reserved_gib'] for x in steps),default=0)
 status='全部完成' if ev else '训练通过、评测待完成' if r.get('status')=='passed' else f'训练中 {len(steps)}/2000'
 lines.append(f"|{'A-lite' if group=='A' else group}|{t.get('trainable','—'):,}|24 / 2000|{alloc:.3f} / {reserve:.3f}|{loss:.4f}|{v['mean_loss'] if v else '—'}|{str(k)+'/'+str(n) if ev else '—'}|{wilson(k,n) if ev else '—'}|{status}|")
 records.append(dict(group=group,run=str(run),trainable=t.get('trainable'),result=r,validation=v,main=ev,loss_last100=loss))
 dest=OUT/'evidence'/group;dest.mkdir(parents=True,exist_ok=True)
 for f in ['manifest.json','trainable.json','result.json','steps.jsonl']:
  if (run/f).exists():shutil.copy2(run/f,dest/f)
 if vals:shutil.copy2(vals[-1],dest/'validation.json')
 if ev:
  edir=Path(ev['path']);shutil.copytree(edir,dest/'main20',dirs_exist_ok=True)
  states=[json.loads(x)['init_state_index'] for x in (edir/'initial_states.jsonl').read_text().splitlines()];assert states==list(range(20,40))
  assert len(ev['successes'])==20
  if group!='A':assert json.loads((edir/'adapter_verified.json').read_text())['exact']
 if r.get('status')=='passed' and group!='A':shutil.copytree(Path(r['checkpoint']),OUT/'models'/group,dirs_exist_ok=True)
 if 'discovery' in summary.get(group,{}) and summary[group]['discovery']:
  shutil.copytree(Path(summary[group]['discovery']['path']),dest/'discovery10',dirs_exist_ok=True)
 if (run/'train/eval').exists():shutil.copytree(run/'train/eval',dest/'native_functional_checks',dirs_exist_ok=True)
lines+=['', '主比较统一 Object0、fixed10、chunk50、相对动作、20Hz配置、seed6000、真实初态20–39。A 额外在初态10–19/seed5000做了10回合诊断（9/10），排障1回合及训练中功能检查均不混入主比较。示范与模拟初态的对应关系未知，所以不把20–39声称为训练外独立初态。','', '![v5实测对比](C:/VLATest/v5_results/v5_comparison.png)\n\n## 训练参数为何调整','', '按用户授权进行容量测试：batch4/8/16/32均完成首次反传与保存恢复，batch32 reserved7.52GiB余量过小。选择batch24/workers2，连续3步峰值allocated6.093/reserved6.266GiB。正式训练每组2000步对应48000起始帧抽样，约8.77等效轮；旧batch1的2000步仅约0.365轮。抽样覆盖量不是不重复样本数量。吞吐按实际数据加载与更新时间记录，约29样本/秒。不能把所有提升单独归因于batch：本轮也建立了专家全参数A-lite基线并增加总抽样预算。','', '统一参数：LR1e-4、warmup100/decay2000、seed42、batch24/workers2；B/C/D LoRA rank8/alpha16。保持原生混合dtype（主要冻结/基座权重BF16，LoRA为FP32），AMP关闭；未将整模型转换为FP32。A-lite训练Expert及状态/动作投影，视觉与VLM基座冻结；B训练Expert q/v与5个投影的LoRA，C额外VLM14 q/v，D额外VLM13/14 q/v。','', '## 对此前0%的诊断','', '已核查完整Episode→任务映射，task24语言与Object0精确一致。44条演示末帧均与目标放入篮子的视觉结果一致；双相机方向、8维状态和7维相对动作接口匹配，归一化张量与已成功官方模型一致。因此目前没有证据支持“选错任务”是此前全零的原因。','', '原v4仅500步×batch1，且从通用base直接做低秩适配；真实梯度与保存恢复虽通过，却不足以证明已学会任务。本轮先建立全专家基线并增加有效抽样量，A-lite在1000/1500步功能检查均成功，最终固定初态10–19诊断9/10，主20回合见表。这支持从base适配任务是可行的，但没有把训练范式、预算、batch等因素分别做单变量隔离，不能宣称已证明唯一原因。','', '500步功能视频显示已接近并夹起目标，但后续转移放置失败；随后节点开始成功。所有失败与原始结果保留，不通过换任务、换初态或替换成smolvla_libero训练起点修饰结果。','', '## 验收与限制','', '- 每组1步→3步→正式训练，静态参数清单及优化器覆盖均记录；A检查代表性专家/投影的真实梯度及更新，LoRA逐步检查所有适配器梯度有限，并按Expert/投影/指定VLM层验证非零梯度和更新。初始LoRA A梯度为0可由B初始化0解释，类别内B参数实际更新。','- A权重差分审计覆盖155个名义可训练张量，其中122个张量数值改变，共58529442个元素改变，345个共同VLM冻结张量逐项一致。33个RMSNorm权重张量未改变；它们采用原生BF16，不能把requires_grad参数量解释为每个元素均实际更新。该量化/有效更新限制未单独做FP32对照。\n- 完成后原生make_policy恢复所有可训练张量逐项一致。独立评测再次比较全部LoRA adapter张量与磁盘权重，避免误评原始base。','- 35条训练5472帧、9条验证1395帧，按episode划分；验证固定72帧、相同flow noise种子，另记录18个观察点的物理动作RMSE。沿用全局归一化统计，因此存在统计量覆盖留出集合的限制。','- 相同训练seed不保证跨结构的minibatch顺序、adapter随机初值、flow noise逐项相同；这是本轮对照的限制。单任务、单训练种子与20回合规模，不能证明稳定泛化或微小差异显著。','- 训练原生单回合检查只用于观察学习进展，不是正式成功率。独立评测默认最多保存10段视频，20回合的全部结果及实际初态均保存。','- MuJoCo EGL仍回退软件渲染，模型训练/推理用CUDA。配置20Hz不代表真实机器人墙钟20Hz；训练速度含审计开销，不能仅按参数量推断更快。','- 首次独立评测因包装脚本seed参数文字拼接错误在环境运行前退出；修正后仅重试评测、未重训，失败日志保留且不计为0成功回合。','', '## 可追溯产物','', '- results_v5.csv：按v5模板填写，未完成结果留空；metadata含指导原文、版本、数据映射、35/9索引、容量/模块检查和示范末帧。','- evidence/A|B|C|D：完整训练参数/每步更新、验证动作诊断、主20回合结果与视频、实际初态、native功能视频。gates目录保存1/3步检查；models/B|C|D是最终adapter。','- A大模型原始权重保留WSL，路径见RUN_INDEX.json与evidence/A/result.json。所有旧v4、recovery和batch1 A2000保留。本目录训练/评测脚本可复核实际实现。','- WSL原始根：/home/xiaolongwu/vla_workspace/outputs/base_single_task_v5。']
name='FINAL_REPORT.md' if complete else 'PROGRESS_REPORT.md';(OUT/name).write_text('\n'.join(lines)+'\n',encoding='utf8')
(OUT/'RUN_INDEX.json').write_text(json.dumps({r['group']:dict(run=r['run'],checkpoint=r['result'].get('checkpoint'),main20_path=r['main']['path'] if r['main'] else None) for r in records},indent=2))
(OUT/'summary_v5.json').write_text(json.dumps([{k:v for k,v in r.items() if k!='validation'} for r in records],indent=2))
for p in ROOT.glob('*.log'):shutil.copy2(p,OUT/p.name)
if (ROOT/'assessment_summary.json').exists():shutil.copy2(ROOT/'assessment_summary.json',OUT/'assessment_summary.json')
print('REPORT_WRITTEN',name,list(summary))

