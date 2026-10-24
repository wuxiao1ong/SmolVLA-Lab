from pathlib import Path
import json,csv,shutil,statistics,runpy,ast
R=Path('/home/xiaolongwu/vla_workspace/outputs/expert_continuation_v6')
O=Path('/home/xiaolongwu/vla_workspace/outputs/base_single_task_v5')
D=Path('/mnt/c/VLATest/v6_results');D.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text())
if not (R/'dense_delta_audit.json').exists() or set(read(R/'dense_delta_audit.json'))!={'E','E2'}:
    runpy.run_path(str(Path(__file__).with_name('v6_delta_audit.py')),run_name='__main__')
s=read(R/'assessment_summary.json');v5=read(O/'assessment_summary.json')
stages={'E1':read(R/'E1_assessment_summary.json'),'E2':read(R/'E2_assessment_summary.json')}
assert 'confirmation10' in s and 'A_confirmation10' in s
chosen=next(k for k,v in stages.items() if v['checkpoint']==s['checkpoint'])
score=lambda ev:f"{sum(ev['successes'])}/{len(ev['successes'])}"
rows=[];stats={}
for name,v in {**v5,**stages}.items():
    extra=s['A_confirmation10'] if name=='A' else s['confirmation10'] if name==chosen else None
    rows.append(dict(group=name,main20_successes=sum(v['main20']['successes']),main20_n=20,confirmation10_successes=sum(extra['successes']) if extra else '',confirmation10_n=10 if extra else '',source='C2000' if name=='E1' else 'E1' if name=='E2' else 'base',total_training_steps=4000 if name=='E2' else 3000 if name=='E1' else 2000,checkpoint=v['checkpoint']))
for name,v in stages.items():
    prefix='E' if name=='E1' else 'E2'
    for stage in ['check1','check3','formal1000']:
        run=sorted(R.glob(f'{prefix}_{stage}_bs24_w2_*'))[-1]
        assert read(run/'result.json')['status']=='passed'
        assert read(run/'initial_weights_verified.json')['all_saved_tensors_exact']
        assert read(run/'frozen_weights_verified.json')['exact']
        dest=D/'evidence'/name/stage;dest.mkdir(parents=True,exist_ok=True)
        for p in run.glob('*.json*'):shutil.copy2(p,dest/p.name)
        if (run/'train/eval').exists():shutil.copytree(run/'train/eval',dest/'native_functional_checks',dirs_exist_ok=True)
    steps=[json.loads(l) for l in (run/'steps.jsonl').read_text().splitlines()]
    assert len(steps)==1000 and read(run/'result.json')['reload_exact']
    pattern='E_Cplus1000_*.json' if name=='E1' else 'E2_Cplus2000_*.json'
    stats[name]=dict(loss=statistics.mean(x['loss'] for x in steps[-100:]),validation=read(sorted((R/'validation').glob(pattern))[-1])['mean_loss'],allocated=max(x['peak_allocated_gib'] for x in steps),reserved=max(x['peak_reserved_gib'] for x in steps),trainable=read(run/'trainable.json')['trainable'])
def copy_eval(name,ev,start,n):
    src=Path(ev['path']);states=[json.loads(l)['init_state_index'] for l in (src/'initial_states.jsonl').read_text().splitlines()]
    assert states==list(range(start,start+n)) and len(ev['successes'])==n
    assert read(src/'dense_verified.json')['exact']
    shutil.copytree(src,D/'evidence'/name,dirs_exist_ok=True)
for name,v in stages.items():copy_eval(name+'_main20',v['main20'],20,20)
copy_eval('selected_confirmation10',s['confirmation10'],40,10)
copy_eval('A_confirmation10',s['A_confirmation10'],40,10)
with (D/'results_v6.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
pairs=[dict(init_state_index=20+i,seed=6000+i,**{g:int(v['main20']['successes'][i]) for g,v in {**v5,**stages}.items()}) for i in range(20)]
with (D/'paired_main20.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
confirm_pairs=[dict(init_state_index=40+i,seed=7000+i,selected_model=chosen,selected_success=int(s['confirmation10']['successes'][i]),A_success=int(s['A_confirmation10']['successes'][i])) for i in range(10)]
with (D/'paired_confirmation10.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(confirm_pairs[0]));w.writeheader();w.writerows(confirm_pairs)
index={'selected':s['checkpoint'],**{name:v['checkpoint'] for name,v in stages.items()},'A_reference':v5['A']['checkpoint'],'C_source':v5['C']['checkpoint'],'merged_start':str(R/'C_merged_start')}
index['selected_continuation']=index.pop('selected')
index.update(recommended_model='A-lite_v5',recommended_checkpoint=v5['A']['checkpoint'],reason='A-lite has higher success on both matched main20 and predeclared confirmation10; continuation models retained as experiments')
(D/'RUN_INDEX.json').write_text(json.dumps(index,indent=2))
for folder in ['metadata','validation']:shutil.copytree(R/folder,D/folder,dirs_exist_ok=True)
for p in R.glob('*.json'):shutil.copy2(p,D/p.name)
for p in R.glob('*.log'):shutil.copy2(p,D/p.name)
for p in Path('/mnt/c/VLATest').glob('v6_*.py'):shutil.copy2(p,D/p.name)
shutil.copy2(Path('/mnt/c/VLATest/v6_prepare.log'),D/'v6_prepare.log')
lines=['# v6：从C继续训练动作专家','', '全部完成：两阶段各1步→3步→1000步、固定验证、各20回合主评测，及选定模型与A各10回合新初态确认。','',f"选定 **{chosen}**：主评测 **{score(s['main20'])}**，原C为8/20；新10个初态 **{score(s['confirmation10'])}**。A-lite参考主评测19/20、新初态{score(s['A_confirmation10'])}。",'', '|模型|主20回合|新10回合|累计训练步数|','|---|---:|---:|---:|']
for row in rows:
    extra=f"{row['confirmation10_successes']}/10" if row['confirmation10_n'] else '未测'
    lines.append(f"|{row['group']}|{row['main20_successes']}/20|{extra}|{row['total_training_steps']}|")
lines.insert(6,f"判定：原20个开发回合的成功率明显提高，但选定模型在新10初态仅{score(s['confirmation10'])}，不能据此宣称已稳定达到或超过A-lite。A在同样新初态为{score(s['A_confirmation10'])}。结果应分组看，不把开发选模与新初态确认混为独立测试。\n")
lines+=['', '## 改动和训练记录','', '从原C2000步LoRA checkpoint开始，安全合并39模块并保留已学的Expert、投影及VLM更新。E1完整训练动作Expert和5个状态/动作投影，冻结视觉与VLM，追加1000步；E2再从E1的正式1000步checkpoint追加1000步。两者均不再是纯LoRA，也不是与原BCD相同预算的消融。', '', '每阶段batch24/workers2、LR5e-5、warmup50/decay1000、seed42、原生混合BF16/FP32、AMP关闭。每阶段先做独立1/3步门槛，正式训练重新从该阶段起点开始，不承接门槛权重。优化器与调度器重建，属于模型权重续训，不是优化器状态原样恢复。', '', '|阶段|名义requires_grad参数|末100步训练Loss|固定72帧验证Loss|峰值allocated / reserved GiB|','|---|---:|---:|---:|---:|']
for name,t in stats.items():lines.append(f"|{name}|{t['trainable']:,}|{t['loss']:.4f}|{t['validation']:.4f}|{t['allocated']:.3f} / {t['reserved']:.3f}|")
lines+=['', 'E1每阶段24000起始帧抽样≈4.39等效轮；E2再增加相同预算，总计相对C额外48000抽样≈8.77等效轮。抽样数不是独立样本数。原C及E1/E2和所有中间checkpoint都保留。', '', '## 结果解释与选择','', f"E1主评测17/20后继续到E2。按主20成功数更高者选择、相同则选E2的规则，最终选定{chosen}，规则和计数保存在selection.json。该20回合已用于开发和选模，存在选择偏差，不是未经调参的最终测试。", '', '新增实际模拟初态40–49、seed7000在看到E1结果前预先固定；选定后只对该模型与A各评测一次，不根据这10回合再调参。主20采用实际初态20–39、seed6000；均为Object0、双相机、8维状态、7维相对动作、fixed10、chunk50和20Hz配置。配置20Hz不表示墙钟20Hz。', '', '扩大动作端训练范围后E1从40%提升到85%，支持这一改进方向有效。它同时改变训练范围、学习率、累计预算、合并和优化器阶段，不能把提升归因于某一个因素，更不能声称所有LoRA都不行。单任务、单训练种子、小评测样本仍需扩大验证。', '', '35训练轨迹5472帧、9验证轨迹1395帧未变，验证固定72帧和flow noise，另保留18观察点动作RMSE。沿用全局归一化，统计覆盖验证集合；示范与模拟初态对应未知，不宣称训练外泛化。各模型轨迹长度不同，未保证所有推理噪声逐次相同。离线Loss不等于闭环成功率。', '', '## 权重和执行验收','', 'C原78个适配器张量加载逐项一致，39模块按PEFT0.18 safe_merge公式核验。首次合并校验未考虑delta先转换dtype而触发断言，修正校验公式后通过，原日志保留。BF16合并有舍入，未声称与未合并FP32适配器输出完全相同。合并dense全张量保存恢复一致。', '', '六个训练阶段均核验：初始权重与对应checkpoint逐项一致、真实梯度和更新、冻结参数前后哈希一致、可训练权重原生恢复一致。所有独立评测再次比较全部dense权重与磁盘文件，记录实际初态。原生每500步单回合检查仅为功能诊断，不混入主评测；E1在500/1000步各1/1成功。', '', '## 可追溯文件','', '- results_v6.csv、paired_main20.csv：全部模型结果与逐初态配对。','- RUN_INDEX.json：选定模型、E1/E2、C来源和A参考完整模型路径；大权重保留WSL，不上传。','- evidence：六阶段每步训练/显存/权重记录、原生功能视频、两次主20及两次确认评测、真实初态和视频。主20每组默认最多保存10视频。','- metadata、validation、源码及日志：数据划分、合并审计和离线动作诊断。','', '![实测成功率对比](C:/VLATest/v6_results/comparison.png)','', '![原C和E1功能检查关键帧](C:/VLATest/v6_results/C_E_functional_frames.jpg)']
lines.insert(6,'**当前推荐继续使用A-lite。** 它在两组相同初态比较中均优于续训模型；E1/E2保留为实验版本，不替换现有最佳模型。RUN_INDEX.json中recommended_checkpoint指向A-lite，selected_continuation指向E1。\n')
delta=read(R/'dense_delta_audit.json')
lines+=['', '## 实际权重差分补充', '']
for prefix,a in delta.items():
    lines.append(f"{prefix}：名义{a['trainable_tensors']}个可训练张量中{a['changed_tensors']}个发生数值改变，共{a['changed_elements']}个元素改变。未变化张量详见dense_delta_audit.json；原生BF16精度下，requires_grad数量不等于每个元素均实际改变，本轮未做独立FP32对照。")
lines+=['','## 原生功能检查（不混入主比较）','']
native={}
for prefix in ['E','E2']:
    checks=[]
    for line in (R/f'{prefix}_formal1000.log').read_text().splitlines():
        if 'Suite overall aggregated:' in line:
            a=ast.literal_eval(line.split('Suite overall aggregated:',1)[1].strip())
            checks.append(dict(step=500*(len(checks)+1),successes=int(a['pc_success']>0),episodes=a['n_episodes']))
    native[prefix]=checks
    lines.append(prefix+'：'+', '.join(f"{a['step']}步 {a['successes']}/{a['episodes']}" for a in checks))
(D/'native_functional_checks.json').write_text(json.dumps(native,indent=2))
report='\n'.join(lines)+'\n';(D/'FINAL_REPORT.md').write_text(report,encoding='utf8');(R/'FINAL_REPORT.md').write_text(report,encoding='utf8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,2,figsize=(12,4.8))
ax[0].bar([r['group'] for r in rows],[5*r['main20_successes'] for r in rows],color=['#3975e8','#b98a36','#3a9a80','#9665bb','#ddad36','#e46e36'])
for i,row in enumerate(rows):ax[0].text(i,5*row['main20_successes']+2,f"{row['main20_successes']}/20",ha='center')
ax[0].set_title('Development comparison: matched 20 states');ax[0].set_ylabel('Success (%)');ax[0].set_ylim(0,110)
for name,key,color in [('A','A_confirmation10','#3975e8'),(chosen,'confirmation10','#e46e36')]:
    k=sum(s[key]['successes']);ax[1].bar(name,10*k,color=color);ax[1].text(name,10*k+2,f'{k}/10',ha='center')
ax[1].set_title('Predeclared new states 40-49');ax[1].set_ylabel('Success (%)');ax[1].set_ylim(0,110)
fig.tight_layout();fig.savefig(D/'comparison.png',dpi=160);plt.close(fig)
print('FINAL_AUDIT_PASS',chosen,score(s['main20']),score(s['confirmation10']),score(s['A_confirmation10']),flush=True)
