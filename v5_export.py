from pathlib import Path
import json,csv,shutil,statistics
root=Path('/home/xiaolongwu/vla_workspace/outputs/base_single_task_v5');out=Path('/mnt/c/VLATest/v5_results');out.mkdir(exist_ok=True)
data=json.loads((root/'metadata/data_audit.json').read_text());rows=[]
fields=['run_id','group','init_policy','task_text','libero_suite','libero_task_id','dataset_revision','train_episodes','train_frames','steps','batch_size','lora_rank','trainable_params','peak_vram_gib','loss_train','loss_val','n_action_steps','eval_seed','success_count','eval_episodes','success_rate','adapter_loaded','notes','status','actual_steps','peak_reserved_gib','sample_budget','run_path']
rows.append(dict(run_id='O_previous',group='official',init_policy='lerobot/smolvla_libero',task_text=data['task_text'],libero_suite='libero_object',libero_task_id=0,n_action_steps=10,eval_seed=5000,success_count=8,eval_episodes=10,success_rate=.8,adapter_loaded=False,status='previous_reference_not_rerun',notes='Skipped per user; recovery_20261021 official init10–19 control'))
for run in sorted(root.glob('*formal*')):
 if not (run/'manifest.json').exists():continue
 m=json.loads((run/'manifest.json').read_text());lines=(run/'steps.jsonl').read_text().splitlines() if (run/'steps.jsonl').exists() else [];s=[json.loads(l) for l in lines if l.strip()];r=json.loads((run/'result.json').read_text()) if (run/'result.json').exists() else {}
 params=json.loads((run/'trainable.json').read_text()) if (run/'trainable.json').exists() else {}
 rows.append(dict(run_id=run.name,group='A-lite' if m['group']=='A' else m['group'],init_policy='lerobot/smolvla_base',task_text=data['task_text'],libero_suite='libero_object',libero_task_id=0,dataset_revision=data['dataset_revision'],train_episodes=len(m['train']),train_frames=5472,steps=m['steps'],batch_size=m['batch'],lora_rank=m['rank'],trainable_params=params.get('trainable',''),peak_vram_gib=max((x['peak_allocated_gib'] for x in s),default=''),loss_train=statistics.mean(x['loss'] for x in s[-100:]) if s else '',n_action_steps=10,notes='Original native dtype; full-dataset stats; checkpoint discovery/progress eval separate from formal assessment',status=r.get('status','running'),actual_steps=len(s),peak_reserved_gib=max((x['peak_reserved_gib'] for x in s),default=''),sample_budget=m['batch']*len(s),run_path=str(run)))
summary=json.loads((root/'assessment_summary.json').read_text()) if (root/'assessment_summary.json').exists() else {}
for row in rows:
 group='A' if row['group']=='A-lite' else row['group']
 if row.get('batch_size')==24:
  valfiles=sorted((root/'validation').glob(f'{group}_bs24_2000_*.json')) if (root/'validation').exists() else []
  if valfiles:row['loss_val']=json.loads(valfiles[-1].read_text())['mean_loss']
 if group in summary and summary[group]['run']==row.get('run_path'):
  result=summary[group]['main20'];outcomes=result['successes'];row.update(eval_seed=6000,success_count=sum(outcomes),eval_episodes=len(outcomes),success_rate=sum(outcomes)/len(outcomes),adapter_loaded=True if group!='A' else 'not_applicable_full_model',status='completed_train_and_main20',notes='Main20 init20–39; fixed10; exact checkpoint reload; native dtype; global-stat validation limitation')
 elif row.get('status')=='passed':row['status']='train_passed_eval_pending'
with (out/'results_v5.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
shutil.copytree(root/'metadata',out/'metadata',dirs_exist_ok=True)
for p in root.glob('*.py'):shutil.copy2(p,out/p.name)
shutil.copy2(root/'PROTOCOL.md',out/'PROTOCOL.md')
for run in sorted(root.glob('*check*')):
 if not (run/'result.json').exists():continue
 dest=out/'gates'/run.name;dest.mkdir(parents=True,exist_ok=True)
 for n in ['manifest.json','trainable.json','result.json','steps.jsonl']:
  if (run/n).exists():shutil.copy2(run/n,dest/n)
(out/'LIVE_STATUS.md').write_text('# v5 当前实测进度\n\n官方任务验证按用户要求跳过，保留已完成的8/10对照。新训练固定从smolvla_base出发。\n\n|运行|状态|实际/计划步数|batch|Loss最近100步|峰值allocated/reserved GiB|\n|---|---|---:|---:|---:|---:|\n'+'\n'.join(f"|{r['run_id']}|{r['status']}|{r.get('actual_steps','')}/{r.get('steps','')}|{r.get('batch_size','')}|{r.get('loss_train','')}|{r.get('peak_vram_gib','')}/{r.get('peak_reserved_gib','')}|" for r in rows)+'\n\n未完成评测的成功率保持空白，不能解释为0%。最终闭环结果与原生单回合功能检查分开记录。\n',encoding='utf8')
print('EXPORT_V5',len(rows),out)

