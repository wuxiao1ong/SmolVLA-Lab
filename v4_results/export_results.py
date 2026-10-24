import json,csv,re,shutil,statistics,datetime
from pathlib import Path
ROOT=Path('/home/xiaolongwu/vla_workspace/vla_interview_project/vla_training_ablation_v4');OUT=Path('/mnt/c/VLATest/v4_results');OUT.mkdir(exist_ok=True)
rows=[];details=[]
for group in 'BCD':
 runs={stage:sorted((ROOT/'runs').glob(f'{group}_{stage}*'))[-1] for stage in ['check1','check3','formal']}
 run=runs['formal'];res=json.loads((run/'result.json').read_text());man=json.loads((run/'manifest.json').read_text());params=json.loads((run/'trainable.json').read_text())
 steps=[json.loads(l) for l in (run/'steps.jsonl').read_text().splitlines()]
 assert all(json.loads((r/'result.json').read_text())['status']=='passed' for r in runs.values())
 evs=sorted((ROOT/'eval').glob(f'{group}_formal*')) if (ROOT/'eval').exists() else []
 ev=json.loads((evs[-1]/'result.json').read_text()) if evs and (evs[-1]/'result.json').exists() else None
 valpath=ROOT/f'validation_{group}.json'; val=json.loads(valpath.read_text()) if valpath.exists() else None
 log=(ROOT/f'{group}_formal_attempt1.log').read_text()
 def stamp(mark):
  line=next(x for x in log.splitlines() if mark in x);m=re.search(r'(2026-\d\d-\d\d \d\d:\d\d:\d\d)',line);return datetime.datetime.fromisoformat(m[1])
 seconds=(stamp('End of training')-stamp('Start offline training')).total_seconds()
 row=dict(group=group,checkpoint=res['checkpoint'],git_commit=man['lerobot_commit'],dataset_revision='a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4',train_episodes=json.dumps(man['train']),train_steps=500,seed=42,lora_r=8,lora_alpha=16,trainable_parameters=params['trainable'],peak_allocated_gib=res['peak_allocated_gib'],peak_reserved_gib=res['peak_reserved_gib'],train_seconds=seconds,eval_task='libero_object/0',eval_episodes=len(ev['successes']) if ev else '',success_count=sum(ev['successes']) if ev else '',success_rate=statistics.mean(ev['successes']) if ev else '',status='completed' if ev and val else 'training_completed',notes='1/3/500 passed; exact reload; AMP off/native dtypes; fixed10; single seed')
 rows.append(row)
 details.append(dict(group=group,run=str(run),trainable=params['trainable'],train_seconds=seconds,total_seconds=res['elapsed_s'],peak_allocated=res['peak_allocated_gib'],peak_reserved=res['peak_reserved_gib'],first50=statistics.mean(x['loss'] for x in steps[:50]),last50=statistics.mean(x['loss'] for x in steps[-50:]),validation=val['mean_loss'] if val else None,successes=sum(ev['successes']) if ev else None,episodes=len(ev['successes']) if ev else None))
 dest=OUT/'evidence'/group;dest.mkdir(parents=True,exist_ok=True)
 for stage,r in runs.items():
  sd=dest/stage;sd.mkdir(exist_ok=True)
  for f in ['manifest.json','result.json','trainable.json','steps.jsonl']:shutil.copy2(r/f,sd/f)
 for name in [f'{group}_check1_attempt1.log',f'{group}_check3_attempt1.log',f'{group}_formal_attempt1.log']:
  if (ROOT/name).exists():shutil.copy2(ROOT/name,dest/name)
 if evs:
  ed=dest/'eval';ed.mkdir(exist_ok=True)
  for f in evs[-1].rglob('*'):
   if f.is_file():
    d=ed/f.relative_to(evs[-1]);d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,d)
 if valpath.exists():shutil.copy2(valpath,dest/valpath.name)
with (OUT/'RESULTS.csv').open('w',newline='',encoding='utf-8-sig') as f:
 writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerow(dict(group='A',status='reference_only',notes='Historical expert full fine-tuning; different checkpoint/budget; no direct causal comparison'));writer.writerows(rows)
(OUT/'summary.json').write_text(json.dumps(details,indent=2))
for name in ['GUIDE_v4.md','PROTOCOL.md','train_verified.py','eval_verified.py','validate_loss.py']:
 shutil.copy2(ROOT/name,OUT/name)
shutil.copytree(ROOT/'module_audit',OUT/'module_audit',dirs_exist_ok=True);shutil.copytree(ROOT/'audit',OUT/'audit',dirs_exist_ok=True)
print(json.dumps(details,indent=2))

