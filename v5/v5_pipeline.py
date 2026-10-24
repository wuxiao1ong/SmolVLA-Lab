from pathlib import Path
import subprocess,json,time,sys
ROOT=Path('/home/xiaolongwu/vla_workspace/outputs/base_single_task_v5');PY='/home/xiaolongwu/vla_workspace/env/bin/python'
TRAIN=str(Path(__file__).resolve().with_name('v5_train_batch_eval.py'));EVAL=str(Path(__file__).resolve().with_name('v5_eval.py'));VAL=str(Path(__file__).resolve().with_name('v5_validate.py'))
A=ROOT/'A_formal2000_bs24_w2_20261021_171218'
print('WAIT_A',A,flush=True)
start=time.monotonic()
while not (A/'result.json').exists():
 if time.monotonic()-start>3600:raise RuntimeError('A completion wait timed out')
 time.sleep(5)

def command(args,log):
 if log.exists():
  suffix=2
  while log.with_name(log.stem+f'_attempt{suffix}'+log.suffix).exists():suffix+=1
  log=log.with_name(log.stem+f'_attempt{suffix}'+log.suffix)
 print('START',args[1:],flush=True)
 with log.open('w') as f:r=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
 if r.returncode:
  print(log.read_text()[-5000:],flush=True);raise RuntimeError(f'Command failed {r.returncode}: {log}')
 print('DONE',log.name,flush=True)

def assess(label,ckpt,n,offset,seed,stage):
 prior=sorted((ROOT/'eval').glob(f'{label}_{n}ep_init{offset}_*')) if (ROOT/'eval').exists() else []
 for ev in prior:
  if (ev/'result.json').exists() and json.loads((ev/'manifest.json').read_text())['seed']==seed:
   print('REUSE_EVAL',ev,flush=True);return dict(path=str(ev),**json.loads((ev/'result.json').read_text()))
 before=set((ROOT/'eval').glob('*')) if (ROOT/'eval').exists() else set()
 command([PY,EVAL,label,str(ckpt),str(n),str(offset),'20',str(seed)],ROOT/f'{label}_{stage}.log')
 new=set((ROOT/'eval').glob('*'))-before;assert len(new)==1
 ev=next(iter(new));result=json.loads((ev/'result.json').read_text());states=[json.loads(x)['init_state_index'] for x in (ev/'initial_states.jsonl').read_text().splitlines()]
 assert states==list(range(offset,offset+n)),states
 print('EVAL_RESULT',label,stage,sum(result['successes']),n,flush=True)
 return dict(path=str(ev),**result)

summary=json.loads((ROOT/'assessment_summary.json').read_text()) if (ROOT/'assessment_summary.json').exists() else {}
for group in 'ABCD':
 if group in summary:print('REUSE_COMPLETE_GROUP',group,flush=True);continue
 if group=='A':run=A
 else:
  for steps in [1,3,2000]:
   stage=f'check{steps}' if steps<=3 else f'formal{steps}'
   prior=[p for p in sorted(ROOT.glob(f'{group}_{stage}_bs24_w2_*')) if (p/'result.json').exists() and json.loads((p/'result.json').read_text())['status']=='passed']
   if prior:run=prior[-1];print('REUSE_TRAIN',run,flush=True);continue
   before=set(ROOT.glob(f'{group}_*'))
   command([PY,TRAIN,group,str(steps),'24','2'],ROOT/f'{group}_bs24_w2_{steps}.log')
   new={p for p in set(ROOT.glob(f'{group}_*'))-before if p.is_dir() and (p/'manifest.json').exists()};assert len(new)==1
   run=next(iter(new));assert json.loads((run/'result.json').read_text())['status']=='passed'
 result=json.loads((run/'result.json').read_text());assert result['status']=='passed' and result['reload_exact'];ckpt=Path(result['checkpoint'])
 existing=list((ROOT/'validation').glob(f'{group}_bs24_2000_*.json')) if (ROOT/'validation').exists() else []
 if not existing:command([PY,VAL,f'{group}_bs24_2000',str(ckpt)],ROOT/f'{group}_bs24_validation.log')
 smoke=assess(f'{group}_bs24_2000',ckpt,1,10,5000,'smoke')
 discovery=assess(f'{group}_bs24_2000',ckpt,10,10,5000,'discovery10') if group=='A' else None
 if group=='A' and not any(discovery['successes']):
  (ROOT/'pipeline_needs_diagnosis.json').write_text(json.dumps(dict(reason='A zero discovery10; do not start BCD',run=str(run)),indent=2));raise RuntimeError('A discovery 0/10; inspect saved videos/actions before extending budget')
 main=assess(f'{group}_bs24_2000',ckpt,20,20,6000,'main20')
 summary[group]=dict(run=str(run),checkpoint=str(ckpt),smoke=smoke,discovery=discovery,main20=main)
 (ROOT/'assessment_summary.json').write_text(json.dumps(summary,indent=2))
 print('GROUP_COMPLETE',group,flush=True)
print('V5_PIPELINE_COMPLETE',flush=True)

