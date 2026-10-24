import subprocess,sys,json,time
from pathlib import Path
root=Path('/home/xiaolongwu/vla_workspace/vla_interview_project/vla_training_ablation_v4')
python='/home/xiaolongwu/vla_workspace/env/bin/python'
# Refuse downstream evaluation unless every prescribed training stage passed.
for group in 'BCD':
 for stage in ['check1','check3','formal']:
  run=sorted((root/'runs').glob(f'{group}_{stage}*'))[-1]
  assert json.loads((run/'result.json').read_text())['status']=='passed',(group,stage)
for group in ['base','B','C','D']:
 for n in [1,10]:
  log=root/f'eval_{group}_{n}_attempt1.log'
  assert not log.exists()
  print('START_EVAL',group,n,flush=True)
  with log.open('w') as f:
   result=subprocess.run([python,str(root/'eval_verified.py'),group,str(n)],stdout=f,stderr=subprocess.STDOUT)
  if result.returncode:
   print(log.read_text()[-6000:],flush=True);sys.exit(result.returncode)
  print('DONE_EVAL',group,n,flush=True)
for group in ['base','B','C','D']:
 log=root/f'validation_{group}.log'
 with log.open('w') as f:
  result=subprocess.run([python,str(root/'validate_loss.py'),group],stdout=f,stderr=subprocess.STDOUT)
 if result.returncode:
  print(log.read_text()[-6000:],flush=True);sys.exit(result.returncode)
 print('DONE_VALIDATION',group,flush=True)
