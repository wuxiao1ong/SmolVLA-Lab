from pathlib import Path
import subprocess,json,sys,time
root=Path('/home/xiaolongwu/vla_workspace/outputs/base_single_task_v5');records=[]
for batch in [4,8,16,32]:
 if batch==32 and records[-1].get('peak_reserved_gib',99)>4.75:break
 log=root/f'capacity_A_bs{batch}_step1.log';assert not log.exists()
 start=time.monotonic()
 with log.open('w') as f:r=subprocess.run(['/home/xiaolongwu/vla_workspace/env/bin/python','/mnt/c/VLATest/v5_train_batch.py','A','1',str(batch)],stdout=f,stderr=subprocess.STDOUT)
 runs=sorted(root.glob(f'A_check1_bs{batch}_w0_*'));run=runs[-1];result=json.loads((run/'result.json').read_text()) if (run/'result.json').exists() else dict(status='failed',error='no result file')
 records.append(dict(batch=batch,run=str(run),exit_code=r.returncode,**result))
 (root/'metadata/capacity.json').write_text(json.dumps(records,indent=2));print('CAPACITY',json.dumps(records[-1]),flush=True)
 if r.returncode:break
print('CAPACITY_DONE',flush=True)
