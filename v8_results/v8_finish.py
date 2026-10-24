from pathlib import Path
import subprocess,time
R=Path('/home/xiaolongwu/vla_workspace/outputs/two_vlm_layers_v8')
PY='/home/xiaolongwu/vla_workspace/env/bin/python'
start=time.monotonic()
while 'V8_PIPELINE_COMPLETE' not in (R/'pipeline.log').read_text():
    log=(R/'pipeline.log').read_text()
    if 'Traceback (most recent call last)' in log:raise RuntimeError('Pipeline failed; inspect pipeline.log, no automatic retraining')
    if time.monotonic()-start>7200:raise RuntimeError('Pipeline wait exceeded2h')
    time.sleep(5)
for script in ['v8_budget_eval.py','v8_report.py','v8_close.py']:
    print('START_FINISH',script,flush=True)
    subprocess.run([PY,'/mnt/c/VLATest/'+script],check=True)
print('V8_ALL_COMPLETE',flush=True)
