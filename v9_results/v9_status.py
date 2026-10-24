import json,time,sys
from pathlib import Path
root=Path('/home/xiaolongwu/vla_workspace/outputs/alite_vlm_lora_v9')
status={'time':time.strftime('%Y-%m-%d %H:%M:%S'),'stages':[],'eval':{}}
for run in sorted(root.glob('*_202610*')):
    if not run.is_dir():continue
    row={'run':run.name}
    if (run/'result.json').exists():
        result=json.loads((run/'result.json').read_text());row.update(status=result['status'],steps=result['steps'])
        if result['status']=='passed':row.update(loss=result['mean_last100_loss'],peak_gib=result['peak_allocated_gib'])
    elif (run/'steps.jsonl').exists():
        last=json.loads((run/'steps.jsonl').read_text().splitlines()[-1]);row.update(status='running',step=last['step'],loss=last['loss'],peak_gib=last['peak_allocated_gib'])
    else:row['status']='loading'
    status['stages'].append(row)
for label in ['A','C_A14','D_A13_14']:
    path=root/'eval'/label
    if (path/'result.json').exists():
        result=json.loads((path/'result.json').read_text());status['eval'][label]={'status':'completed','success':sum(c['success'] for c in result['cases']),'episodes':len(result['cases'])}
    elif (path/'initial_states.jsonl').exists():
        started=len((path/'initial_states.jsonl').read_text().splitlines());latest={}
        if (path/'trajectory.jsonl').exists():
            for line in (path/'trajectory.jsonl').read_text().splitlines():
                try:
                    row=json.loads(line);latest[row['case']]=row
                except json.JSONDecodeError:pass
        completed=started-1
        if started-1 in latest and (latest[started-1]['success'] or latest[started-1]['step']>=280):completed+=1
        status['eval'][label]={'status':'running','started_episodes':started,'completed_episodes':completed,'success_completed':sum(latest[i]['success'] for i in range(completed) if i in latest)}
        if label!='A':
            actual=[json.loads(x) for x in (path/'initial_states.jsonl').read_text().splitlines()]
            reference=[json.loads(x) for x in (root/'eval/A/initial_states.jsonl').read_text().splitlines()]
            status['eval'][label]['initial_observations_match_A']=actual==reference[:len(actual)]
            noise_path=path/'noise_audit.jsonl'
            if noise_path.exists():
                actual_noise=[json.loads(x) for x in noise_path.read_text().splitlines()]
                reference_noise=[json.loads(x) for x in (root/'eval/A/noise_audit.jsonl').read_text().splitlines()]
                status['eval'][label]['noise_matches_A']=actual_noise==reference_noise[:len(actual_noise)]
print(json.dumps(status['eval'] if '--eval-only' in sys.argv else status,indent=2))
out=Path('/mnt/c/VLATest/v9_results');out.mkdir(exist_ok=True)
(out/'progress.json').write_text(json.dumps(status,indent=2))
if (root/'PLAN.json').exists():(out/'PLAN.json').write_text((root/'PLAN.json').read_text())
