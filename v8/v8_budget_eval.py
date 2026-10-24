from pathlib import Path
import json,subprocess,time
R=Path('/home/xiaolongwu/vla_workspace/outputs/two_vlm_layers_v8')
OLD=Path('/home/xiaolongwu/vla_workspace/outputs/base_single_task_v5')
PY='/home/xiaolongwu/vla_workspace/env/bin/python'
assert 'V8_PIPELINE_COMPLETE' in (R/'pipeline.log').read_text()
a=json.loads((OLD/'assessment_summary.json').read_text())['A']
ckpt=Path(a['run'])/'train/checkpoints/001000/pretrained_model';label='A_budget1000'
prior=[p for p in sorted((R/'eval').glob(label+'_20ep_init20_*')) if (p/'result.json').exists()]
if prior:ev=prior[-1]
else:
    log=R/'A_budget1000_eval.log'
    if log.exists():log=log.with_name(log.stem+'_'+time.strftime('%H%M%S')+log.suffix)
    with log.open('w') as f:
        subprocess.run([PY,str(Path(__file__).resolve().with_name('v8_eval.py')),label,str(ckpt),'20','20','20','6000'],stdout=f,stderr=subprocess.STDOUT,check=True)
    ev=sorted((R/'eval').glob(label+'_20ep_init20_*'))[-1]
result=json.loads((ev/'result.json').read_text())
states=[json.loads(l)['init_state_index'] for l in (ev/'initial_states.jsonl').read_text().splitlines()]
assert states==list(range(20,40)) and len(result['successes'])==20
assert json.loads((ev/'dense_verified.json').read_text())['exact']
(R/'budget_eval_summary.json').write_text(json.dumps(dict(A1000=dict(path=str(ev),checkpoint=str(ckpt),**result),A2000=a['main20'],purpose='validation of existing checkpoints only, no new training; fixed development cases, no fresh test claim'),indent=2))
print('A_BUDGET_EVAL_COMPLETE',sum(result['successes']),20,flush=True)
