from pathlib import Path
import subprocess,json,time
ROOT=Path('/home/xiaolongwu/vla_workspace/outputs/expert_continuation_v6')
PY='/home/xiaolongwu/vla_workspace/env/bin/python'
if (ROOT/'E1_assessment_summary.json').exists():
    raise SystemExit('Initial pipeline finished. Use v6_extend.py to resume extension; do not overwrite selection.')
def command(script,args,log):
    log=ROOT/log
    if log.exists(): log=log.with_name(log.stem+'_'+time.strftime('%H%M%S')+log.suffix)
    print('START',script,args,flush=True)
    with log.open('w') as f:
        p=subprocess.run([PY,str(Path(__file__).resolve().parent / script),*map(str,args)],stdout=f,stderr=subprocess.STDOUT)
    if p.returncode:
        print(log.read_text()[-6000:],flush=True)
        raise RuntimeError(f'Failed: {log}')
    print('DONE',log,flush=True)
def assess(label,ckpt,n,offset,seed):
    prior=sorted((ROOT/'eval').glob(f'{label}_{n}ep_init{offset}_*'))
    for ev in reversed(prior):
        if (ev/'result.json').exists():
            m=json.loads((ev/'manifest.json').read_text())
            if m['seed']==seed and m['policy']==str(ckpt): break
    else:
        command('v6_eval.py',[label,ckpt,n,offset,20,seed],f'{label}_{n}ep_init{offset}.log')
        ev=sorted((ROOT/'eval').glob(f'{label}_{n}ep_init{offset}_*'))[-1]
    r=json.loads((ev/'result.json').read_text())
    states=[json.loads(l)['init_state_index'] for l in (ev/'initial_states.jsonl').read_text().splitlines()]
    assert states==list(range(offset,offset+n)) and len(r['successes'])==n
    assert json.loads((ev/'dense_verified.json').read_text())['exact']
    print('EVAL',label,sum(r['successes']),n,flush=True)
    return dict(path=str(ev),**r)
assert (ROOT/'merge_audit.json').exists()
for steps in [1,3,1000]:
    stage=f'check{steps}' if steps<=3 else f'formal{steps}'
    prior=[p for p in sorted(ROOT.glob(f'E_{stage}_bs24_w2_*')) if (p/'result.json').exists() and json.loads((p/'result.json').read_text()).get('status')=='passed']
    if prior:
        run=prior[-1];print('REUSE',run,flush=True)
    else:
        command('v6_train.py',[steps,24,2],f'E_{stage}.log')
        run=sorted(ROOT.glob(f'E_{stage}_bs24_w2_*'))[-1]
    r=json.loads((run/'result.json').read_text());assert r['status']=='passed' and r['reload_exact']
    assert json.loads((run/'frozen_weights_verified.json').read_text())['exact']
    assert json.loads((run/'initial_weights_verified.json').read_text())['all_saved_tensors_exact']
ckpt=Path(r['checkpoint'])
if not list((ROOT/'validation').glob('E_Cplus1000_*.json')):
    command('v6_validate.py',['E_Cplus1000',ckpt],'E_validation.log')
main=assess('E_Cplus1000',ckpt,20,20,6000)
summary=dict(group='E',run=str(run),checkpoint=str(ckpt),source='C_v5_2000',additional_steps=1000,prior_steps=2000,trainable_scope='full Expert + five projections, frozen C-merged VLM',main20=main)
(ROOT/'assessment_summary.json').write_text(json.dumps(summary,indent=2))
# Predeclared new development-unseen simulator states; no training changes based on these results.
extra=assess('E_Cplus1000_confirmation',ckpt,10,40,7000)
old=Path('/home/xiaolongwu/vla_workspace/outputs/base_single_task_v5')
a_ckpt=Path(json.loads((old/'assessment_summary.json').read_text())['A']['checkpoint'])
a_extra=assess('A_v5_confirmation',a_ckpt,10,40,7000)
summary.update(confirmation10=extra,A_confirmation10=a_extra)
(ROOT/'assessment_summary.json').write_text(json.dumps(summary,indent=2))
print('V6_PIPELINE_COMPLETE',sum(main['successes']),20,flush=True)
