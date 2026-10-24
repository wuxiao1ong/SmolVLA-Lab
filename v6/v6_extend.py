from pathlib import Path
import json,subprocess,os,time
ROOT=Path('/home/xiaolongwu/vla_workspace/outputs/expert_continuation_v6')
OLD=Path('/home/xiaolongwu/vla_workspace/outputs/base_single_task_v5')
PY='/home/xiaolongwu/vla_workspace/env/bin/python'
first=json.loads((ROOT/'assessment_summary.json').read_text())
if not (ROOT/'E1_assessment_summary.json').exists():
    (ROOT/'E1_assessment_summary.json').write_text(json.dumps(first,indent=2))
first=json.loads((ROOT/'E1_assessment_summary.json').read_text())
env=os.environ.copy();env.update(V6_LABEL='E2',V6_START=first['checkpoint'],V6_PRIOR_STEPS='3000')
def command(script,args,log,custom=None):
    p=ROOT/log
    if p.exists():p=p.with_name(p.stem+'_'+time.strftime('%H%M%S')+p.suffix)
    print('START',script,args,flush=True)
    with p.open('w') as f:
        result=subprocess.run([PY,str(Path(__file__).resolve().parent / script),*map(str,args)],stdout=f,stderr=subprocess.STDOUT,env=custom)
    if result.returncode:
        print(p.read_text()[-5000:],flush=True);raise RuntimeError(str(p))
    print('DONE',p,flush=True)
def assess(label,ckpt,n,offset,seed):
    for ev in reversed(sorted((ROOT/'eval').glob(f'{label}_{n}ep_init{offset}_*'))):
        if (ev/'result.json').exists() and json.loads((ev/'manifest.json').read_text())['policy']==str(ckpt):break
    else:
        command('v6_eval.py',[label,ckpt,n,offset,20,seed],f'{label}_{n}ep.log')
        ev=sorted((ROOT/'eval').glob(f'{label}_{n}ep_init{offset}_*'))[-1]
    result=json.loads((ev/'result.json').read_text())
    states=[json.loads(l)['init_state_index'] for l in (ev/'initial_states.jsonl').read_text().splitlines()]
    assert states==list(range(offset,offset+n)) and len(result['successes'])==n
    assert json.loads((ev/'dense_verified.json').read_text())['exact']
    print('EVAL',label,sum(result['successes']),n,flush=True)
    return dict(path=str(ev),**result)
for steps in [1,3,1000]:
    stage=f'check{steps}' if steps<=3 else 'formal1000'
    prior=[p for p in sorted(ROOT.glob(f'E2_{stage}_bs24_w2_*')) if (p/'result.json').exists() and json.loads((p/'result.json').read_text())['status']=='passed']
    if prior:run=prior[-1]
    else:
        command('v6_train.py',[steps,24,2],f'E2_{stage}.log',env)
        run=sorted(ROOT.glob(f'E2_{stage}_bs24_w2_*'))[-1]
    r=json.loads((run/'result.json').read_text());assert r['status']=='passed' and r['reload_exact']
    assert json.loads((run/'initial_weights_verified.json').read_text())['all_saved_tensors_exact']
    assert json.loads((run/'frozen_weights_verified.json').read_text())['exact']
ckpt=Path(r['checkpoint'])
if not list((ROOT/'validation').glob('E2_Cplus2000_*.json')):
    command('v6_validate.py',['E2_Cplus2000',ckpt],'E2_validation.log')
main=assess('E2_Cplus2000',ckpt,20,20,6000)
second=dict(group='E2',run=str(run),checkpoint=str(ckpt),source='E1 additional1000 from C',additional_steps=2000,prior_steps=2000,trainable_scope=first['trainable_scope'],main20=main)
(ROOT/'E2_assessment_summary.json').write_text(json.dumps(second,indent=2))
# Select using the reused development 20, before inspecting predeclared new states.
chosen=second if sum(main['successes'])>=sum(first['main20']['successes']) else first
chosen=dict(chosen)
(ROOT/'selection.json').write_text(json.dumps(dict(rule='larger main20 count; E2 wins ties; no confirmation peeking',E1=sum(first['main20']['successes']),E2=sum(main['successes']),selected=chosen['group']),indent=2))
extra=assess(chosen['group']+'_selected_confirmation',Path(chosen['checkpoint']),10,40,7000)
a=Path(json.loads((OLD/'assessment_summary.json').read_text())['A']['checkpoint'])
a_extra=assess('A_v5_confirmation',a,10,40,7000)
chosen.update(confirmation10=extra,A_confirmation10=a_extra,selection='selected using main20 from E1 and E2; new confirmation assessed once after selection')
(ROOT/'assessment_summary.json').write_text(json.dumps(chosen,indent=2))
print('V6_EXTENSION_COMPLETE',chosen['group'],sum(chosen['main20']['successes']),20,flush=True)
