import json,subprocess,time,shutil
from pathlib import Path
root=Path('/home/xiaolongwu/vla_workspace/outputs/alite_vlm_lora_v9');root.mkdir(exist_ok=True)
py='/home/xiaolongwu/vla_workspace/env/bin/python'
base=json.loads(Path('/mnt/c/VLATest/v8_results/RUN_INDEX.json').read_text())['A_reference']
plan=dict(base=base,groups={'C_A14':[14],'D_A13_14':[13,14]},scope='VLM-only q/v LoRA; freeze all A weights including Expert and projections',rank=8,alpha=16,lr=1e-4,batch=24,workers=2,extra_steps=500,seed=42,stages=[1,500],three_step_check='explicitly skipped by user',gate_weights='formal run independently restarts A with fresh zero-output LoRA; does not inherit 1-step check',eval_states=list(range(20,40)),eval_schedules=[12300,12400],eval_seed_formula='schedule*100000+state*100+chunk_index',evaluation='40 paired development episodes per model; same observation hashes and explicit chunk noise; no fresh holdout claim')
if (root/'PLAN.json').exists():assert json.loads((root/'PLAN.json').read_text())==plan
else:(root/'PLAN.json').write_text(json.dumps(plan,indent=2))
def command(script,args,label):
    log=root/f'{label}_{time.strftime("%H%M%S")}.log'
    print('V9_START',label,flush=True)
    with log.open('w') as f:result=subprocess.run([py,'/mnt/c/VLATest/'+script,*map(str,args)],stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:
        print(log.read_text()[-7000:],flush=True);raise RuntimeError(str(log))
    print('V9_DONE',label,flush=True)
def stage(label,steps):
    name='check1' if steps==1 else 'formal500'
    passed=[p for p in sorted(root.glob(f'{label}_{name}_*')) if (p/'result.json').exists() and json.loads((p/'result.json').read_text())['status']=='passed']
    if not passed:
        command('v9_train.py',[label,steps],label+'_'+name)
        passed=[p for p in sorted(root.glob(f'{label}_{name}_*')) if (p/'result.json').exists() and json.loads((p/'result.json').read_text())['status']=='passed']
    run=passed[-1];result=json.loads((run/'result.json').read_text())
    assert result['reload_exact'] and result['original_A_weights_frozen_exact']
    assert json.loads((run/'manifest.json').read_text())['base']==base
    assert json.loads((run/'zero_adapter_equivalence.json').read_text())['exact_loss']
    return run,result
summary={}
for label in ['C_A14','D_A13_14']:
    stage(label,1)
    run,result=stage(label,500)
    if not list((root/'validation').glob(label+'_500_*.json')):command('v9_validate.py',[label+'_500',result['checkpoint']],label+'_validation')
    summary[label]=dict(run=str(run),**result)
    (root/'training_summary.json').write_text(json.dumps(summary,indent=2))
for label,path in [('A',base)]+[(label,summary[label]['checkpoint']) for label in summary]:
    if not (root/'eval'/label/'result.json').exists():command('v9_eval.py',[label,path],label+'_eval')
print('V9_TRAIN_AND_EVAL_COMPLETE',flush=True)
