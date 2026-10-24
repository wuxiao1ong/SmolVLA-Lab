from pathlib import Path
import subprocess,json,time,shutil,os
R=Path('/home/xiaolongwu/vla_workspace/outputs/two_vlm_layers_v8');R.mkdir(exist_ok=True)
OLD=Path('/home/xiaolongwu/vla_workspace/outputs/base_single_task_v5')
PY='/home/xiaolongwu/vla_workspace/env/bin/python'
(R/'metadata/v4_module_audit').mkdir(parents=True,exist_ok=True)
for name in ['train_episodes.json','val_episodes.json']:shutil.copy2(OLD/'metadata'/name,R/'metadata'/name)
shutil.copy2(OLD/'metadata/v4_module_audit/target_summary.json',R/'metadata/v4_module_audit/target_summary.json')
source=json.loads(Path('/mnt/c/VLATest/v7_results/RUN_INDEX.json').read_text())['F14']
(R/'PLAN.json').write_text(json.dumps(dict(source=source,groups={'G13_14':[13,14],'H14':[14]},expert_lr=1e-5,vlm_lr=1e-6,extra_steps_each=500,seed=42,workers=2,control='both groups independently continue same F14 model and layer14 FP32 masters; only extra unfrozen layer13 differs; fresh optimizer/scheduler',protocol='same fixed development main20 init20-39 seed6000 and second10 init40-49 seed7000; A/F14 reference reused; no fresh holdout claim',adequacy='audit episode/frame distributions and reuse saved A checkpoints for validation learning curve; no data-size training restarts'),indent=2))
def command(script,args,log,group=None):
    log=R/log
    if log.exists():log=log.with_name(log.stem+'_'+time.strftime('%H%M%S')+log.suffix)
    env=os.environ.copy()
    if group:env.update(V8_GROUP=group,V8_START=source)
    print('START',script,args,group,flush=True)
    with log.open('w') as f:p=subprocess.run([PY,str(Path(__file__).resolve().parent / script),*map(str,args)],stdout=f,stderr=subprocess.STDOUT,env=env)
    if p.returncode:
        tail=log.read_text()[-6000:];print(tail,flush=True)
        if 'out of memory' in tail.lower():raise MemoryError(str(log))
        raise RuntimeError(str(log))
    print('DONE',log,flush=True)
def stage(group,steps,batch):
    name=f'check{steps}' if steps<=3 else f'formal{steps}'
    prior=[p for p in sorted(R.glob(f'{group}_{name}_bs{batch}_w2_*')) if (p/'result.json').exists() and json.loads((p/'result.json').read_text())['status']=='passed']
    if prior:run=prior[-1]
    else:
        command('v8_train.py',[steps,batch,2],f'{group}_{name}_bs{batch}.log',group)
        run=sorted(R.glob(f'{group}_{name}_bs{batch}_w2_*'))[-1]
    result=json.loads((run/'result.json').read_text());assert result['status']=='passed' and result['reload_exact']
    assert json.loads((run/'manifest.json').read_text())['base']==source
    assert json.loads((run/'initial_weights_verified.json').read_text())['all_saved_tensors_exact']
    assert json.loads((run/'frozen_weights_verified.json').read_text())['exact']
    assert json.loads((run/'master_restore_verified.json').read_text())['rounded_source_model_exact']
    return run,result
def validate(label,ckpt):
    if not list((R/'validation').glob(label+'_*.json')):command('v8_validate.py',[label,ckpt],label+'_validation.log')
def assess(label,ckpt,n,offset,seed):
    prior=[p for p in sorted((R/'eval').glob(f'{label}_{n}ep_init{offset}_*')) if (p/'result.json').exists()]
    if prior:ev=prior[-1]
    else:
        command('v8_eval.py',[label,ckpt,n,offset,20,seed],f'{label}_{n}ep_init{offset}.log')
        ev=sorted((R/'eval').glob(f'{label}_{n}ep_init{offset}_*'))[-1]
    m=json.loads((ev/'manifest.json').read_text());assert m['policy']==str(ckpt) and m['seed']==seed
    states=[json.loads(l)['init_state_index'] for l in (ev/'initial_states.jsonl').read_text().splitlines()]
    assert states==list(range(offset,offset+n)) and json.loads((ev/'dense_verified.json').read_text())['exact']
    result=json.loads((ev/'result.json').read_text());assert len(result['successes'])==n
    print('EVAL',label,n,sum(result['successes']),flush=True)
    return dict(path=str(ev),**result)
batch=24
try:
    for steps in [1,3]:
        _,result=stage('G13_14',steps,batch)
        if result['peak_reserved_gib']>7.3:raise MemoryError('capacity reserve above7.3GiB')
except MemoryError:
    batch=16
    for steps in [1,3]:stage('G13_14',steps,batch)
(R/'selected_batch.json').write_text(json.dumps(dict(batch=batch,num_workers=2,applies_to_both_groups=True)))
summary=json.loads((R/'assessment_summary.json').read_text()) if (R/'assessment_summary.json').exists() else {}
for group in ['G13_14','H14']:
    if group in summary:continue
    for steps in [1,3,500]:run,result=stage(group,steps,batch)
    ckpt=Path(result['checkpoint']);validate(group+'_500',ckpt)
    main=assess(group+'_500',ckpt,20,20,6000);second=assess(group+'_500',ckpt,10,40,7000)
    summary[group]=dict(group=group,run=str(run),checkpoint=str(ckpt),source=source,extra_steps=500,batch=batch,main20=main,second10=second)
    (R/'assessment_summary.json').write_text(json.dumps(summary,indent=2))
    print('GROUP_COMPLETE',group,flush=True)
# These are validation-only checks of existing checkpoints, not additional training.
a_run=Path(json.loads((OLD/'assessment_summary.json').read_text())['A']['run'])
for step in [500,1000,1500]:validate(f'A_curve_{step}',a_run/f'train/checkpoints/{step:06d}/pretrained_model')
for group in ['G13_14','H14']:
    ckpt=Path(summary[group]['checkpoint']);validate(group+'_250',ckpt.parent.parent/'000250/pretrained_model')
print('V8_PIPELINE_COMPLETE',flush=True)
