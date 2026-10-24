from pathlib import Path
import subprocess,json,time,shutil
R=Path('/home/xiaolongwu/vla_workspace/outputs/one_vlm_layer_v7');R.mkdir(exist_ok=True)
OLD=Path('/home/xiaolongwu/vla_workspace/outputs/base_single_task_v5')
PY='/home/xiaolongwu/vla_workspace/env/bin/python'
(R/'metadata/v4_module_audit').mkdir(parents=True,exist_ok=True)
for name in ['train_episodes.json','val_episodes.json']:shutil.copy2(OLD/'metadata'/name,R/'metadata'/name)
shutil.copy2(OLD/'metadata/v4_module_audit/target_summary.json',R/'metadata/v4_module_audit/target_summary.json')
source=json.loads((Path('/mnt/c/VLATest/v6_results/RUN_INDEX.json')).read_text())['recommended_checkpoint']
(R/'PLAN.json').write_text(json.dumps(dict(source=source,group='F14',full_vlm_layer=14,expert_lr=1e-5,vlm_lr=1e-6,extra_steps=500,seed=42,workers=2,optimizer='fresh AdamW; FP32 masters only for native BF16 VLM14, saved alongside checkpoints; all other model weights continue exactly',protocol='same35/9 split; main20 init20-39 seed6000 and second10 init40-49 seed7000; both already used for development, not new holdout; reference A already measured, do not rerun'),indent=2))
def command(script,args,log):
    log=R/log
    if log.exists():log=log.with_name(log.stem+'_'+time.strftime('%H%M%S')+log.suffix)
    print('START',script,args,flush=True)
    with log.open('w') as f:p=subprocess.run([PY,str(Path(__file__).resolve().parent / script),*map(str,args)],stdout=f,stderr=subprocess.STDOUT)
    if p.returncode:
        tail=log.read_text()[-6000:];print(tail,flush=True)
        if 'out of memory' in tail.lower():raise MemoryError(str(log))
        raise RuntimeError(str(log))
    print('DONE',log,flush=True)
def stage(steps,batch):
    name=f'check{steps}' if steps<=3 else f'formal{steps}'
    prior=[p for p in sorted(R.glob(f'F14_{name}_bs{batch}_w2_*')) if (p/'result.json').exists() and json.loads((p/'result.json').read_text())['status']=='passed']
    if prior:run=prior[-1]
    else:
        command('v7_train.py',[steps,batch,2],f'F14_{name}_bs{batch}.log')
        run=sorted(R.glob(f'F14_{name}_bs{batch}_w2_*'))[-1]
    result=json.loads((run/'result.json').read_text());assert result['status']=='passed' and result['reload_exact']
    assert json.loads((run/'initial_weights_verified.json').read_text())['all_saved_tensors_exact']
    assert json.loads((run/'frozen_weights_verified.json').read_text())['exact']
    return run,result
batch=24
try:
    _,one=stage(1,batch)
    if one['peak_reserved_gib']>7.3:raise MemoryError('capacity reserve above7.3GiB')
    stage(3,batch)
except MemoryError:
    batch=16;stage(1,batch);stage(3,batch)
(R/'selected_batch.json').write_text(json.dumps(dict(batch=batch,num_workers=2)))
run,result=stage(500,batch);ckpt=Path(result['checkpoint'])
if not list((R/'validation').glob('F14_plus500_*.json')):command('v7_validate.py',['F14_plus500',ckpt],'F14_validation.log')
def assess(n,offset,seed):
    label='F14_plus500'
    prior=[p for p in sorted((R/'eval').glob(f'{label}_{n}ep_init{offset}_*')) if (p/'result.json').exists()]
    if prior:ev=prior[-1]
    else:
        command('v7_eval.py',[label,ckpt,n,offset,20,seed],f'F14_{n}ep_init{offset}.log')
        ev=sorted((R/'eval').glob(f'{label}_{n}ep_init{offset}_*'))[-1]
    m=json.loads((ev/'manifest.json').read_text());assert m['policy']==str(ckpt) and m['seed']==seed
    states=[json.loads(l)['init_state_index'] for l in (ev/'initial_states.jsonl').read_text().splitlines()]
    assert states==list(range(offset,offset+n))
    assert json.loads((ev/'dense_verified.json').read_text())['exact']
    result=json.loads((ev/'result.json').read_text());assert len(result['successes'])==n
    print('EVAL',n,sum(result['successes']),flush=True)
    return dict(path=str(ev),**result)
main=assess(20,20,6000);second=assess(10,40,7000)
(R/'assessment_summary.json').write_text(json.dumps(dict(group='F14',run=str(run),checkpoint=str(ckpt),source=source,extra_steps=500,batch=batch,main20=main,second10=second),indent=2))
print('V7_PIPELINE_COMPLETE',sum(main['successes']),20,sum(second['successes']),10,flush=True)
