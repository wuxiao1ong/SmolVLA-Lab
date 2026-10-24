def main():
    import os,sys,json,time,traceback,hashlib
    from pathlib import Path
    ROOT=Path('/home/xiaolongwu/vla_workspace/outputs/expert_continuation_v6')
    GROUP='A';STEPS=int(sys.argv[1]);BATCH=int(sys.argv[2]) if len(sys.argv)>2 else 24;WORKERS=int(sys.argv[3]) if len(sys.argv)>3 else 2
    LABEL=os.environ.get('V6_LABEL','E')
    STAGE=f'check{STEPS}' if STEPS<=3 else f'formal{STEPS}'
    RUN=ROOT/f'{LABEL}_{STAGE}_bs{BATCH}_w{WORKERS}_{time.strftime("%Y%m%d_%H%M%S")}';RUN.mkdir(exist_ok=False)
    os.environ.update(HF_HOME='/home/xiaolongwu/vla_workspace/hf_cache',HF_LEROBOT_HOME='/home/xiaolongwu/vla_workspace/hf_lerobot_full',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
    BASE=os.environ.get('V6_START','/home/xiaolongwu/vla_workspace/outputs/expert_continuation_v6/C_merged_start')
    TRAIN=json.loads((ROOT/'metadata/train_episodes.json').read_text());VAL=json.loads((ROOT/'metadata/val_episodes.json').read_text());assert not set(TRAIN)&set(VAL)
    EP=TRAIN # v5 checks use complete same training split, unlike v4 eight-episode smoke.
    specs=json.loads((ROOT/'metadata/v4_module_audit/target_summary.json').read_text());target=list(specs.values())['BCD'.index(GROUP)]['target_regex'] if GROUP!='A' else None
    LR=5e-5;WARMUP=(STEPS//20 if STEPS>=500 else 0)
    manifest=dict(group='E_C_to_full_expert',steps=STEPS,base=BASE,train=TRAIN,validation=VAL,train_frames=5472,seed=42,lr=LR,warmup=WARMUP,decay=STEPS,precision='native mixed dtype; AMP off',batch=BATCH,num_workers=WORKERS,equivalent_coverage=BATCH*STEPS/5472,video_backend='torchcodec',target=target,rank=8 if GROUP!='A' else 0,alpha=16 if GROUP!='A' else 0,start='continue C2000 merged model weights; fresh optimizer/scheduler; cumulative C2000+additional steps',normalization='existing full-dataset stats; validation-stat overlap acknowledged')
    (RUN/'manifest.json').write_text(json.dumps(manifest,indent=2))
    manifest.update(group=LABEL,prior_steps=int(os.environ.get('V6_PRIOR_STEPS','2000')),start='continue exact model weights from base field; fresh optimizer/scheduler')
    (RUN/'manifest.json').write_text(json.dumps(manifest,indent=2))
    import torch
    from lerobot.scripts import lerobot_train as tr
    from lerobot.policies.factory import dataset_to_policy_features
    from lerobot.configs.types import FeatureType
    holder={};rows=[];orig_make=tr.make_policy
    
    def make(*args,**kwargs):
     cfg=kwargs['cfg'];meta=kwargs['ds_meta'];rename=kwargs.get('rename_map') or {}
     cfg.input_features={rename.get(k,k):v for k,v in dataset_to_policy_features(meta.features).items() if v.type!=FeatureType.ACTION}
     holder['meta']=meta
     policy=orig_make(*args,**kwargs)
     from safetensors.torch import load_file
     saved=load_file(str(Path(BASE)/'model.safetensors'));actual=policy.state_dict()
     assert all(k in actual and torch.equal(v,actual[k].detach().cpu()) for k,v in saved.items())
     (RUN/'initial_weights_verified.json').write_text(json.dumps(dict(all_saved_tensors_exact=True,count=len(saved),source=BASE)))
     del saved
     return policy
    tr.make_policy=make
    
    def cat(n):
     if '.vlm.model.text_model.layers.' in n:return 'vlm'+n.split('.text_model.layers.')[1].split('.')[0]
     return 'expert' if '.lm_expert.' in n else 'projections'
    orig_build=tr.make_optimizer_and_scheduler
    
    def build(cfg,policy):
     params={n:p for n,p in policy.named_parameters() if p.requires_grad};assert params
     cats={cat(n) for n in params};expected={'expert','projections'}|({'vlm14'} if GROUP=='C' else {'vlm13','vlm14'} if GROUP=='D' else set());assert cats==expected,cats
     if GROUP!='A':assert all('lora_' in n for n in params)
     else:assert not any('lora_' in n or '.vlm.model.' in n for n in params)
     opt,sch=orig_build(cfg,policy);ids={id(p) for g in opt.param_groups for p in g['params']};assert all(id(p) in ids for p in params.values())
     dtypes={}
     for p in params.values():dtypes[str(p.dtype)]=dtypes.get(str(p.dtype),0)+p.numel()
     (RUN/'trainable.json').write_text(json.dumps(dict(trainable=sum(p.numel() for p in params.values()),total=sum(p.numel() for p in policy.parameters()),dtypes=dtypes,optimizer_coverage=True,parameters={n:list(p.shape) for n,p in params.items()}),indent=2))
     def digest_frozen(model):
      h=hashlib.sha256()
      for n,p in model.named_parameters():
       if not p.requires_grad:
        h.update(n.encode());h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
      return h.hexdigest()
     holder.update(params=params,policy=policy,frozen_digest=digest_frozen(policy),digest_frozen=digest_frozen)
     audited=params if GROUP!='A' else {n:p for n,p in params.items() if (n.endswith('q_proj.weight') and ('.layers.0.' in n or '.layers.14.' in n)) or ('.vlm_with_expert.' not in n)}
     def pre(opt,args,kwargs):
      grad={};before={}
      for n,p in audited.items():
       g=p.grad;grad[n]=dict(grad_none=g is None,grad_finite=bool(torch.isfinite(g).all()) if g is not None else False,grad_norm=float(g.float().norm()) if g is not None else None)
       before[n]=p.detach().clone()
      holder.update(grad=grad,before=before,lr_used=opt.param_groups[0]['lr'])
     def post(opt,args,kwargs):
      records=holder['grad'];before=holder.pop('before');sums={}
      for n,p in audited.items():
       d=float((p.detach()-before[n]).abs().max());records[n]['max_update']=d;s=sums.setdefault(cat(n),dict(nonzero_grad=0,updated=0))
       assert records[n]['grad_finite'],(n,records[n]);s['nonzero_grad']+=int(records[n]['grad_norm']>0);s['updated']+=int(d>0)
      assert all(s['nonzero_grad']>0 and s['updated']>0 for s in sums.values()),sums
      holder['step_audit']=dict(categories=sums,parameters=records)
     opt.register_step_pre_hook(pre);opt.register_step_post_hook(post)
     return opt,sch
    tr.make_optimizer_and_scheduler=build
    orig_update=tr.update_policy
    
    def update(*args,**kwargs):
     start=time.monotonic();out=orig_update(*args,**kwargs);torch.cuda.synchronize();assert torch.isfinite(torch.tensor(out[0].grad_norm.val))
     row=dict(data_s=float(out[0].dataloading_s.val),step=len(rows)+1,loss=float(out[0].loss.val),lr=holder['lr_used'],update_s=time.monotonic()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,**holder['step_audit'])
     rows.append(row)
     with (RUN/'steps.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
     if len(rows)<=3 or len(rows)%100==0:print('AUDIT',row['step'],row['loss'],row['lr'],row['peak_allocated_gib'],row['categories'],flush=True)
     return out
    tr.update_policy=update
    rename={'observation.images.image':'observation.images.camera1','observation.images.image2':'observation.images.camera2'}
    sys.argv=[sys.argv[0],f'--policy.path={BASE}','--dataset.repo_id=lerobot/libero','--dataset.root=/home/xiaolongwu/vla_workspace/hf_lerobot_full/lerobot/libero',f'--dataset.episodes={json.dumps(EP)}','--dataset.video_backend=torchcodec','--policy.freeze_vision_encoder=true','--policy.train_expert_only=true',f'--policy.optimizer_lr={LR}',f'--batch_size={BATCH}',f'--num_workers={WORKERS}',f'--steps={STEPS}',f'--policy.scheduler_warmup_steps={WARMUP}',f'--policy.scheduler_decay_steps={STEPS}',f'--save_freq={500 if STEPS>=500 else STEPS}','--log_freq=20' if STEPS>3 else '--log_freq=1','--seed=42',f'--output_dir={RUN}/train',f'--job_name=v6_{GROUP}_{STEPS}','--policy.device=cuda','--policy.push_to_hub=false','--wandb.enable=false',f'--env_eval_freq={500 if STEPS>=500 else 0}',f'--rename_map={json.dumps(rename)}']
    if GROUP!='A':sys.argv+=['--peft.method_type=LORA','--peft.r=8','--peft.lora_alpha=16',f'--peft.target_modules={target}']
    if STEPS>=500: sys.argv+=['--env.type=libero','--env.task=libero_object','--env.task_ids=[0]','--env.control_mode=relative','--env.fps=20','--env.init_states=true','--env.max_parallel_tasks=1','--eval.n_episodes=1','--eval.batch_size=1','--policy.n_action_steps=10']
    start=time.monotonic()
    try:
     tr.train()
     from lerobot.configs.policies import PreTrainedConfig
     ckpt=next(p for p in (RUN/'train/checkpoints').glob('*/pretrained_model') if p.parent.name.isdigit() and int(p.parent.name)==STEPS)
     model=holder.pop('policy');params=holder.pop('params');expected={n:p.detach().cpu().clone() for n,p in params.items()}
     assert holder['frozen_digest']==holder['digest_frozen'](model)
     (RUN/'frozen_weights_verified.json').write_text(json.dumps(dict(exact=True,sha256=holder['frozen_digest'])))
     model.to('cpu');del params;torch.cuda.empty_cache()
     cfg=PreTrainedConfig.from_pretrained(str(ckpt));cfg.pretrained_path=ckpt
     restored=orig_make(cfg=cfg,ds_meta=holder['meta'],rename_map=rename);loaded=dict(restored.named_parameters())
     assert all(n in loaded and torch.equal(v,loaded[n].detach().cpu()) for n,v in expected.items())
     result=dict(status='passed',checkpoint=str(ckpt),steps=len(rows),elapsed_s=time.monotonic()-start,reload_exact=True,peak_allocated_gib=max(x['peak_allocated_gib'] for x in rows),peak_reserved_gib=max(x['peak_reserved_gib'] for x in rows),update_seconds=sum(x['update_s'] for x in rows),mean_last100_loss=sum(x['loss'] for x in rows[-100:])/len(rows[-100:]))
     (RUN/'result.json').write_text(json.dumps(result,indent=2));print('STAGE_PASS',RUN,flush=True)
    except BaseException as e:
     (RUN/'result.json').write_text(json.dumps(dict(status='failed',steps=len(rows),error=repr(e),traceback=traceback.format_exc()),indent=2));raise
    
    
if __name__ == '__main__':
    main()



