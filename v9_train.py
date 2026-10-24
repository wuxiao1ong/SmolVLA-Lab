def main():
    import os, sys, json, time, hashlib, traceback, inspect
    from pathlib import Path
    root=Path('/home/xiaolongwu/vla_workspace/outputs/alite_vlm_lora_v9')
    root.mkdir(parents=True,exist_ok=True)
    label=sys.argv[1]; steps=int(sys.argv[2]); assert label in ('C_A14','D_A13_14')
    assert steps in (1,500), 'User explicitly skipped the 3-step check'
    layers=[14] if label=='C_A14' else [13,14]
    run=root/f'{label}_{"check1" if steps==1 else "formal500"}_{time.strftime("%Y%m%d_%H%M%S")}'
    run.mkdir(exist_ok=False)
    os.environ.update(HF_HOME='/home/xiaolongwu/vla_workspace/hf_cache',HF_LEROBOT_HOME='/home/xiaolongwu/vla_workspace/hf_lerobot_full',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    base=json.loads(Path('/mnt/c/VLATest/v8_results/RUN_INDEX.json').read_text())['A_reference']
    metadata=Path('/home/xiaolongwu/vla_workspace/outputs/base_single_task_v5/metadata')
    train=json.loads((metadata/'train_episodes.json').read_text());val=json.loads((metadata/'val_episodes.json').read_text())
    assert not set(train)&set(val) and len(train)==35 and len(val)==9
    target=r'model\.vlm_with_expert\.vlm\.model\.text_model\.layers\.'+('14' if len(layers)==1 else '(13|14)')+r'\.self_attn\.(q|v)_proj'
    manifest=dict(label=label,steps=steps,base=base,prior_steps=2000,start='exact A-lite2000 weights; new zero-output VLM-only adapters; fresh optimizer/scheduler',layers=layers,target=target,train=train,validation=val,train_frames=5472,seed=42,batch=24,num_workers=2,lr=1e-4,warmup=25 if steps==500 else 0,decay=steps,rank=8,alpha=16,dropout=0.,precision='native BF16/FP32 frozen base; FP32 LoRA; AMP off',scope='only selected VLM q/v LoRA; all original Expert/projections/VLM/vision weights frozen',normalization='same full-dataset statistics as A; validation-stat overlap acknowledged',three_step_check='skipped at explicit user request',evaluation='predeclared development states20-39; two shared chunk-noise schedules; no independent holdout claim')
    (run/'manifest.json').write_text(json.dumps(manifest,indent=2))
    import torch
    from safetensors.torch import load_file
    from lerobot.scripts import lerobot_train as tr
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import dataset_to_policy_features
    from lerobot.configs.types import FeatureType
    holder={};rows=[];original_make=tr.make_policy
    def make(*args,**kwargs):
        cfg=kwargs['cfg'];meta=kwargs['ds_meta'];rename=kwargs.get('rename_map') or {}
        cfg.input_features={rename.get(k,k):v for k,v in dataset_to_policy_features(meta.features).items() if v.type!=FeatureType.ACTION}
        holder['meta']=meta
        model=original_make(*args,**kwargs)
        saved=load_file(str(Path(base)/'model.safetensors'));actual=model.state_dict()
        assert set(saved)==set(actual)
        assert all(torch.equal(v,actual[k].detach().cpu()) for k,v in saved.items())
        (run/'initial_weights_verified.json').write_text(json.dumps(dict(source=base,all_tensors_exact=True,count=len(saved))))
        return model
    tr.make_policy=make
    def digest_frozen(model):
        h=hashlib.sha256()
        for n,p in model.named_parameters():
            if not p.requires_grad:
                h.update(n.encode());h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()
    original_build=tr.make_optimizer_and_scheduler
    def build(cfg,model):
        params={n:p for n,p in model.named_parameters() if p.requires_grad}
        assert len(params)==4*len(layers),list(params)
        assert all('lora_' in n and '.vlm.model.text_model.layers.' in n and any(f'.layers.{l}.' in n for l in layers) for n in params)
        assert all(p.dtype==torch.float32 for p in params.values())
        assert all(torch.count_nonzero(p)==0 for n,p in params.items() if '.lora_B.' in n)
        assert model.peft_config['default'].base_model_name_or_path==base
        assert model.peft_config['default'].lora_dropout==0.
        opt=cfg.optimizer.build(list(params.values()))
        sch=cfg.scheduler.build(opt,cfg.steps) if cfg.scheduler is not None else None
        ids={id(p) for group in opt.param_groups for p in group['params']}
        assert ids=={id(p) for p in params.values()}
        holder.update(policy=model,params=params,initial={n:p.detach().cpu().clone() for n,p in params.items()},frozen_digest=digest_frozen(model))
        (run/'trainable.json').write_text(json.dumps(dict(count=sum(p.numel() for p in params.values()),parameters={n:list(p.shape) for n,p in params.items()},base_path_verified=True,optimizer_exact_coverage=True),indent=2))
        def pre(opt,args,kwargs):
            holder['before']={n:p.detach().clone() for n,p in params.items()}
            holder['grads']={n:dict(finite=p.grad is not None and bool(torch.isfinite(p.grad).all()),norm=float(p.grad.float().norm()) if p.grad is not None else None) for n,p in params.items()}
            holder['lr']=opt.param_groups[0]['lr']
        def post(opt,args,kwargs):
            records=holder['grads'];before=holder.pop('before');categories={}
            for n,p in params.items():
                assert records[n]['finite'],(n,records[n])
                records[n]['max_update']=float((p.detach()-before[n]).abs().max())
                layer=n.split('.text_model.layers.')[1].split('.')[0]
                cat=categories.setdefault(layer,dict(nonzero_grad=0,updated=0))
                cat['nonzero_grad']+=int(records[n]['norm']>0);cat['updated']+=int(records[n]['max_update']>0)
            assert all(v['nonzero_grad']>0 and v['updated']>0 for v in categories.values()),categories
            holder['audit']=dict(categories=categories,parameters=records)
        opt.register_step_pre_hook(pre);opt.register_step_post_hook(post)
        return opt,sch
    tr.make_optimizer_and_scheduler=build
    original_update=tr.update_policy
    def update(*args,**kwargs):
        if not rows:
            bound=inspect.signature(original_update).bind(*args,**kwargs).arguments
            model=bound['policy'];batch=bound['batch'];model.eval()
            with torch.no_grad(),torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(12345);torch.cuda.manual_seed_all(12345);enabled=model.forward(batch)[0]
                with model.disable_adapter():
                    torch.manual_seed(12345);torch.cuda.manual_seed_all(12345);disabled=model.forward(batch)[0]
            assert torch.equal(enabled,disabled),(enabled,disabled)
            (run/'zero_adapter_equivalence.json').write_text(json.dumps(dict(exact_loss=True,enabled=float(enabled),disabled=float(disabled))))
        start=time.monotonic();result=original_update(*args,**kwargs);torch.cuda.synchronize()
        row=dict(step=len(rows)+1,loss=float(result[0].loss.val),lr=holder['lr'],update_s=time.monotonic()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,**holder['audit'])
        rows.append(row)
        with (run/'steps.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if len(rows)<=2 or len(rows)%100==0:print('V9_AUDIT',label,row['step'],row['loss'],row['peak_allocated_gib'],row['categories'],flush=True)
        return result
    tr.update_policy=update
    rename={'observation.images.image':'observation.images.camera1','observation.images.image2':'observation.images.camera2'}
    sys.argv=[sys.argv[0],f'--policy.path={base}','--dataset.repo_id=lerobot/libero','--dataset.root=/home/xiaolongwu/vla_workspace/hf_lerobot_full/lerobot/libero',f'--dataset.episodes={json.dumps(train)}','--dataset.video_backend=torchcodec','--policy.freeze_vision_encoder=true','--policy.train_expert_only=true','--policy.optimizer_lr=0.0001','--batch_size=24','--num_workers=2',f'--steps={steps}',f'--policy.scheduler_warmup_steps={25 if steps==500 else 0}',f'--policy.scheduler_decay_steps={steps}',f'--save_freq={250 if steps==500 else 1}','--log_freq=20','--seed=42',f'--output_dir={run}/train',f'--job_name=v9_{label}_{steps}','--policy.device=cuda','--policy.push_to_hub=false','--wandb.enable=false','--env_eval_freq=0',f'--rename_map={json.dumps(rename)}','--peft.method_type=LORA','--peft.r=8','--peft.lora_alpha=16',f'--peft.target_modules={target}']
    start=time.monotonic()
    try:
        tr.train()
        ckpt=run/'train/checkpoints'/f'{steps:06d}'/'pretrained_model';assert ckpt.exists()
        model=holder.pop('policy');params=holder.pop('params')
        frozen=digest_frozen(model);assert frozen==holder['frozen_digest']
        expected={n:p.detach().cpu().clone() for n,p in params.items()}
        delta={n:float((v-holder['initial'][n]).abs().max()) for n,v in expected.items()}
        model.to('cpu');del params;torch.cuda.empty_cache()
        cfg=PreTrainedConfig.from_pretrained(str(ckpt));cfg.pretrained_path=ckpt
        restored=original_make(cfg=cfg,ds_meta=holder['meta'],rename_map=rename);loaded=dict(restored.named_parameters())
        assert all(torch.equal(v,loaded[n].detach().cpu()) for n,v in expected.items())
        assert digest_frozen(restored)==frozen
        (run/'frozen_weights_verified.json').write_text(json.dumps(dict(all_original_weights_unchanged=True,reloaded_frozen_exact=True,sha256=frozen,adapter_max_deltas=delta),indent=2))
        result=dict(status='passed',checkpoint=str(ckpt),steps=len(rows),reload_exact=True,original_A_weights_frozen_exact=True,elapsed_s=time.monotonic()-start,peak_allocated_gib=max(r['peak_allocated_gib'] for r in rows),peak_reserved_gib=max(r['peak_reserved_gib'] for r in rows),mean_last100_loss=sum(r['loss'] for r in rows[-100:])/len(rows[-100:]))
        (run/'result.json').write_text(json.dumps(result,indent=2));print('V9_STAGE_PASS',run,flush=True)
    except BaseException as e:
        (run/'result.json').write_text(json.dumps(dict(status='failed',steps=len(rows),error=repr(e),traceback=traceback.format_exc()),indent=2));raise

if __name__=='__main__':main()

