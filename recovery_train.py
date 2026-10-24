import os, sys, json, time, re, traceback
from pathlib import Path
V4=Path('/home/xiaolongwu/vla_workspace/vla_interview_project/vla_training_ablation_v4')
ROOT=V4.parent/'recovery_20261021'
SCENARIO=sys.argv[1]; GROUP='B'; STEPS=int(sys.argv[2]); STAGE={1:'check1',3:'check3',500:'formal500',2000:'formal2000',5000:'formal5000'}[STEPS]
RUN=ROOT/'runs'/f'{SCENARIO}_{GROUP}_{STAGE}_{time.strftime("%Y%m%d_%H%M%S")}'
RUN.mkdir(parents=True,exist_ok=False)
os.environ.update(HF_HOME='/home/xiaolongwu/vla_workspace/hf_cache',HF_LEROBOT_HOME='/home/xiaolongwu/vla_workspace/hf_lerobot_full',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
BASE='/home/xiaolongwu/vla_workspace/hf_cache/hub/models--lerobot--smolvla_base/snapshots/5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6'
if SCENARIO=='libero': BASE='/home/xiaolongwu/vla_workspace/hf_cache/hub/models--lerobot--smolvla_libero/snapshots/a24f1f380461b665158d52acd140152c79e8b36d'
assert SCENARIO in ['base','libero']
ALL=[813,818,825,850,852,857,860,876,877,887,905,908,934,945,978,983,984,985,1002,1011,1020,1024,1035,1074,1075,1100,1102,1108,1131,1143,1148,1151,1152,1166,1171,1187,1205,1219,1224,1225,1233,1242,1243,1245]
import random
split=ALL.copy(); random.Random(42).shuffle(split)
VAL=sorted(split[:9]); TRAIN=sorted(split[9:]); EP=TRAIN[:8] if STEPS<500 else TRAIN
specs=json.loads((V4/'module_audit/target_summary.json').read_text()); target=list(specs.values())['BCD'.index(GROUP)]['target_regex']
(RUN/'manifest.json').write_text(json.dumps(dict(group=GROUP,stage=STAGE,steps=STEPS,base=BASE,train=TRAIN,validation=VAL,used_episodes=EP,seed=42,rank=8,alpha=16,target=target,precision='native mixed BF16/FP32; AMP off',scenario=SCENARIO,batch=1,lr=1e-4,lerobot_commit='7e241bd630a3719a56157a497ce5d08f244784f1'),indent=2))
import torch
from lerobot.scripts import lerobot_train as tr
from lerobot.policies.factory import dataset_to_policy_features
from lerobot.configs.types import FeatureType
orig_make=tr.make_policy
holder={}; rows=[]
def make_policy(*args,**kwargs):
 cfg=kwargs['cfg']; meta=kwargs['ds_meta']
 feats=dataset_to_policy_features(meta.features)
 rename=kwargs.get('rename_map') or {}
 cfg.input_features={rename.get(k,k):v for k,v in feats.items() if v.type!=FeatureType.ACTION}
 holder['meta']=meta
 return orig_make(*args,**kwargs)
tr.make_policy=make_policy
orig_optim=tr.make_optimizer_and_scheduler
def category(n):
 if '.vlm.model.text_model.layers.' in n:
  return 'vlm'+n.split('.text_model.layers.')[1].split('.')[0]
 return 'expert' if '.lm_expert.' in n else 'projections'
def build(cfg,policy):
 params={n:p for n,p in policy.named_parameters() if p.requires_grad}
 assert params and all('lora_' in n for n in params)
 cats={category(n) for n in params}
 assert cats==({'expert','projections'} | ({'vlm14'} if GROUP=='C' else {'vlm13','vlm14'} if GROUP=='D' else set())),cats
 opt,sch=orig_optim(cfg,policy)
 ids={id(p) for g in opt.param_groups for p in g['params']}
 assert all(id(p) in ids for p in params.values())
 (RUN/'trainable.json').write_text(json.dumps(dict(trainable=sum(p.numel() for p in params.values()),total=sum(p.numel() for p in policy.parameters()),parameters={n:list(p.shape) for n,p in params.items()},optimizer_all_targets=True),indent=2))
 holder.update(policy=policy,params=params)
 def pre(opt,args,kwargs):
  record={}; before={}
  for n,p in params.items():
   g=p.grad
   record[n]=dict(grad_none=g is None,grad_finite=bool(torch.isfinite(g).all()) if g is not None else False,grad_norm=float(g.norm()) if g is not None else None)
   before[n]=p.detach().clone()
  holder.update(record=record,before=before)
 def post(opt,args,kwargs):
  record=holder['record']; before=holder.pop('before'); sums={}
  for n,p in params.items():
   delta=float((p.detach()-before[n]).abs().max()); record[n]['max_update']=delta
   cat=category(n); s=sums.setdefault(cat,dict(nonzero_grad=0,updated=0))
   assert record[n]['grad_finite'],(n,record[n])
   s['nonzero_grad']+=int(record[n]['grad_norm']>0); s['updated']+=int(delta>0)
  assert all(s['nonzero_grad']>0 and s['updated']>0 for s in sums.values()),sums
  holder['step_audit']=dict(categories=sums,parameters=record)
 opt.register_step_pre_hook(pre); opt.register_step_post_hook(post)
 return opt,sch
tr.make_optimizer_and_scheduler=build
orig_update=tr.update_policy
def update(*args,**kwargs):
 out=orig_update(*args,**kwargs); torch.cuda.synchronize()
 row=dict(step=len(rows)+1,loss=float(out[0].loss.avg),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,**holder['step_audit'])
 rows.append(row)
 with (RUN/'steps.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
 print('AUDIT',row['step'],row['loss'],row['peak_allocated_gib'],row['categories'],flush=True)
 return out
tr.update_policy=update
rename={'observation.images.image':'observation.images.camera1','observation.images.image2':'observation.images.camera2'}
sys.argv=[sys.argv[0],f'--policy.path={BASE}','--dataset.repo_id=lerobot/libero','--dataset.root=/home/xiaolongwu/vla_workspace/hf_lerobot_full/lerobot/libero',f'--dataset.episodes={json.dumps(EP)}','--policy.freeze_vision_encoder=true','--policy.train_expert_only=true','--peft.method_type=LORA','--peft.r=8','--peft.lora_alpha=16',f'--peft.target_modules={target}','--batch_size=1','--num_workers=0',f'--steps={STEPS}',f'--policy.scheduler_warmup_steps={STEPS//20 if STEPS>=500 else 0}',f'--policy.scheduler_decay_steps={STEPS}',f'--save_freq={500 if STEPS>=2000 else STEPS}','--log_freq=1','--seed=42',f'--output_dir={RUN}/train',f'--job_name=recovery_{SCENARIO}_{GROUP}_{STAGE}','--policy.device=cuda','--policy.push_to_hub=false','--wandb.enable=false','--env_eval_freq=0',f'--rename_map={json.dumps(rename)}']
start=time.monotonic()
try:
 tr.train()
 from lerobot.configs.policies import PreTrainedConfig
 from safetensors.torch import load_file
 ckpts=sorted((RUN/'train/checkpoints').glob('*/pretrained_model'))
 ckpt=next(p for p in ckpts if p.parent.name.isdigit() and int(p.parent.name)==STEPS)
 # Free original GPU model before factory reload; expected adapters retained on CPU.
 expected={n:p.detach().cpu().clone() for n,p in holder['params'].items()}
 holder.pop('policy'); holder.pop('params'); torch.cuda.empty_cache()
 cfg=PreTrainedConfig.from_pretrained(str(ckpt)); cfg.pretrained_path=ckpt
 restored=orig_make(cfg=cfg,ds_meta=holder['meta'],rename_map=rename)
 restored_params=dict(restored.named_parameters())
 assert all(n in restored_params and torch.equal(v,restored_params[n].detach().cpu()) for n,v in expected.items())
 (RUN/'result.json').write_text(json.dumps(dict(status='passed',steps=len(rows),elapsed_s=time.monotonic()-start,checkpoint=str(ckpt),reload_exact=True,peak_allocated_gib=max(x['peak_allocated_gib'] for x in rows),peak_reserved_gib=max(x['peak_reserved_gib'] for x in rows)),indent=2))
 print('STAGE_PASS',RUN,flush=True)
except BaseException as e:
 (RUN/'result.json').write_text(json.dumps(dict(status='failed',error=repr(e),steps=len(rows),elapsed_s=time.monotonic()-start,traceback=traceback.format_exc()),indent=2)); raise


