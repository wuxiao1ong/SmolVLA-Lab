import sys,os,json,time
from pathlib import Path
ROOT=Path('/home/xiaolongwu/vla_workspace/vla_interview_project/vla_training_ablation_v4')
GROUP=sys.argv[1]
os.environ.update(HF_HOME='/home/xiaolongwu/vla_workspace/hf_cache',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
run=sorted((ROOT/'runs').glob('B_formal*' if GROUP=='base' else f'{GROUP}_formal*'))[-1]
manifest=json.loads((run/'manifest.json').read_text());ckpt=ROOT/'base_eval' if GROUP=='base' else Path(json.loads((run/'result.json').read_text())['checkpoint'])
import torch,numpy as np
from torch.utils.data import default_collate
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy,make_pre_post_processors
cfg=PreTrainedConfig.from_pretrained(str(ckpt));cfg.pretrained_path=ckpt
pipeline=TrainPipelineConfig(dataset=DatasetConfig(repo_id='lerobot/libero',root='/home/xiaolongwu/vla_workspace/hf_lerobot_full/lerobot/libero',episodes=manifest['validation']),policy=cfg)
ds=make_dataset(pipeline)
rename={'observation.images.image':'observation.images.camera1','observation.images.image2':'observation.images.camera2'}
model=make_policy(cfg=cfg,ds_meta=ds.meta,rename_map=rename);model.eval()
from collections import Counter
dtypes=Counter(); train_dtypes=Counter()
for n,p in model.named_parameters():
 dtypes[str(p.dtype)]+=p.numel()
 if 'lora_' in n:train_dtypes[str(p.dtype)]+=p.numel()
(ROOT/'audit'/f'dtypes_{GROUP}.json').write_text(json.dumps(dict(parameter_counts=dict(dtypes),lora_parameter_counts=dict(train_dtypes),examples={n:str(p.dtype) for n,p in model.named_parameters() if n.endswith(('q_proj.weight','state_proj.weight','lora_A.default.weight'))},use_amp=cfg.use_amp),indent=2))
pre,_=make_pre_post_processors(policy_cfg=cfg,pretrained_path=str(ckpt),preprocessor_overrides={'device_processor':{'device':'cuda'},'rename_observations_processor':{'rename_map':rename}})
# Nine held-out trajectories, eight fixed uniformly spread frames per trajectory.
eps=np.asarray(ds.hf_dataset['episode_index']); indices=[]
for ep in manifest['validation']:
 positions=np.flatnonzero(eps==ep);indices.extend(int(i) for i in positions[np.linspace(0,len(positions)-1,8,dtype=int)])
rows=[];start=time.monotonic()
with torch.no_grad():
 for j,index in enumerate(indices):
  batch=default_collate([ds[index]])
  for key in ds.meta.camera_keys:
   if batch[key].dtype==torch.uint8:batch[key]=batch[key].float()/255
  batch=pre(batch);torch.manual_seed(6000+j);torch.cuda.manual_seed_all(6000+j)
  loss,_=model.forward(batch)
  rows.append(dict(index=index,episode=int(eps[index]),loss=float(loss)))
result=dict(group=GROUP,n=len(rows),mean_loss=float(np.mean([r['loss'] for r in rows])),per_episode={str(ep):float(np.mean([r['loss'] for r in rows if r['episode']==ep])) for ep in manifest['validation']},rows=rows,elapsed_s=time.monotonic()-start,checkpoint=str(ckpt))
(ROOT/f'validation_{GROUP}.json').write_text(json.dumps(result,indent=2))
print('VALIDATION_PASS',GROUP,result['mean_loss'])

