import sys,os,json,time
from pathlib import Path
ROOT=Path('/home/xiaolongwu/vla_workspace/outputs/two_vlm_layers_v8');LABEL=sys.argv[1];CKPT=Path(sys.argv[2])
os.environ.update(HF_HOME='/home/xiaolongwu/vla_workspace/hf_cache',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
VAL=json.loads((ROOT/'metadata/val_episodes.json').read_text())
import torch,numpy as np
from torch.utils.data import default_collate
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy,make_pre_post_processors
cfg=PreTrainedConfig.from_pretrained(str(CKPT));cfg.pretrained_path=CKPT
pipeline=TrainPipelineConfig(dataset=DatasetConfig(repo_id='lerobot/libero',root='/home/xiaolongwu/vla_workspace/hf_lerobot_full/lerobot/libero',episodes=VAL,video_backend='torchcodec'),policy=cfg)
ds=make_dataset(pipeline);rename={'observation.images.image':'observation.images.camera1','observation.images.image2':'observation.images.camera2'}
model=make_policy(cfg=cfg,ds_meta=ds.meta,rename_map=rename);model.eval()
pre,post=make_pre_post_processors(policy_cfg=cfg,pretrained_path=str(CKPT),preprocessor_overrides={'device_processor':{'device':'cuda'},'rename_observations_processor':{'rename_map':rename}})
eps=np.asarray(ds.hf_dataset['episode_index']);indices=[]
for ep in VAL:
 p=np.flatnonzero(eps==ep);indices.extend(int(i) for i in p[np.linspace(0,len(p)-1,8,dtype=int)])
rows=[];start=time.monotonic()
with torch.no_grad():
 for j,index in enumerate(indices):
  batch=default_collate([ds[index]])
  for key in ds.meta.camera_keys:
   if batch[key].dtype==torch.uint8:batch[key]=batch[key].float()/255
  batch=pre(batch);torch.manual_seed(6000+j);torch.cuda.manual_seed_all(6000+j);loss,_=model.forward(batch)
  row=dict(index=index,episode=int(eps[index]),loss=float(loss))
  if j%8==0 or j%8==4:
   model.reset();torch.manual_seed(7000+j);noise=torch.randn((1,cfg.chunk_size,cfg.max_action_dim),device='cuda');pred=model.predict_action_chunk(batch,noise=noise)
   physical=post(pred).cpu().numpy()[0];target=post(batch['action']).cpu().numpy()[0];valid=(~batch['action_is_pad']).cpu().numpy()[0]
   row.update(pred_first=physical[0].tolist(),target_first=target[0].tolist(),physical_rmse_by_dim=np.sqrt(np.mean((physical[valid]-target[valid])**2,axis=0)).tolist(),pred_min=physical.min(0).tolist(),pred_max=physical.max(0).tolist())
  rows.append(row)
result=dict(label=LABEL,checkpoint=str(CKPT),n=len(rows),mean_loss=float(np.mean([r['loss'] for r in rows])),per_episode={str(ep):float(np.mean([r['loss'] for r in rows if r['episode']==ep])) for ep in VAL},rows=rows,elapsed_s=time.monotonic()-start)
out=ROOT/'validation';out.mkdir(exist_ok=True);p=out/f'{LABEL}_{time.strftime("%Y%m%d_%H%M%S")}.json';p.write_text(json.dumps(result,indent=2));print('VAL_PASS',LABEL,result['mean_loss'],p,flush=True)



