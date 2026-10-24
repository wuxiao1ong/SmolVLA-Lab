import os,json
from pathlib import Path
os.environ.update(HF_HOME='/home/xiaolongwu/vla_workspace/hf_cache',HF_HUB_OFFLINE='1')
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image,ImageDraw
import numpy as np
root=Path('/home/xiaolongwu/vla_workspace/outputs/base_single_task_v5');m=root/'metadata';train=json.loads((m/'train_episodes.json').read_text());val=json.loads((m/'val_episodes.json').read_text());ids=sorted(train+val)
ds=LeRobotDataset('lerobot/libero',root='/home/xiaolongwu/vla_workspace/hf_lerobot_full/lerobot/libero',episodes=ids,video_backend='pyav');eps=np.asarray(ds.hf_dataset['episode_index'])
for page in range(2):
 selected=ids[page*22:(page+1)*22];im=Image.new('RGB',(4*192,6*216),'white');draw=ImageDraw.Draw(im)
 for j,ep in enumerate(selected):
  idx=int(np.flatnonzero(eps==ep)[-1]);s=ds[idx];a=(s['observation.images.image'].numpy().transpose(1,2,0)*255).clip(0,255).astype(np.uint8);x=(j%4)*192;y=(j//4)*216
  im.paste(Image.fromarray(a).resize((192,192)),(x,y+24));draw.text((x+3,y+5),f'ep{ep} {"train" if ep in train else "val"} end',fill='black')
 im.save(m/f'demonstration_endpoints_{page+1}.jpg')
print('ENDPOINT_GALLERY',len(ids))
