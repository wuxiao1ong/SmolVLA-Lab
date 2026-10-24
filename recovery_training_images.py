import os
os.environ['HF_HOME']='/home/xiaolongwu/vla_workspace/hf_cache';os.environ['HF_HUB_OFFLINE']='1'
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image,ImageDraw
from pathlib import Path
import numpy as np
out=Path('/mnt/c/VLATest/recovery_results');out.mkdir(exist_ok=True)
ds=LeRobotDataset('lerobot/libero',root='/home/xiaolongwu/vla_workspace/hf_lerobot_full/lerobot/libero',episodes=[813],video_backend='pyav')
image=Image.new('RGB',(768,560),'white');draw=ImageDraw.Draw(image)
for col,index in enumerate([0,len(ds)//2,len(ds)-1]):
 sample=ds[index]
 for row,key in enumerate(['observation.images.image','observation.images.image2']):
  a=sample[key].numpy().transpose(1,2,0);a=(a*255).clip(0,255).astype(np.uint8)
  image.paste(Image.fromarray(a).resize((256,256)),(col*256,row*280+24));draw.text((col*256+3,row*280+5),f'ep813 frame{index} camera{row+1}',fill='black')
image.save(out/'training_cameras.jpg');print(out/'training_cameras.jpg')
