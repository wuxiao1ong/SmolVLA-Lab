from pathlib import Path
import av,numpy as np
from PIL import Image,ImageDraw
ROOT=Path('/home/xiaolongwu/vla_workspace/vla_interview_project');OUT=Path('/mnt/c/VLATest/recovery_results');OUT.mkdir(exist_ok=True)
paths={}
for group in ['base','B','C','D']:
 p=sorted((ROOT/'vla_training_ablation_v4/eval').glob(f'{group}_formal*'))[-1]
 paths[group]=next((p/'videos').rglob('eval_episode_0.mp4'))
p=sorted((ROOT/'recovery_20261021/eval').glob('official_libero_1ep*'))[-1];paths['official_libero']=next((p/'videos').rglob('*.mp4'))
sheet=Image.new('RGB',(256*3,280*len(paths)),(255,255,255));draw=ImageDraw.Draw(sheet)
for row,(name,p) in enumerate(paths.items()):
 with av.open(str(p)) as container:frames=[frame.to_image() for frame in container.decode(video=0)]
 for col,i in enumerate([0,len(frames)//2,len(frames)-1]):
  im=frames[i].resize((256,256));sheet.paste(im,(col*256,row*280+24));draw.text((col*256+5,row*280+5),f'{name} frame {i}',fill='black')
sheet.save(OUT/'rollout_comparison.jpg');print(OUT/'rollout_comparison.jpg')
