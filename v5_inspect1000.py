from pathlib import Path
import av
from PIL import Image,ImageDraw
root=Path('/home/xiaolongwu/vla_workspace/outputs/base_single_task_v5');out=Path('/mnt/c/VLATest/v5_results')
im=Image.new('RGB',(1024,560),'white');d=ImageDraw.Draw(im)
for row,(g,step) in enumerate([('A',1000),('B',1000)]):
 run=sorted(root.glob(f'{g}_formal2000_bs24_w2_*'))[-1];p=next((run/f'train/eval/videos_step_{step:06}').rglob('*.mp4'))
 with av.open(str(p)) as c:frames=[f.to_image() for f in c.decode(video=0)]
 for col,i in enumerate([0,len(frames)//3,2*len(frames)//3,len(frames)-1]):
  im.paste(frames[i].resize((256,256)),(col*256,row*280+24));d.text((col*256+3,row*280+4),f'{g} step1000 frame{i}',fill='black')
im.save(out/'A_B_step1000_frames.jpg');print('VIDEO_DIAGNOSTIC_SAVED')
