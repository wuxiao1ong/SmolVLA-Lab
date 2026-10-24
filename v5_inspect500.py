from pathlib import Path
import av
from PIL import Image,ImageDraw
root=Path('/home/xiaolongwu/vla_workspace/outputs/base_single_task_v5');run=sorted(root.glob('A_formal2000_bs24_w2_*'))[-1];p=next((run/'train/eval/videos_step_000500').rglob('*.mp4'))
with av.open(str(p)) as c:frames=[f.to_image() for f in c.decode(video=0)]
im=Image.new('RGB',(1024,280),'white');d=ImageDraw.Draw(im)
for col,i in enumerate([0,len(frames)//3,2*len(frames)//3,len(frames)-1]):im.paste(frames[i].resize((256,256)),(256*col,24));d.text((col*256+4,4),f'A batch24 step500 frame{i}',fill='black')
im.save('/mnt/c/VLATest/v5_results/A_step500_frames.jpg');print(len(frames))
