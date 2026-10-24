from pathlib import Path
import av
from PIL import Image,ImageDraw
root=Path('/home/xiaolongwu/vla_workspace/outputs')
old=root/'base_single_task_v5';new=root/'expert_continuation_v6'
items=[('C original step2000',sorted(old.glob('C_formal2000_bs24_w2_*'))[-1]/'train/eval/videos_step_002000'),('E continued step500',sorted(new.glob('E_formal1000_bs24_w2_*'))[-1]/'train/eval/videos_step_000500')]
im=Image.new('RGB',(1024,560),'white');d=ImageDraw.Draw(im)
for row,(label,folder) in enumerate(items):
    p=next(folder.rglob('*.mp4'))
    with av.open(str(p)) as container:frames=[f.to_image() for f in container.decode(video=0)]
    for col,i in enumerate([0,len(frames)//3,2*len(frames)//3,len(frames)-1]):
        im.paste(frames[i].resize((256,256)),(col*256,row*280+24))
        d.text((col*256+3,row*280+4),f'{label} frame{i}',fill='black')
im.save('/mnt/c/VLATest/v6_results/C_E_functional_frames.jpg')
print('VIDEO_CONTACT_SHEET_SAVED')
