import os,json,shutil
from pathlib import Path
root=Path('/home/xiaolongwu/vla_workspace/vla_interview_project/vla_training_ablation_v4')
run=sorted((root/'runs').glob('B_formal*'))[-1];ckpt=Path(json.loads((run/'result.json').read_text())['checkpoint'])
base=root/'base_eval';base.mkdir(exist_ok=False)
cfg=json.loads((ckpt/'config.json').read_text());cfg['use_peft']=False
(base/'config.json').write_text(json.dumps(cfg,indent=2))
for p in ckpt.glob('policy_*'):shutil.copy2(p,base/p.name)
src=Path(json.loads((run/'manifest.json').read_text())['base'])/'model.safetensors'
(base/'model.safetensors').symlink_to(src)
