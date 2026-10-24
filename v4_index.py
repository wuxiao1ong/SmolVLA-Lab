from pathlib import Path
import json,hashlib
root=Path('/home/xiaolongwu/vla_workspace/vla_interview_project/vla_training_ablation_v4')
index={}
for group in 'BCD':
 index[group]={}
 for stage in ['check1','check3','formal']:
  run=sorted((root/'runs').glob(f'{group}_{stage}*'))[-1]
  result=json.loads((run/'result.json').read_text());assert result['status']=='passed'
  index[group][stage]=dict(run=str(run),**result)
base=Path(json.loads((Path(index['B']['formal']['run'])/'manifest.json').read_text())['base'])/'model.safetensors'
(root/'audit/base_weight_sha256.json').write_text(json.dumps(dict(path=str(base),sha256=hashlib.sha256(base.read_bytes()).hexdigest()),indent=2))
(root/'RUN_INDEX.json').write_text(json.dumps(index,indent=2))
print('NINE_TRAINING_STAGES_VERIFIED')
