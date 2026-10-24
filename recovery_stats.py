from pathlib import Path
import json
from safetensors.torch import load_file
r=Path('/home/xiaolongwu/vla_workspace/vla_interview_project/recovery_20261021')
base=Path('/home/xiaolongwu/vla_workspace/hf_cache/hub/models--lerobot--smolvla_base/snapshots/5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6')
lib=Path('/home/xiaolongwu/vla_workspace/hf_cache/hub/models--lerobot--smolvla_libero/snapshots/a24f1f380461b665158d52acd140152c79e8b36d')
b=Path(json.loads(next(Path('/home/xiaolongwu/vla_workspace/vla_interview_project/vla_training_ablation_v4/runs').glob('B_formal_*/result.json')).read_text())['checkpoint'])
result={}
for name,path in [('base',base),('official_libero',lib),('v4_B',b)]:
 d={}
 for pattern in ['*normalizer*.safetensors','*unnormalizer*.safetensors']:
  for p in path.glob(pattern):
   d[p.name]={k:v.tolist() for k,v in load_file(str(p)).items() if k.startswith(('action.','observation.state.'))}
 result[name]=d
(r/'normalization_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
