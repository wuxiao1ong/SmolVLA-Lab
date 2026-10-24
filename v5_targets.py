import os,json,re
from pathlib import Path
os.environ.update(HF_HOME='/home/xiaolongwu/vla_workspace/hf_cache',HF_HUB_OFFLINE='1',CUDA_VISIBLE_DEVICES='')
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
root=Path('/home/xiaolongwu/vla_workspace/outputs/base_single_task_v5');base='/home/xiaolongwu/vla_workspace/hf_cache/hub/models--lerobot--smolvla_base/snapshots/5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6'
cfg=PreTrainedConfig.from_pretrained(base);cfg.device='cpu';model=SmolVLAPolicy.from_pretrained(base,config=cfg);assert all(p.device.type=='cpu' for p in model.parameters())
names=[n for n,_ in model.named_modules()];out=root/'metadata/module_audit_v5';out.mkdir(exist_ok=True);(out/'module_names.txt').write_text('\n'.join(names))
old=json.loads((root/'metadata/v4_module_audit/target_summary.json').read_text());summary={}
for key,spec in old.items():
 matched=[n for n in names if re.fullmatch(spec['target_regex'],n)];assert matched==spec['modules'];summary[key]=dict(spec,modules=matched,device='cpu')
 (out/f'{key}_matched.txt').write_text('\n'.join(matched))
(out/'target_summary.json').write_text(json.dumps(summary,indent=2));assert [x['matched'] for x in summary.values()]==[37,39,41];print('TARGET_AUDIT_V5_PASS',[x['matched'] for x in summary.values()])
