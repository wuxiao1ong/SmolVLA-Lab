from pathlib import Path
import os, json, shutil, hashlib

os.environ.update(HF_HOME='/home/xiaolongwu/vla_workspace/hf_cache', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
old = Path('/home/xiaolongwu/vla_workspace/outputs/base_single_task_v5')
root = Path('/home/xiaolongwu/vla_workspace/outputs/expert_continuation_v6')
root.mkdir(exist_ok=True)
out = root/'C_merged_start'
if (root/'merge_audit.json').exists():
    assert (out/'model.safetensors').exists()
    print('REUSE_PREPARED', out)
    raise SystemExit
source = Path(json.loads((old/'assessment_summary.json').read_text())['C']['checkpoint'])
import torch
from safetensors.torch import load_file
from peft.utils.save_and_load import get_peft_model_state_dict
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy

cfg = PreTrainedConfig.from_pretrained(str(source)); cfg.device='cpu'; cfg.pretrained_path=source
meta = LeRobotDatasetMetadata('lerobot/libero', root='/home/xiaolongwu/vla_workspace/hf_lerobot_full/lerobot/libero')
rename = {'observation.images.image':'observation.images.camera1','observation.images.image2':'observation.images.camera2'}
model = make_policy(cfg=cfg, ds_meta=meta, rename_map=rename)
saved = load_file(str(source/'adapter_model.safetensors'))
loaded = get_peft_model_state_dict(model)
assert set(saved)==set(loaded) and all(torch.equal(v,loaded[k].cpu()) for k,v in saved.items())
expected = {}
for name, module in model.get_base_model().named_modules():
    if hasattr(module, 'lora_A') and 'default' in module.lora_A:
        base = module.get_base_layer().weight.detach().clone()
        delta = module.get_delta_weight('default').detach()
        # PEFT 0.18 safe_merge casts delta before the in-place addition.
        expected[name+'.weight'] = base + delta.to(base.dtype)
assert len(expected)==39
merged = model.merge_and_unload(safe_merge=True)
actual = merged.state_dict()
assert all(torch.equal(v, actual[k]) for k,v in expected.items())
assert not any('lora_' in n for n,_ in merged.named_parameters())
merged.config.use_peft=False; merged.config.pretrained_path=out
merged.save_pretrained(out)
for p in source.iterdir():
    if p.is_file() and (p.name.startswith('policy_preprocessor') or p.name.startswith('policy_postprocessor')):
        shutil.copy2(p, out/p.name)
cfg2=PreTrainedConfig.from_pretrained(str(out));cfg2.device='cpu';cfg2.pretrained_path=out
restored=make_policy(cfg=cfg2, ds_meta=meta, rename_map=rename)
rs=restored.state_dict()
assert set(actual)==set(rs) and all(torch.equal(v,rs[k]) for k,v in actual.items())
(root/'metadata/v4_module_audit').mkdir(parents=True,exist_ok=True)
for n in ['train_episodes.json','val_episodes.json']:
    shutil.copy2(old/'metadata'/n,root/'metadata'/n)
shutil.copy2(old/'metadata/v4_module_audit/target_summary.json',root/'metadata/v4_module_audit/target_summary.json')
audit=dict(source=str(source),output=str(out),source_adapter_exact=True,adapter_tensors=len(saved),merged_modules=len(expected),merge_formula_exact=True,dense_reload_all_tensors_exact=True,merge_dtype='native dtype; BF16 additions can round, not asserted functionally identical to unmerged FP32 adapters',optimizer='fresh optimizer and scheduler, model weights continued; not exact optimizer resume',plan='E: C merged weights -> full Expert and five projections; freeze vision/VLM including retained C VLM updates; batch24 workers2 LR5e-5 warmup50 additional1000 steps')
(root/'merge_audit.json').write_text(json.dumps(audit,indent=2))
print('PREPARE_PASS',json.dumps(audit),flush=True)
