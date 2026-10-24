from pathlib import Path
import json
root=Path('/home/xiaolongwu/vla_workspace/outputs/base_single_task_v5');specs=list(json.loads((root/'metadata/module_audit_v5/target_summary.json').read_text()).values());summary={}
for i,g in enumerate('BCD'):
 runs=sorted(root.glob(f'{g}_formal2000_bs24_w2_*'))
 if not runs or not (runs[-1]/'result.json').exists():continue
 run=runs[-1];r=json.loads((run/'result.json').read_text())
 if r['status']!='passed':continue
 rows=[json.loads(l) for l in (run/'steps.jsonl').read_text().splitlines()];expected=set(specs[i]['modules']);seen=set();updated_params=set();finite=True
 for row in rows:
  for name,d in row['parameters'].items():
   finite=finite and d['grad_finite']
   if d['max_update']>0 and d['grad_norm']>0:
    seen.add(name.split('.lora_')[0].removeprefix('base_model.model.'));updated_params.add(name)
 assert finite and expected==seen,(g,expected-seen)
 summary[g]=dict(steps=len(rows),all_recorded_adapter_gradients_finite=finite,expected_modules=len(expected),modules_with_nonzero_gradient_and_weight_update=len(seen),adapter_tensors_ever_updated=len(updated_params),modules=sorted(seen))
(root/'metadata/lora_module_update_audit.json').write_text(json.dumps(summary,indent=2));print(json.dumps({g:{k:v for k,v in r.items() if k!='modules'} for g,r in summary.items()},indent=2))
