from pathlib import Path
import json,re,ast
import torch
from safetensors import safe_open
root=Path('/home/xiaolongwu/vla_workspace/outputs/base_single_task_v5');allchecks={}
for group in 'ABCD':
 log=root/('A_bs24_w2_formal2000.log' if group=='A' else f'{group}_bs24_w2_2000.log')
 if not log.exists():continue
 step=None;rows=[]
 for line in log.read_text().splitlines():
  m=re.search(r'Eval policy at step (\d+)',line)
  if m:step=int(m[1])
  if 'Suite overall aggregated:' in line:rows.append(dict(step=step,**ast.literal_eval(line.split('Suite overall aggregated:',1)[1].strip())))
 allchecks[group]=rows
(root/'metadata/native_functional_checks.json').write_text(json.dumps(allchecks,indent=2))
run=sorted(root.glob('A_formal2000_bs24_w2_*'))[-1];res=json.loads((run/'result.json').read_text());params=json.loads((run/'trainable.json').read_text())['parameters'];m=json.loads((run/'manifest.json').read_text());rows=[];frozen_equal=True;frozen_checked=0
with safe_open(str(Path(m['base'])/'model.safetensors'),framework='pt',device='cpu') as a,safe_open(str(Path(res['checkpoint'])/'model.safetensors'),framework='pt',device='cpu') as b:
 common=set(a.keys())&set(b.keys())
 for name in sorted(common):
  if name in params:
   old=a.get_tensor(name);new=b.get_tensor(name);changed=int(torch.count_nonzero(old!=new));rows.append(dict(name=name,numel=old.numel(),changed=changed,max_delta=float((old.float()-new.float()).abs().max())))
  elif '.vlm_with_expert.vlm.' in name:
   frozen_equal=frozen_equal and torch.equal(a.get_tensor(name),b.get_tensor(name));frozen_checked+=1
report=dict(trainable_parameters_covered=len(rows),expected_trainable_tensors=len(params),changed_tensors=sum(r['changed']>0 for r in rows),changed_elements=sum(r['changed'] for r in rows),total_trainable_elements=sum(r['numel'] for r in rows),frozen_vlm_common_tensors_checked=frozen_checked,frozen_vlm_identical=frozen_equal,parameters=rows)
assert len(rows)==len(params);assert frozen_equal;report['unchanged_tensors']=[r for r in rows if r['changed']==0];print('UNCHANGED',json.dumps(report['unchanged_tensors']))
(root/'metadata/A_dense_weight_update_audit.json').write_text(json.dumps(report,indent=2));print('DENSE_AUDIT_PASS',len(rows),report['changed_elements'],frozen_checked);print('NATIVE_CHECKS',[(g,[(x['step'],x['pc_success']) for x in s]) for g,s in allchecks.items()])

