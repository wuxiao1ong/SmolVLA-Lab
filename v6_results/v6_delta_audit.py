from pathlib import Path
import json
import torch
from safetensors.torch import load_file
torch.set_num_threads(4)
r=Path('/home/xiaolongwu/vla_workspace/outputs/expert_continuation_v6')
results={}
for prefix in ['E','E2']:
    runs=[p for p in sorted(r.glob(f'{prefix}_formal1000_bs24_w2_*')) if (p/'result.json').exists() and json.loads((p/'result.json').read_text())['status']=='passed']
    if not runs:continue
    run=runs[-1];meta=json.loads((run/'manifest.json').read_text());result=json.loads((run/'result.json').read_text())
    names=json.loads((run/'trainable.json').read_text())['parameters']
    source=load_file(str(Path(meta['base'])/'model.safetensors'))
    final=load_file(str(Path(result['checkpoint'])/'model.safetensors'))
    rows=[]
    for name in names:
        assert name in source and name in final
        n=int(torch.count_nonzero(source[name]!=final[name]))
        rows.append(dict(name=name,numel=source[name].numel(),changed_elements=n))
    results[prefix]=dict(trainable_tensors=len(rows),changed_tensors=sum(x['changed_elements']>0 for x in rows),changed_elements=sum(x['changed_elements'] for x in rows),unchanged=[x for x in rows if x['changed_elements']==0])
    del source,final
(r/'dense_delta_audit.json').write_text(json.dumps(results,indent=2))
print('DENSE_DELTA_AUDIT',json.dumps({k:{a:b for a,b in v.items() if a!='unchanged'} for k,v in results.items()}),flush=True)
