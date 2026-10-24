import json
from pathlib import Path
import numpy as np
root=Path('/home/xiaolongwu/vla_workspace/outputs/alite_vlm_lora_v9')
base=json.loads(Path('/mnt/c/VLATest/v5_results/evidence/A/validation.json').read_text())
baseline={r['index']:r for r in base['rows'] if 'pred_first' in r}
result={}
for label in ['C_A14','D_A13_14']:
    files=sorted((root/'validation').glob(label+'_500_*.json'))
    if not files:continue
    val=json.loads(files[-1].read_text());selected=[r for r in val['rows'] if 'pred_first' in r]
    assert {r['index'] for r in selected}==set(baseline)
    diffs=np.array([np.array(r['pred_first'])-np.array(baseline[r['index']]['pred_first']) for r in selected])
    result[label]=dict(points=len(selected),changed_first_actions=int(np.any(diffs!=0,axis=1).sum()),mean_abs_delta_by_dim=np.abs(diffs).mean(0).tolist(),max_abs_delta_by_dim=np.abs(diffs).max(0).tolist(),note='18 fixed validation observations, same explicit flow noise, first physical 7D action only; confirms output effect, not a closed-loop quality measure')
(root/'action_delta.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
