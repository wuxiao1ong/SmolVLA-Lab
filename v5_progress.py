from pathlib import Path
import json,re,ast
root=Path('/home/xiaolongwu/vla_workspace/outputs/base_single_task_v5');summary=json.loads((root/'assessment_summary.json').read_text()) if (root/'assessment_summary.json').exists() else {}
for group in 'ABCD':
 runs=sorted(root.glob(f'{group}_formal2000_bs24_w2_*'))
 if not runs:continue
 run=runs[-1];s=[json.loads(x) for x in (run/'steps.jsonl').read_text().splitlines()] if (run/'steps.jsonl').exists() else [];r=json.loads((run/'result.json').read_text()) if (run/'result.json').exists() else {};val=sorted((root/'validation').glob(f'{group}_bs24_2000_*.json')) if (root/'validation').exists() else []
 row=dict(group=group,run=run.name,status=r.get('status','running'),step=len(s),avg_last100_loss=sum(x['loss'] for x in s[-100:])/len(s[-100:]) if s else None,lr=s[-1]['lr'] if s else None,peak_allocated=max((x['peak_allocated_gib'] for x in s),default=None),peak_reserved=max((x['peak_reserved_gib'] for x in s),default=None),samples_per_s=24/(sum(x['update_s']+x.get('data_s',0) for x in s[-20:])/len(s[-20:])) if s else None,val_loss=json.loads(val[-1].read_text())['mean_loss'] if val else None)
 if group in summary:row['main20_success']=f"{sum(summary[group]['main20']['successes'])}/20"
 print(json.dumps(row))
