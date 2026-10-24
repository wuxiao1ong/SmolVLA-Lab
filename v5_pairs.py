from pathlib import Path
import json,csv
root=Path('/home/xiaolongwu/vla_workspace/outputs/base_single_task_v5');s=json.loads((root/'assessment_summary.json').read_text());rows=[]
for i in range(20):
 r=dict(init_state_index=20+i,env_seed=6000+i)
 for g in 'ABCD':r[g]=int(s[g]['main20']['successes'][i]) if g in s else ''
 rows.append(r)
with (root/'paired_main20.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['init_state_index','env_seed','A','B','C','D']);w.writeheader();w.writerows(rows)
if 'A' in s and 'B' in s:
 print(json.dumps(dict(A_fail_states=[r['init_state_index'] for r in rows if r['A']==0],B_success_states=[r['init_state_index'] for r in rows if r['B']==1],B_recovers_A_fail=[r['init_state_index'] for r in rows if r['A']==0 and r['B']==1]),indent=2))
