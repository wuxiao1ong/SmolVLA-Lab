from pathlib import Path
import pyarrow.parquet as pq,numpy as np,json
root=Path('/home/xiaolongwu/vla_workspace/hf_lerobot_full/lerobot/libero')
xs=[];acts=[];first=[]
for p in (root/'data').rglob('*.parquet'):
 t=pq.read_table(p,columns=['task_index','episode_index','frame_index','observation.state','action']).to_pydict()
 for i,task in enumerate(t['task_index']):
  if task!=24:continue
  xs.append(t['observation.state'][i]);acts.append(t['action'][i])
  if t['frame_index'][i]==0:first.append(dict(episode=t['episode_index'][i],state=t['observation.state'][i],action=t['action'][i]))
x=np.array(xs);a=np.array(acts);r=dict(state_mean=x.mean(0).tolist(),state_min=x.min(0).tolist(),state_max=x.max(0).tolist(),action_mean=a.mean(0).tolist(),first=sorted(first,key=lambda x:x['episode']))
Path('/home/xiaolongwu/vla_workspace/vla_interview_project/recovery_20261021/task24_distribution.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2)[:4500])
