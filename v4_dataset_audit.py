from pathlib import Path
import hashlib,json,subprocess
import pyarrow.parquet as pq
root=Path('/home/xiaolongwu/vla_workspace/vla_interview_project/vla_training_ablation_v4')
run=sorted((root/'runs').glob('B_formal*'))[-1]; manifest=json.loads((run/'manifest.json').read_text()); want=set(manifest['train']+manifest['validation'])
data=Path('/home/xiaolongwu/vla_workspace/hf_lerobot_full/lerobot/libero'); evidence={}; selected=[]
for p in sorted((data/'data').rglob('*.parquet')):
 t=pq.read_table(p,columns=['episode_index','task_index']).to_pydict()
 found=set(t['episode_index'])&want
 if found:
  selected.append(str(p.relative_to(data)))
  for ep,task in zip(t['episode_index'],t['task_index']):
   if ep in want:
    d=evidence.setdefault(str(ep),{'frames':0,'tasks':[]}); d['frames']+=1
    if task not in d['tasks']:d['tasks'].append(task)
assert set(map(int,evidence))==want
assert all(x['tasks']==[24] for x in evidence.values())
files=list((data/'meta').rglob('*.parquet'))+[data/'meta/info.json',data/'meta/stats.json']+[data/x for x in selected]
hashes={str(p.relative_to(data)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
meta=list((data/'.cache/huggingface').rglob('*.metadata'))
revisions={p.read_text().splitlines()[0] for p in meta}
(root/'audit/dataset.json').write_text(json.dumps(dict(episodes=evidence,sha256=hashes,cached_download_revisions=sorted(revisions)),indent=2))
print('DATA_PASS',len(evidence),sum(x['frames'] for x in evidence.values()),revisions)
