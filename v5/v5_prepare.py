from pathlib import Path
import json,csv,subprocess,hashlib,shutil
import pyarrow.parquet as pq
root=Path('/home/xiaolongwu/vla_workspace/outputs/base_single_task_v5');m=root/'metadata';m.mkdir(parents=True,exist_ok=True)
assert not (m/'train_episodes.json').exists()
data=Path('/home/xiaolongwu/vla_workspace/hf_lerobot_full/lerobot/libero')
tasks=pq.read_table(data/'meta/tasks.parquet').to_pandas();text_by_id={int(row.task_index):str(idx) for idx,row in tasks.iterrows()}
mapping={};counts={}
for p in sorted((data/'data').rglob('*.parquet')):
 d=pq.read_table(p,columns=['episode_index','task_index']).to_pydict()
 for ep,t in zip(d['episode_index'],d['task_index']):
  assert ep not in mapping or mapping[ep]==t
  mapping[ep]=t;counts[ep]=counts.get(ep,0)+1
with (m/'episode_task_map.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['episode_index','task_index','task_text','frames'])
 w.writerows((ep,t,text_by_id[t],counts[ep]) for ep,t in sorted(mapping.items()))
v4=Path('/home/xiaolongwu/vla_workspace/vla_interview_project/vla_training_ablation_v4')
manifest=json.loads(next((v4/'runs').glob('B_formal_*/manifest.json')).read_text())
for name,key in [('train_episodes.json','train'),('val_episodes.json','validation')]:
 ids=manifest[key];assert all(mapping[e]==24 for e in ids);(m/name).write_text(json.dumps(ids,indent=2))
assert not set(manifest['train'])&set(manifest['validation'])
(m/'data_audit.json').write_text(json.dumps(dict(dataset_root=str(data),dataset_revision='a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4',task_index=24,task_text=text_by_id[24],suite='libero_object',task_id=0,train_episodes=len(manifest['train']),val_episodes=len(manifest['validation']),train_frames=sum(counts[e] for e in manifest['train']),val_frames=sum(counts[e] for e in manifest['validation']),base_revision='5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6',official_control='skipped per user; already 8/10 on init10-19, recovery_20261021',epochs_2000=2000/sum(counts[e] for e in manifest['train'])),indent=2))
(m/'pip_freeze.txt').write_text(subprocess.check_output(['/home/xiaolongwu/vla_workspace/env/bin/python','-m','pip','freeze'],text=True))
(m/'lerobot_commit.txt').write_text(subprocess.check_output(['git','-C','/home/xiaolongwu/vla_workspace/lerobot','rev-parse','HEAD'],text=True))
shutil.copytree(v4/'module_audit',m/'v4_module_audit',dirs_exist_ok=True)
print((m/'data_audit.json').read_text())
