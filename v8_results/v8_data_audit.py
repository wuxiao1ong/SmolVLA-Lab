from pathlib import Path
import json,csv,math
import numpy as np
import pyarrow.dataset as ds
R=Path('/home/xiaolongwu/vla_workspace/outputs/two_vlm_layers_v8')
OUT=Path('/mnt/c/VLATest/v8_results');OUT.mkdir(exist_ok=True)
data=Path('/home/xiaolongwu/vla_workspace/hf_lerobot_full/lerobot/libero')
meta=Path('/mnt/c/VLATest/v5_results/metadata')
train=json.loads((meta/'train_episodes.json').read_text());val=json.loads((meta/'val_episodes.json').read_text())
assert len(train)==35 and len(val)==9 and not set(train)&set(val)
mapping=list(csv.DictReader((meta/'episode_task_map.csv').open()))
task=[x for x in mapping if int(x['task_index'])==24]
assert len(task)==44 and {int(x['episode_index']) for x in task}==set(train)|set(val)
info=json.loads((data/'meta/info.json').read_text())
dataset=ds.dataset(str(data/'data'),format='parquet')
columns=[x for x in ['episode_index','frame_index','timestamp','task_index','observation.state','action'] if x in dataset.schema.names]
table=dataset.to_table(columns=columns,filter=ds.field('episode_index').isin(train+val))
eps=np.asarray(table['episode_index'].to_pylist());frames=np.asarray(table['frame_index'].to_pylist())
state=np.asarray(table['observation.state'].to_pylist(),dtype=np.float32)
action=np.asarray(table['action'].to_pylist(),dtype=np.float32)
assert state.shape==(6867,8) and action.shape==(6867,7)
assert np.isfinite(state).all() and np.isfinite(action).all()
if 'task_index' in columns:assert set(table['task_index'].to_pylist())=={24}
lengths={int(ep):int(np.sum(eps==ep)) for ep in sorted(set(eps))}
def describe(ids):
    mask=np.isin(eps,ids);ls=[lengths[ep] for ep in ids]
    initial=np.stack([state[np.flatnonzero(eps==ep)[np.argmin(frames[eps==ep])]] for ep in ids])
    return dict(episodes=len(ids),frames=int(mask.sum()),length_min=min(ls),length_max=max(ls),length_mean=float(np.mean(ls)),full_50_frame_windows=sum(max(n-49,0) for n in ls),state_min=state[mask].min(0).tolist(),state_max=state[mask].max(0).tolist(),state_initial_min=initial.min(0).tolist(),state_initial_max=initial.max(0).tolist(),action_min=action[mask].min(0).tolist(),action_max=action[mask].max(0).tolist(),action_std=action[mask].std(0).tolist())
t=describe(train);v=describe(val);assert t['frames']==5472 and v['frames']==1395
tm=np.isin(eps,train);vm=np.isin(eps,val)
def outside(x):
    lo=x[tm].min(0);hi=x[tm].max(0)
    return np.mean((x[vm]<lo)|(x[vm]>hi),axis=0).tolist()
plan=json.loads((R/'PLAN.json').read_text())
result=dict(task_index=24,task_text=task[0]['task_text'],dataset_revision='a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4',train=t,validation=v,metadata_fps=info.get('fps'),features=info['features'],parquet_columns=dataset.schema.names,finite_all=True,validation_state_fraction_outside_train_minmax=outside(state),validation_action_fraction_outside_train_minmax=outside(action),episode_lengths=lengths,object_pose_annotations=[n for n in dataset.schema.names if 'object' in n.lower() and ('pos' in n.lower() or 'pose' in n.lower())],limits=['35 independent recorded episodes, temporally overlapping action windows; frame count not independent scene count','task mapping and finite values verified, not a proof of diverse object placements','demonstration-to-simulator initial-state identity unknown','validation normalization uses existing global stats','data-size sufficiency cannot be proved by subset fine-tuning a checkpoint already exposed to all35 training episodes'],legacy_epochs_field='v5 metadata data_audit epochs_2000=0.365 assumed batch1; use actual run batch and steps, not that historical field',plan=plan)
(R/'data_sufficiency_audit.json').write_text(json.dumps(result,indent=2))
result['timing_alignment']=dict(dataset_metadata_fps=info.get('fps'),current_eval_control_freq=20,source_check='lerobot/envs/configs.py passes fps to control_freq; libero.py passes it to simulator',all_comparison_groups_same=True,limitation='metadata timestamps10Hz versus simulator control20Hz observed; original downsampling and per-action time semantics not reconstructed, so not proof of an error or wrong task; no timing change mid-comparison')
(R/'data_sufficiency_audit.json').write_text(json.dumps(result,indent=2))
(OUT/'data_sufficiency_audit.json').write_text(json.dumps(result,indent=2))
print('DATA_AUDIT_PASS',json.dumps(dict(train=t['episodes'],val=v['episodes'],train_frames=t['frames'],val_frames=v['frames'],metadata_fps=info.get('fps'),episode_length_range=[min(lengths.values()),max(lengths.values())],object_pose_annotation_columns=result['object_pose_annotations'])),flush=True)
