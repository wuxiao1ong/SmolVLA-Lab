import sys,os,json,time,hashlib
from pathlib import Path
ROOT=Path('/home/xiaolongwu/vla_workspace/vla_interview_project/vla_training_ablation_v4')
GROUP=sys.argv[1]; N=int(sys.argv[2]); STAGE='smoke' if N==1 else 'formal'
OUT=ROOT/'eval'/f'{GROUP}_{STAGE}_{time.strftime("%Y%m%d_%H%M%S")}'
OUT.mkdir(parents=True,exist_ok=False)
os.environ.update(HF_HOME='/home/xiaolongwu/vla_workspace/hf_cache',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',TORCH_HOME='/home/xiaolongwu/vla_workspace/hf_cache/torch')
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'recent_methods'))
from state_audit import install_state_audit
install_state_audit(offset=0 if N==1 else 10,trace=OUT/'initial_states.jsonl')
if GROUP=='base':
 POLICY=ROOT/'base_eval'
else:
 run=sorted((ROOT/'runs').glob(f'{GROUP}_formal_*'))[-1]
 result=json.loads((run/'result.json').read_text()); assert result['status']=='passed' and result['reload_exact']
 POLICY=Path(result['checkpoint'])
import torch
from lerobot.scripts import lerobot_eval as ev
orig=ev.make_policy
calls=0

def factory(*args,**kwargs):
 global calls
 model=orig(*args,**kwargs)
 if GROUP!='base':
  from safetensors.torch import load_file
  from peft.utils.save_and_load import get_peft_model_state_dict
  saved=load_file(str(POLICY/'adapter_model.safetensors'))
  loaded=get_peft_model_state_dict(model)
  assert set(saved)==set(loaded)
  assert all(torch.equal(saved[n],loaded[n].detach().cpu()) for n in saved)
  (OUT/'adapter_verified.json').write_text(json.dumps(dict(exact=True,count=len(saved),checkpoint=str(POLICY),sha256=hashlib.sha256((POLICY/'adapter_model.safetensors').read_bytes()).hexdigest()),indent=2))
 original=model.select_action
 def select(*args,**kwargs):
  global calls
  if not model.get_base_model()._queues['action'] if GROUP!='base' else not model._queues['action']:
   calls+=1
  return original(*args,**kwargs)
 model.select_action=select
 return model

ev.make_policy=factory
rename={'observation.images.image':'observation.images.camera1','observation.images.image2':'observation.images.camera2'}
(OUT/'manifest.json').write_text(json.dumps(dict(group=GROUP,episodes=N,policy=str(POLICY),task='libero_object/0',seed=5000,initial_states=list(range(0,1)) if N==1 else list(range(10,10+N)),n_action_steps=10,chunk_size=50,controller='fixed10',training_init_state_overlap='unknown; not claimed held-out'),indent=2))
sys.argv=[sys.argv[0],f'--policy.path={POLICY}','--policy.device=cuda','--policy.n_action_steps=10','--env.type=libero','--env.task=libero_object','--env.task_ids=[0]','--env.control_mode=relative','--env.fps=20','--env.init_states=true','--eval.batch_size=1',f'--eval.n_episodes={N}','--env.max_parallel_tasks=1','--seed=5000',f'--rename_map={json.dumps(rename)}',f'--output_dir={OUT}']
start=time.monotonic()
ev.main()
info=json.loads((OUT/'eval_info.json').read_text()); success=info['per_task'][0]['metrics']['successes']
(OUT/'result.json').write_text(json.dumps(dict(status='passed',successes=success,elapsed_s=time.monotonic()-start,calls=calls),indent=2))
print('EVAL_PASS',GROUP,sum(success),len(success),OUT,flush=True)
