import sys,os,json,time,hashlib
from pathlib import Path
ROOT=Path('/home/xiaolongwu/vla_workspace/vla_interview_project/recovery_20261021');ROOT.mkdir(exist_ok=True)
label=sys.argv[1];policy=Path(sys.argv[2]);n=int(sys.argv[3]);offset=int(sys.argv[4]);fps=int(sys.argv[5]) if len(sys.argv)>5 else 20
out=ROOT/'eval'/f'{label}_{n}ep_init{offset}_{time.strftime("%Y%m%d_%H%M%S")}';out.mkdir(parents=True)
os.environ.update(HF_HOME='/home/xiaolongwu/vla_workspace/hf_cache',HF_HUB_OFFLINE='1',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',TOKENIZERS_PARALLELISM='false')
sys.path.insert(0,str(ROOT.parent/'recent_methods'))
from state_audit import install_state_audit
install_state_audit(offset=offset,trace=out/'initial_states.jsonl')
import numpy as np,torch
from lerobot.envs.libero import LiberoEnv
orig_step=LiberoEnv.step
counter={}
def step(self,action):
 result=orig_step(self,action);i=counter.get(id(self),0)+1;counter[id(self)]=i
 if i<=5 or i%10==0 or result[4]['is_success']:
  raw=self._env.env._get_observations()
  row=dict(step=i,action=np.asarray(action).tolist(),success=bool(result[4]['is_success']),positions={k:np.asarray(v).tolist() for k,v in raw.items() if k.endswith('_pos') or k=='robot0_gripper_qpos'})
  with (out/'trajectory.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 return result
LiberoEnv.step=step
from lerobot.scripts import lerobot_eval as ev
orig_make=ev.make_policy
adapter=(policy/'adapter_model.safetensors').exists()
def make(*args,**kwargs):
 model=orig_make(*args,**kwargs)
 if adapter:
  from safetensors.torch import load_file
  from peft.utils.save_and_load import get_peft_model_state_dict
  saved=load_file(str(policy/'adapter_model.safetensors'));loaded=get_peft_model_state_dict(model)
  assert set(saved)==set(loaded) and all(torch.equal(v,loaded[k].detach().cpu()) for k,v in saved.items())
  (out/'adapter_verified.json').write_text(json.dumps(dict(exact=True,count=len(saved))))
 return model
ev.make_policy=make
rename={'observation.images.image':'observation.images.camera1','observation.images.image2':'observation.images.camera2'}
(out/'manifest.json').write_text(json.dumps(dict(label=label,policy=str(policy),episodes=n,offset=offset,seed=5000,task='libero_object/0',fps=fps,n_action_steps=10,adapter=adapter),indent=2))
sys.argv=[sys.argv[0],f'--policy.path={policy}','--policy.device=cuda','--policy.n_action_steps=10','--env.type=libero','--env.task=libero_object','--env.task_ids=[0]','--env.control_mode=relative',f'--env.fps={fps}','--env.init_states=true','--eval.batch_size=1',f'--eval.n_episodes={n}','--env.max_parallel_tasks=1','--seed=5000',f'--rename_map={json.dumps(rename)}',f'--output_dir={out}']
start=time.monotonic();ev.main();info=json.loads((out/'eval_info.json').read_text());success=info['per_task'][0]['metrics']['successes']
(out/'result.json').write_text(json.dumps(dict(successes=success,elapsed_s=time.monotonic()-start),indent=2));print('DIAG_EVAL',label,sum(success),n,out,flush=True)
