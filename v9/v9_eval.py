import sys, os, json, time, hashlib
from pathlib import Path
label=sys.argv[1];policy=Path(sys.argv[2])
root=Path('/home/xiaolongwu/vla_workspace/outputs/alite_vlm_lora_v9')
out=root/'eval'/label;out.mkdir(parents=True,exist_ok=False)
base=json.loads(Path('/mnt/c/VLATest/v8_results/RUN_INDEX.json').read_text())['A_reference']
os.environ.update(HF_HOME='/home/xiaolongwu/vla_workspace/hf_cache',HF_HUB_OFFLINE='1',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',TOKENIZERS_PARALLELISM='false')
import torch,numpy as np
from safetensors.torch import load_file
from lerobot.envs.libero import LiberoEnv
from lerobot.scripts import lerobot_eval as ev
cases=[(state,schedule) for schedule in (12300,12400) for state in range(20,40)]
context={'episode':-1,'chunk':0};reset_count={}
def observation_hash(obs):
    digest=hashlib.sha256()
    def add(value):
        if isinstance(value,dict):
            for key in sorted(value):digest.update(key.encode());add(value[key])
        else:
            array=np.asarray(value);digest.update(str(array.shape).encode());digest.update(str(array.dtype).encode());digest.update(array.tobytes())
    add(obs);return digest.hexdigest()
original_reset=LiberoEnv.reset
def reset(self,seed=None,**kwargs):
    index=reset_count.get(id(self),0);assert index<len(cases)
    state,schedule=cases[index];reset_count[id(self)]=index+1;self.init_state_id=state
    assert seed==6000+state-20
    result=original_reset(self,seed=seed,**kwargs)
    self._v9_case=index;self._v9_step=0
    with (out/'initial_states.jsonl').open('a') as f:f.write(json.dumps(dict(case=index,state=state,schedule=schedule,env_seed=seed,observation_sha256=observation_hash(result[0])))+'\n')
    return result
LiberoEnv.reset=reset
original_step=LiberoEnv.step
def step(self,action):
    result=original_step(self,action);self._v9_step+=1
    if self._v9_step<=2 or self._v9_step%10==0 or result[4]['is_success']:
        raw=self._env.env._get_observations()
        row=dict(case=self._v9_case,step=self._v9_step,action=np.asarray(action).tolist(),success=bool(result[4]['is_success']),positions={k:np.asarray(v).tolist() for k,v in raw.items() if k.endswith('_pos') or k=='robot0_gripper_qpos'})
        with (out/'trajectory.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    return result
LiberoEnv.step=step
original_rollout=ev.rollout
def rollout(*args,**kwargs):
    context['episode']+=1;context['chunk']=0
    state,schedule=cases[context['episode']];kwargs['seeds']=[6000+state-20];kwargs['render_callback']=None
    return original_rollout(*args,**kwargs)
ev.rollout=rollout
original_make=ev.make_policy
def make(*args,**kwargs):
    model=original_make(*args,**kwargs)
    adapter=(policy/'adapter_model.safetensors').exists()
    if adapter:
        from peft.utils.save_and_load import get_peft_model_state_dict
        saved=load_file(str(policy/'adapter_model.safetensors'));actual=get_peft_model_state_dict(model)
        assert set(saved)==set(actual) and all(torch.equal(v,actual[k].detach().cpu()) for k,v in saved.items())
        assert model.peft_config['default'].base_model_name_or_path==base
        dense=model.get_base_model()
        original=load_file(str(Path(base)/'model.safetensors'))
        frozen={k.replace('.base_layer.','.'):v for k,v in dense.state_dict().items() if 'lora_' not in k}
        assert set(original)==set(frozen)
        assert all(torch.equal(v,frozen[k].detach().cpu()) for k,v in original.items())
        del original,saved,actual,frozen
    else:
        assert str(policy)==base
        dense=model;saved=load_file(str(policy/'model.safetensors'));actual=model.state_dict()
        assert set(saved)==set(actual) and all(torch.equal(v,actual[k].detach().cpu()) for k,v in saved.items())
        del saved,actual
    (out/'weight_audit.json').write_text(json.dumps(dict(adapter=adapter,base=base,all_A_weights_exact=True,adapter_reload_exact=True if adapter else None)))
    original_chunk=dense._get_action_chunk
    def chunk(batch,noise=None,**kw):
        state,schedule=cases[context['episode']];index=context['chunk'];context['chunk']+=1
        seed=schedule*100000+state*100+index
        generator=torch.Generator(device='cuda').manual_seed(seed)
        noise=torch.randn((1,dense.config.chunk_size,dense.config.max_action_dim),generator=generator,device='cuda',dtype=torch.float32)
        if index==0:
            with (out/'noise_audit.jsonl').open('a') as f:f.write(json.dumps(dict(case=context['episode'],seed=seed,noise_sha256=hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest()))+'\n')
        return original_chunk(batch,noise=noise,**kw)
    dense._get_action_chunk=chunk
    return model
ev.make_policy=make
rename={'observation.images.image':'observation.images.camera1','observation.images.image2':'observation.images.camera2'}
(out/'manifest.json').write_text(json.dumps(dict(label=label,checkpoint=str(policy),base=base,cases=cases,purpose='predeclared repeated development states; paired initial observations and per-chunk noise; no fresh holdout claim',noise_seed_formula='schedule*100000+state*100+chunk_index',n_action_steps=10,fps=20),indent=2))
sys.argv=[sys.argv[0],f'--policy.path={policy}','--policy.device=cuda','--policy.n_action_steps=10','--env.type=libero','--env.task=libero_object','--env.task_ids=[0]','--env.control_mode=relative','--env.fps=20','--env.init_states=true','--eval.batch_size=1',f'--eval.n_episodes={len(cases)}','--env.max_parallel_tasks=1','--seed=42',f'--rename_map={json.dumps(rename)}',f'--output_dir={out}']
start=time.monotonic();ev.main()
success=json.loads((out/'eval_info.json').read_text())['per_task'][0]['metrics']['successes']
assert len(success)==len(cases)==context['episode']+1
(out/'result.json').write_text(json.dumps(dict(label=label,cases=[dict(state=s,schedule=n,success=bool(v)) for (s,n),v in zip(cases,success)],elapsed_s=time.monotonic()-start),indent=2))
print('V9_EVAL_COMPLETE',label,sum(success),len(success),flush=True)
