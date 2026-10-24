import sys, os, json, time
from pathlib import Path
label, policy_path, source_path = sys.argv[1:4]
policy = Path(policy_path)
out = Path('/home/xiaolongwu/vla_workspace/outputs/cd_diagnosis_20261022') / label
out.mkdir(parents=True, exist_ok=False)
os.environ.update(HF_HOME='/home/xiaolongwu/vla_workspace/hf_cache', HF_HUB_OFFLINE='1', MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', TOKENIZERS_PARALLELISM='false')
import torch
from safetensors.torch import load_file
from lerobot.envs.libero import LiberoEnv
from lerobot.scripts import lerobot_eval as ev
cases = [(state, schedule) for schedule in (8100, 9100) for state in (39, 42, 45, 49)]
context = {'episode': -1, 'chunk': 0}
reset_count = {}
original_reset = LiberoEnv.reset
def reset(self, seed=None, **kwargs):
    index = reset_count.get(id(self), 0)
    assert index < len(cases), ('unexpected reset', index)
    state, schedule = cases[index]
    reset_count[id(self)] = index + 1
    self.init_state_id = state
    expected_seed = 6019 if state == 39 else 7000 + state - 40
    assert seed == expected_seed, (seed, expected_seed)
    result = original_reset(self, seed=seed, **kwargs)
    with (out/'initial_states.jsonl').open('a') as f:
        f.write(json.dumps(dict(case=index, state=state, schedule=schedule, env_seed=seed))+'\n')
    return result
LiberoEnv.reset = reset
original_rollout = ev.rollout
def rollout(*args, **kwargs):
    context['episode'] += 1
    context['chunk'] = 0
    state, schedule = cases[context['episode']]
    kwargs['seeds'] = [6019 if state == 39 else 7000 + state - 40]
    kwargs['render_callback'] = None
    return original_rollout(*args, **kwargs)
ev.rollout = rollout
original_make = ev.make_policy
def make(*args, **kwargs):
    model = original_make(*args, **kwargs)
    saved = load_file(str(policy/'model.safetensors'))
    actual = model.state_dict()
    assert set(saved) == set(actual)
    assert all(torch.equal(v, actual[k].detach().cpu()) for k,v in saved.items())
    changed = []
    if source_path != '-':
        source = load_file(str(Path(source_path)/'model.safetensors'))
        selected = [k for k in actual if '.vlm.model.text_model.layers.13.' in k]
        assert len(selected) == 9, selected
        with torch.no_grad():
            for k in selected:
                if not torch.equal(saved[k], source[k]): changed.append(k)
                actual[k].copy_(source[k])
        assert all(torch.equal(actual[k].detach().cpu(), source[k] if k in selected else v) for k,v in saved.items())
    (out/'weight_audit.json').write_text(json.dumps(dict(original_checkpoint_exact=True, source=source_path, reset_layer=13 if source_path!='-' else None, changed=changed, all_other_weights_exact=True), indent=2))
    del saved
    if source_path != '-': del source
    original_chunk = model._get_action_chunk
    def chunk(batch, noise=None, **kw):
        state, schedule = cases[context['episode']]
        seed = schedule * 100000 + state * 100 + context['chunk']
        context['chunk'] += 1
        generator = torch.Generator(device='cuda').manual_seed(seed)
        noise = torch.randn((1, model.config.chunk_size, model.config.max_action_dim), generator=generator, device='cuda', dtype=torch.float32)
        return original_chunk(batch, noise=noise, **kw)
    model._get_action_chunk = chunk
    return model
ev.make_policy = make
rename = {'observation.images.image':'observation.images.camera1','observation.images.image2':'observation.images.camera2'}
(out/'manifest.json').write_text(json.dumps(dict(label=label, checkpoint=str(policy), reset_source=source_path, cases=cases, purpose='Selected failure cases for mechanism diagnosis, not an unbiased success-rate estimate', noise_seed_formula='schedule*100000+state*100+chunk_index', n_action_steps=10, fps=20),indent=2))
sys.argv = [sys.argv[0],f'--policy.path={policy}','--policy.device=cuda','--policy.n_action_steps=10','--env.type=libero','--env.task=libero_object','--env.task_ids=[0]','--env.control_mode=relative','--env.fps=20','--env.init_states=true','--eval.batch_size=1',f'--eval.n_episodes={len(cases)}','--env.max_parallel_tasks=1','--seed=42',f'--rename_map={json.dumps(rename)}',f'--output_dir={out}']
start=time.monotonic()
ev.main()
info=json.loads((out/'eval_info.json').read_text())
success=info['per_task'][0]['metrics']['successes']
assert context['episode']+1 == len(cases)
(out/'result.json').write_text(json.dumps(dict(label=label,cases=[dict(state=s,schedule=n,success=bool(v)) for (s,n),v in zip(cases,success)],elapsed_s=time.monotonic()-start),indent=2))
print('DIAGNOSIS_COMPLETE',label,success,flush=True)
