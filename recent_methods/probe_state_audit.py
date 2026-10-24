"""No policy rollout: prove offset/audit propagation to AsyncVectorEnv."""
import os
os.environ["MUJOCO_GL"]="egl"
os.environ["PYOPENGL_PLATFORM"]="egl"
from pathlib import Path
import json

import gymnasium as gym
from libero.libero import benchmark
from lerobot.envs.libero import LiberoEnv
from state_audit import install_state_audit

if __name__ == "__main__":
    path=Path.home()/"vla_workspace/vla_interview_project/outputs_recent/state_offset_probe.jsonl"
    install_state_audit(offset=10,trace=path)
    suite=benchmark.get_benchmark_dict()["libero_spatial"]()
    env=gym.vector.AsyncVectorEnv([lambda:LiberoEnv(task_suite=suite,task_id=0,task_suite_name="libero_spatial",observation_width=64,observation_height=64,obs_type="pixels_agent_pos")])
    try:
        env.reset(seed=4000)
    finally:
        env.close()
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    assert [row["init_state_index"] for row in rows]==[10],rows
    print("PASS actual AsyncVectorEnv used init_state_index=10")
