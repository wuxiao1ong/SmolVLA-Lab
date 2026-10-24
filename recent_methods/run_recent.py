"""Run one locked recent-method evaluation without shell interpolation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from recent_controller import METHODS

ROOT = Path.home() / "vla_workspace"
PROJECT = ROOT / "vla_interview_project"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--split", choices=["smoke", "heldout", "confirmation"], required=True)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--state-offset", type=int, default=0)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    out = PROJECT / "outputs_recent" / args.split / args.method / f"{args.suite}_task{args.task}_seed{args.seed}"
    if out.exists():
        raise SystemExit(f"refusing existing output: {out}")
    out.mkdir(parents=True)
    policy = (ROOT / "audit/smolvla_libero_model_path.txt").read_text().strip()
    revision = (ROOT / "audit/smolvla_libero_model_revision.txt").read_text().strip()
    from safetensors.torch import load_file
    state = load_file(str(next(Path(policy).glob("*unnormalizer*.safetensors"))))
    boundary = -float(state["action.mean"][-1]) / (float(state["action.std"][-1]) + 1e-8)
    plan = dict(**vars(args), policy=policy, model_revision=revision,
                gripper_boundary=boundary, controller=METHODS[args.method],
                code_sha256=hashlib.sha256((source / "recent_controller.py").read_bytes()).hexdigest(),
                fps=20, init_states=True, control_mode="relative", n_action_steps=50,
                minimum_horizon=5, maximum_horizon=25, covariance_ridge=1e-6,
                rerank_window=10, backward_discount=0.8, consensus_weight=1, backward_weight=1)
    (out / "planned.json").write_text(json.dumps(plan, indent=2))
    env = os.environ.copy()
    env.update(HF_HOME=str(ROOT / "hf_cache"), HF_LEROBOT_HOME=str(ROOT / "hf_lerobot"),
               TORCH_HOME=str(ROOT / "hf_cache/torch"), MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
               TOKENIZERS_PARALLELISM="false")
    env["PATH"] = str(ROOT / "env/bin") + os.pathsep + env["PATH"]
    native = [f"--policy.path={policy}", "--policy.device=cuda", "--policy.n_action_steps=50",
              "--env.type=libero", f"--env.task={args.suite}", f"--env.task_ids=[{args.task}]",
              "--env.control_mode=relative", "--env.fps=20", "--env.init_states=true",
              "--eval.batch_size=1", f"--eval.n_episodes={args.episodes}", "--env.max_parallel_tasks=1",
              f"--seed={args.seed}", "--rename_map=" + json.dumps({"observation.images.image":"observation.images.camera1", "observation.images.image2":"observation.images.camera2"}),
              f"--output_dir={out}"]
    command = [sys.executable, str(source / "recent_eval_entry.py"), "--method", args.method,
               "--trace", str(out / "planner.jsonl"), "--gripper-boundary", str(boundary),
               "--seed", str(args.seed)]
    if args.split == "confirmation":
        command += ["--state-offset", str(args.state_offset), "--state-trace", str(out / "initial_states.jsonl")]
    command += ["--"] + native
    print("START", args.method, args.suite, args.task, args.seed, args.episodes, flush=True)
    with (out / "run.log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT / "lerobot", env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        print((out / "run.log").read_text()[-7000:], flush=True)
        raise SystemExit(result.returncode)
    data = json.loads((out / "eval_info.json").read_text())
    outcomes = data["per_task"][0]["metrics"]["successes"]
    events = [json.loads(line) for line in (out / "planner.jsonl").read_text().splitlines()]
    chunks = [event for event in events if event.get("event") == "chunk"]
    assert len(outcomes) == args.episodes and chunks
    assert len(data["overall"]["video_paths"]) == args.episodes
    if args.split == "confirmation":
        states = [json.loads(line)["init_state_index"] for line in (out / "initial_states.jsonl").read_text().splitlines()]
        assert states == list(range(args.state_offset,args.state_offset+args.episodes)), states
    (out / "manifest.json").write_text(json.dumps(plan, indent=2))
    print("PASS", args.method, args.task, f"{sum(outcomes)}/{len(outcomes)}", "calls", sum(c["model_calls"] for c in chunks), flush=True)


if __name__ == "__main__":
    main()
