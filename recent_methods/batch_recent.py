import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=["base10", "sample4_fixed10", "rerank10", "aac_first", "aac_rerank"])
    parser.add_argument("--tasks", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--split", default="heldout")
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--state-offset", type=int, default=0)
    args = parser.parse_args()
    project = Path.home() / "vla_workspace/vla_interview_project"
    output = project / "outputs_recent"
    output.mkdir(exist_ok=True)
    plan_path = output / f"{args.split}_plan.json"
    plan = vars(args)
    if plan_path.exists():
        old_plan=json.loads(plan_path.read_text())
        old_plan.setdefault("state_offset",0)
        assert old_plan == plan, "locked plan differs"
    else:
        plan_path.write_text(json.dumps(plan, indent=2))
    for method in args.methods:
        for task in args.tasks:
            folder = output / args.split / method / f"{args.suite}_task{task}_seed{args.seed}"
            if (folder / "manifest.json").exists():
                print("ALREADY COMPLETE", method, task, flush=True)
                continue
            subprocess.run([sys.executable, str(Path(__file__).with_name("run_recent.py")),
                            "--method", method, "--split", args.split, "--suite", args.suite,
                            "--task", str(task), "--seed", str(args.seed), "--episodes", str(args.episodes),
                            "--state-offset",str(args.state_offset)], check=True)


if __name__ == "__main__":
    main()
