from pathlib import Path
import json
import argparse

import av
from PIL import Image, ImageDraw

root = Path.home() / "vla_workspace/vla_interview_project/outputs_recent"
parser = argparse.ArgumentParser()
parser.add_argument("task", type=int, nargs="?", default=0)
parser.add_argument("--episode", type=int)
parser.add_argument("--methods", nargs="+", default=["rerank10", "aac_first", "aac_rerank", "aac_nogrip"])
parser.add_argument("--split",default="heldout",choices=["heldout","confirmation"])
parser.add_argument("--seed",type=int,default=3000)
args = parser.parse_args()
task = args.task
base = root / args.split / "base10" / f"libero_spatial_task{task}_seed{args.seed}"
info = json.loads((base / "eval_info.json").read_text())
successes = info["per_task"][0]["metrics"]["successes"]
failed = [i for i, outcome in enumerate(successes) if not outcome]
print("failed episodes", task, failed)
if not failed and args.episode is None:
    raise SystemExit(0)
index = failed[0] if args.episode is None else args.episode
methods = ["base10"]
for name in args.methods:
    folder = root / args.split / name / f"libero_spatial_task{task}_seed{args.seed}"
    if (folder / "eval_info.json").exists():
        methods.append(name)
canvas = Image.new("RGB", (7*280, len(methods)*305), "white")
draw = ImageDraw.Draw(canvas)
for row, method in enumerate(methods):
    data = json.loads((root / args.split / method / f"libero_spatial_task{task}_seed{args.seed}/eval_info.json").read_text())
    video = data["overall"]["video_paths"][index]
    with av.open(video) as container:
        frames = [frame.to_image() for frame in container.decode(video=0)]
    positions = [0, 40, 80, 120, 160, 200, len(frames)-1]
    outcome = data["per_task"][0]["metrics"]["successes"][index]
    for column, frame_idx in enumerate(positions):
        if frame_idx >= len(frames) and column != len(positions)-1:
            draw.text((280*column+5,305*row+120), f"Episode ended at f{len(frames)-1}\nsuccess={outcome}", fill="black")
            continue
        actual = min(frame_idx, len(frames)-1)
        canvas.paste(frames[actual].resize((280,280)), (280*column,305*row+25))
        draw.text((280*column+2,305*row+4), f"{method} ep{index} f{actual} success={outcome}", fill="black")
output = root / f"{'confirmation_' if args.split=='confirmation' else ''}failure_task{task}_episode{index}.png"
canvas.save(output)
local = Path("/mnt/c/VLATest") / output.name
canvas.save(local)
print(output)
print(local)
