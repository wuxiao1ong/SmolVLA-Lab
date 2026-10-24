from pathlib import Path
import json

from analyze_recent import ROOT, frame_digest

checked = 0
for task in (0, 1):
    paths = [ROOT / "heldout" / method / f"libero_spatial_task{task}_seed3000" for method in ("base10", "sample4_fixed10")]
    if not all((path / "manifest.json").exists() for path in paths):
        continue
    data = [json.loads((path / "eval_info.json").read_text()) for path in paths]
    assert data[0]["per_task"][0]["metrics"]["successes"] == data[1]["per_task"][0]["metrics"]["successes"]
    for left, right in zip(data[0]["overall"]["video_paths"], data[1]["overall"]["video_paths"]):
        assert frame_digest(left) == frame_digest(right), (task, left, right)
        checked += 1
print("PASS identical decoded budget-control videos", checked)
