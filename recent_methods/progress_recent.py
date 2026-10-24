from pathlib import Path
import json
import sys

root = Path.home() / "vla_workspace/vla_interview_project/outputs_recent" / (sys.argv[1] if len(sys.argv)>1 else "heldout")
for plan in sorted(root.glob("*/*/planned.json")):
    folder = plan.parent
    if (folder / "manifest.json").exists():
        continue
    path = folder / "planner.jsonl"
    chunks = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") == "chunk":
                chunks.append(row)
    print(folder.parent.name, folder.name, "started_episodes", len({r["rollout"] for r in chunks}),
          "replans", len(chunks), "last_step", chunks[-1]["step"] if chunks else None)
