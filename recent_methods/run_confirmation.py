import json
from pathlib import Path
import subprocess
import sys

source = Path(__file__).resolve().parent
root = Path.home() / "vla_workspace/vla_interview_project/outputs_recent"
protocol = json.loads((root / "confirmation_protocol.json").read_text())
selected = json.loads((root / "selected_method.json").read_text())["method"]
if selected is None:
    raise SystemExit("No method improved exploratory success; confirmation not launched.")
assert selected in protocol["eligible_methods"]
selection = dict(method=selected, protocol=protocol)
destination = root / "selected_confirmation.json"
if destination.exists():
    assert json.loads(destination.read_text()) == selection
else:
    destination.write_text(json.dumps(selection, indent=2))
subprocess.run([sys.executable, str(source / "batch_recent.py"), "--split", "confirmation",
                "--methods", protocol["baseline"], selected, "--suite", protocol["suite"],
                "--tasks", *map(str,protocol["tasks"]), "--seed", str(protocol["seed"]),
                "--episodes", str(protocol["episodes_per_task"]),
                "--state-offset", str(protocol["initial_state_offset"])], check=True)
