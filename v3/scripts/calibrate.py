"""Calibrate score threshold WITHOUT reading successes or rewards."""
import argparse
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("trace", type=Path)
parser.add_argument("--quantile", type=float, default=0.5)
args = parser.parse_args()
if not 0 < args.quantile < 1:
    raise SystemExit("quantile must be in (0,1)")
rows = [json.loads(line) for line in args.trace.read_text().splitlines() if line.strip()]
scores = np.array([r["score"] for r in rows if r.get("event") == "chunk"], dtype=float)
if len(scores) < 5:
    raise SystemExit("Not enough chunks for calibration (need >=5); run more pilot episodes")
print(json.dumps({"n_chunks": int(len(scores)), "quantile": args.quantile,
                  "threshold": float(np.quantile(scores, args.quantile)),
                  "note": "calibrated from action scores only, never use held-out evaluation successes"},
                 indent=2))
