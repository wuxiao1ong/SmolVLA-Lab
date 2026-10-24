#!/usr/bin/env bash
# Usage: ./run_one.sh CONDITION MODE LONG SHORT THRESHOLD SUITE TASK_ID SEED EPISODES [POLICY]
# Requires completion of the companion v2 environment setup in WSL2.
set -euo pipefail
if (( $# < 9 || $# > 10 )); then
  echo "Usage: $0 CONDITION {fixed|probe|adaptive} LONG SHORT THRESHOLD SUITE TASK_ID SEED EPISODES [POLICY]" >&2
  exit 2
fi
CONDITION="$1"; MODE="$2"; LONG="$3"; SHORT="$4"; THRESHOLD="$5"
SUITE="$6"; TASK_ID="$7"; SEED="$8"; EPISODES="$9"
POLICY="${10:-lerobot/smolvla_libero}"
if [[ ! "$CONDITION" =~ ^[a-zA-Z0-9_-]+$ ]]; then echo "invalid CONDITION" >&2; exit 2; fi
if [[ ! "$SUITE" =~ ^libero_(object|spatial|goal|10)$ ]]; then echo "invalid SUITE" >&2; exit 2; fi
for n in "$LONG" "$SHORT" "$TASK_ID" "$SEED" "$EPISODES"; do
  [[ "$n" =~ ^[0-9]+$ ]] || { echo "not a nonnegative integer: $n" >&2; exit 2; }
done
(( EPISODES >= 1 && LONG >= SHORT && SHORT >= 1 && LONG <= 50 )) || { echo "invalid horizon or episodes" >&2; exit 2; }
ROOT="$(cd "$(dirname "$0")" && pwd)"
VLA_ROOT="${VLA_ROOT:-$HOME/vla_workspace}"
source "$VLA_ROOT/miniforge3/etc/profile.d/conda.sh"
conda activate "$VLA_ROOT/env"
export HF_HOME="$VLA_ROOT/hf_cache"
export HF_LEROBOT_HOME="$VLA_ROOT/hf_lerobot"
export TORCH_HOME="$VLA_ROOT/hf_cache/torch"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false
OUT="$ROOT/outputs/$CONDITION/${SUITE}_task${TASK_ID}_seed${SEED}"
if [[ -e "$OUT/manifest.json" || -e "$OUT/eval_info.json" || -e "$OUT/planner.jsonl" ]]; then
  echo "ERROR: result exists; do not overwrite: $OUT" >&2
  exit 3
fi
mkdir -p "$OUT"
# This mapping is required for the stated official checkpoint in LeRobot v0.6.1.
RENAME_MAP='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}'
cd "$VLA_ROOT/lerobot"
python "$ROOT/scripts/eval_entry.py" \
  --mode "$MODE" --long "$LONG" --short "$SHORT" \
  --threshold "$THRESHOLD" --trace "$OUT/planner.jsonl" -- \
  --policy.path="$POLICY" \
  --policy.device=cuda \
  --policy.n_action_steps=50 \
  --env.type=libero \
  --env.task="$SUITE" \
  --env.task_ids="[$TASK_ID]" \
  --env.control_mode=relative \
  --env.fps=20 \
  --env.init_states=true \
  --eval.batch_size=1 \
  --eval.n_episodes="$EPISODES" \
  --env.max_parallel_tasks=1 \
  --seed="$SEED" \
  --rename_map="$RENAME_MAP" \
  --output_dir="$OUT" 2>&1 | tee "$OUT/run.log"
test -s "$OUT/eval_info.json" || { echo "Missing eval_info.json" >&2; exit 4; }
python - "$OUT" "$CONDITION" "$SUITE" "$TASK_ID" "$SEED" "$EPISODES" "$POLICY" "$MODE" "$LONG" "$SHORT" "$THRESHOLD" <<'PY'
import json, sys
from pathlib import Path
out, condition, suite, task, seed, episodes, policy, mode, long, short, threshold = sys.argv[1:]
obj = dict(condition=condition, suite=suite, task_id=int(task), seed=int(seed),
           episodes=int(episodes), policy=policy, mode=mode,
           long=int(long), short=int(short), threshold=float(threshold),
           fps=20, init_states=True, control_mode='relative',
           eval_batch_size=1, max_parallel_tasks=1,
           n_action_steps=50, chunk_size=50,
           rename_map={"observation.images.image":"observation.images.camera1",
                       "observation.images.image2":"observation.images.camera2"},
           source='LeRobot v0.6.1')
(Path(out)/'manifest.json').write_text(json.dumps(obj,indent=2,ensure_ascii=False), encoding='utf-8')
print('PASS:', out)
PY
