# Recent VLA inference modules on frozen SmolVLA

This experiment adapts two published inference ideas to the existing RTX 5060
8GB / LeRobot v0.6.1 setup. The VLM and flow-matching weights remain the pinned
official `lerobot/smolvla_libero` snapshot, revision
`a24f1f380461b665158d52acd140152c79e8b36d`.

Sources:

- [Adaptive Action Chunking at Inference-time for VLA Models, 2026](https://arxiv.org/abs/2604.04161)
- [Bidirectional Decoding, ICLR 2025, author project](https://bid-robot.github.io/)

## Modules

1. **Action entropy execution horizon (AAC adaptation).** Draw four candidate
   50-step chunks sequentially. At each future step estimate separate 3D
   translation/rotation covariances, each regularized with `1e-6 I`, and binary
   gripper entropy. Convert physical gripper zero into normalized model space
   using the checkpoint's exact mean/std. Select the largest upward change in
   prefix-average entropy, then bound execution to 5–25 steps. N=4, sequential
   sampling and the 25-step cap are explicit memory/control adaptations.
2. **Whole-candidate reranking (BID-inspired adaptation).** Select the candidate
   minimizing equal-weight positive-sample consensus and discounted backward
   overlap distance. Use the first six normalized action dimensions, a 10-step
   window and discount 0.8. Preserve the chosen candidate's gripper sequence.
   This omits BID's weak-policy negative contrast and is not a full BID
   reproduction. It also does not average incompatible candidate trajectories.

The implementation modifies candidate selection and queue scheduling in an
evaluation process. It keeps action preprocessing and postprocessing in the
official LeRobot path. Candidate disagreement is a proxy for uncertainty,
not a calibrated probability of grasp failure. Temporal agreement can preserve
an incorrect plan; lower action discontinuity alone does not imply success.

## Predeclared experiment

All conditions use Spatial tasks 0 and 1, base seed 3000, ten episodes per task,
20 Hz, relative actions, fixed init states and batch size 1. These task/seed
pairs were not used to choose module parameters. The official checkpoint is
LIBERO-trained, so this is not an unseen-training-task generalization claim.

| Condition | Candidate draws | Selection | Execution |
|---|---:|---|---|
| base10 | 1 | first | fixed 10 |
| sample4_fixed10 | 4 | first | fixed 10 |
| rerank10 | 4 | consensus + backward | fixed 10 |
| aac_first | 4 | first | full AAC entropy |
| aac_rerank | 4 | consensus + backward | full AAC entropy |
| aac_nogrip | 4 | first | continuous entropy only |

Auxiliary candidates use an isolated torch RNG stream. Thus base10 and
sample4_fixed10 should produce the same primary actions. Unit tests check this
exactly; the actual controls are also compared using decoded video-frame hashes.
The matched budget control is needed to distinguish a module change from an
accidental change in the primary random samples.

Hypotheses fixed before outcomes: reranking might improve consistency and
success at the same horizon/budget; entropy-based execution might improve
feedback timing; removing gripper entropy might change this benefit. The 2x2
selection/horizon comparison and gripper ablation test each claim separately.
No success increase is assumed in advance.

## Reproduce

Use the existing WSL Python environment. Tests:

```bash
cd ~/vla_workspace/vla_interview_project/recent_methods
~/vla_workspace/env/bin/python -m pytest -q test_recent_controller.py
```

The independent real smoke used Object task 0, seed 1200, one episode through
both modules. The locked main plan is saved before launching:

```bash
~/vla_workspace/env/bin/python batch_recent.py \
  --split heldout \
  --methods base10 sample4_fixed10 rerank10 aac_first aac_rerank aac_nogrip \
  --suite libero_spatial --tasks 0 1 --seed 3000 --episodes 10
```

Each evaluation is a separate subprocess and only one uses the GPU at a time.
Existing incomplete outputs are rejected; completed outputs are skipped for
resumption. Every run saves the model revision, controller SHA256, settings,
raw per-episode outcomes, trace and videos. Outputs live in `outputs_recent/`.

```bash
~/vla_workspace/env/bin/python analyze_recent.py
```

The analysis reports exact model sample counts (four draws count as four),
replanning counts, synchronized sampling-block time, total evaluation time,
memory, module diagnostics and paired exploratory success differences. Timing
includes different overheads as labeled; none is called pure GPU kernel time.
There are only twenty episodes per condition. Intervals use within-task paired
resampling, assume episode independence and are not multiple-testing adjusted.

## Independent confirmation and a corrected state split

Source inspection found that this LeRobot version selects fixed initial states
by episode index, independently of the random seed. Merely changing seed does
not reserve new states. The main run uses indices 0–9; confirmation uses 10–19
and seed 4000 for both baseline and the selected method. Runtime environment
wrappers record the actual indices from each worker in initial_states.jsonl.
No permanent LeRobot source modification is made. A real AsyncVectorEnv reset
probe verified that index 10 reaches the worker before confirmation starts.

The selection protocol is saved before confirmation: select highest success
above baseline on seed 3000, tie-break by fewer actual model samples; change no
module parameters after selection. Run run_confirmation.py, then
analyze_confirmation.py. It reports confirmation separately from exploration.
