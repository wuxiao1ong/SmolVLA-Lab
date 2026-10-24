# SmolVLA inference methods

Inference controllers and LIBERO evaluation runners for frozen SmolVLA. The original experiment used RTX 5060 8GB, LeRobot v0.6.1, and `lerobot/smolvla_libero` revision `a24f1f380461b665158d52acd140152c79e8b36d`.

## Controllers

- `recent_controller.py`: adaptive action chunking and candidate reranking. Adaptive chunking samples four candidate action chunks, uses translation/rotation covariance and binary gripper entropy, and bounds the execution horizon to 5-25 steps. Reranking uses positive-sample consensus and backward overlap; it does not implement BID's weak-policy negative contrast.
- `state_audit.py`: propagates the initial-state offset to LIBERO workers and records the actual indices used. Versioned evaluation scripts also use this helper.

Candidate disagreement is an uncertainty proxy, not a calibrated failure probability. Smoother or more consistent actions alone do not establish task success.

## Evaluation

`run_recent.py` runs one method/task combination through `recent_eval_entry.py`. `batch_recent.py` runs multiple combinations sequentially. These runners require the original LeRobot environment, model-location files under `~/vla_workspace/audit/`, and the configured LIBERO data.

```bash
python recent_methods/batch_recent.py \
  --split heldout \
  --methods base10 sample4_fixed10 rerank10 aac_first aac_rerank aac_nogrip \
  --suite libero_spatial --tasks 0 1 --seed 3000 --episodes 10
```

The original development evaluations used initial states 0-9; confirmation used states 10-19. Changing the random seed alone does not reserve new initial states in this LeRobot version. Supply `--split confirmation --state-offset 10` when reproducing the latter setup, along with the intended methods and seed. The checkpoint was LIBERO-trained; these runs do not demonstrate unseen-training-task generalization.

Runs save raw evaluation outputs, settings, traces, and videos outside the source tree. Existing incomplete outputs are rejected and completed batch entries are skipped.

## Tests

```bash
python -m pytest recent_methods/test_recent_controller.py recent_methods/test_state_audit.py
```

The tests cover controller behavior, primary RNG isolation, and propagation of actual initial-state indices. Retrospective analysis, plotting, report generation, and export scripts are excluded from this directory's tracked files.
