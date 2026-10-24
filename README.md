# SmolVLA-Lab

SmolVLA training, LoRA adaptation, and LIBERO evaluation experiments, organized by version.

| Directory | VLA experiment |
| --- | --- |
| `v1/` | Initial environment stage; source snapshot not found (documented explicitly) |
| `v2/` | Recovered early baseline, subset-training, and checkpoint-evaluation scripts |
| `v3/` | Fixed/adaptive action-horizon controller, calibration, runner, and tests |
| `v4/` | LoRA ablation on the Action Expert, projections, and selected VLM layers |
| `v5/` | Single-task smolvla_base training and batch/worker variants |
| `v6/` | Merge v5 adapters and continue Action Expert training |
| `v7/` | Continue training with VLM layer 14 unfrozen |
| `v8/` | Compare training VLM layer 14 versus layers 13/14 |
| `v9/` | VLM-only LoRA on a frozen A-lite baseline |
| `recovery/` | Recovery training and closed-loop evaluation |
| `recent_methods/` | Adaptive action chunking, candidate reranking, evaluation runners, and controller tests |

Each version directory contains one copy of its training, validation, and evaluation programs. Report generation, plotting, retrospective experiment analysis, status polling, artifact export, and duplicate source snapshots are excluded from the current source tree.

## Environment and execution

The programs were developed with Python, PyTorch, LeRobot, PEFT, and LIBERO in a Windows/WSL environment with CUDA. They retain their original experiment settings, checkpoint identifiers, and external data/output paths. Review those paths before execution; this is an experiment source repository rather than an installable package.

Run a version's script directly, for example:

```bash
python v9/v9_train.py C_A14 1
python v9/v9_train.py C_A14 500
python v9/v9_validate.py C_A14_500 /path/to/checkpoint
python v9/v9_eval.py C_A14 /path/to/checkpoint
```

Versioned pipelines resolve their sibling scripts relative to their own files. Evaluations that use `state_audit.py` load it from this repository's `recent_methods/` directory.

Dataset metadata, checkpoints, and earlier experiment outputs are required inputs. In particular, v4 training requires a generated `module_audit/target_summary.json`; v5 preparation uses the v4 split and module specification; later versions reuse earlier checkpoints and metadata. Some v7-v9 scripts still consume existing external `RUN_INDEX.json` files under the original `v*_results/` paths. These are local experiment inputs and are not bundled; update their locations or supply the original outputs before running. Removing report utilities does not remove these input requirements.

The historical v4 manifest labels precision as `float32`; actual training uses native mixed BF16/FP32 with AMP disabled. Validation normalization statistics may overlap the validation split, and the reused development initial states do not constitute an independent holdout.

See each directory's README and the argument handling in its scripts for version-specific details. Controller tests can be run in an environment with PyTorch and pytest:

```bash
python -m pytest recent_methods/test_recent_controller.py recent_methods/test_state_audit.py
```

The Git allowlist includes only the curated VLA source files and README documentation. Model weights, datasets, generated results, media, logs, caches, and local auxiliary programs remain outside the tracked tree.

