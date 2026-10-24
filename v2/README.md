# v2: recovered early training and evaluation scripts

These original scripts were recovered from `~/vla_workspace/audit/`. The project handoff records their output directories under the early training history (v2 corrections). The scripts themselves do not carry version tags; their placement here reflects that recorded experiment stage, not an independently preserved v2 source release.

- `eval_official_retry.sh`: official SmolVLA-LIBERO evaluation with camera-key mapping.
- `train_smoke_run3.sh`: 20-step training with camera-key mapping.
- `train_subset_500.sh`: 500-step subset training.
- `eval_smoke_checkpoint.sh`: evaluate the saved 20-step checkpoint.
- `eval_custom_subset500.sh`: evaluate the saved 500-step checkpoint.

The subset training scripts use episodes 0-7, which are mixed-task demonstrations and do not include LIBERO Object task 0. They preserve the historical preliminary experiment and must not be presented as the corrected task24 training setup. The handoff describes later task24 training, but standalone scripts for those corrected runs have not been found.

Scripts retain the original absolute environment, cache, output, and checkpoint paths. Inspect those paths and output-existence checks before running. These recovered scripts were syntax-checked; training and simulation were not rerun. Duplicate failed setup attempts, audit/report generators, and installer scripts are excluded.

[Repository overview](../README.md)
