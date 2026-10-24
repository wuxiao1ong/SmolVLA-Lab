# v4

Initial LoRA ablation: Action Expert and projection adapters, with optional VLM layer 14 or layers 13/14.

## Programs

- `check_targets.py`
- `v4_eval.py`
- `v4_eval_pipeline.py`
- `v4_prepare_eval.py`
- `v4_train.py`
- `v4_validate.py`

These scripts preserve the original experiment settings and external data/checkpoint paths. See the root README for environment prerequisites. Training stages can depend on outputs from earlier versions.

Run `check_targets.py --model <base-checkpoint> --out <v4-output>/module_audit` before training. After training, `v4_prepare_eval.py` prepares the unadapted baseline before evaluation. The original manifest says `float32`, but training actually uses native mixed BF16/FP32 with AMP disabled.

[Repository overview](../README.md)
