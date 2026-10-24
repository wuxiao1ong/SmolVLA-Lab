# v9

Freeze the A-lite baseline and train only VLM q/v LoRA adapters in layer 14 or layers 13/14.

## Programs

- `v9_eval.py`
- `v9_pipeline.py`
- `v9_train.py`
- `v9_validate.py`

These scripts preserve the original experiment settings and external data/checkpoint paths. See the root README for environment prerequisites. Training stages can depend on outputs from earlier versions.
