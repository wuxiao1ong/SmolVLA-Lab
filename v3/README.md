# v3: fixed and adaptive action horizons

Recovered from the actual WSL project at `~/vla_workspace/vla_interview_project/`, with the original layout preserved.

- `scripts/adaptive_controller.py`: fixed-horizon and action-volatility-based adaptive execution.
- `scripts/eval_entry.py`: controller integration with the upstream LeRobot evaluation entry point.
- `scripts/calibrate.py`: calibrate the adaptive threshold from pilot chunk scores without using success labels.
- `run_one.sh`: launch one evaluation condition.
- `tests/test_controller.py`: scoring, horizon selection, and action-queue tests.

This is the earlier single-chunk volatility controller, distinct from the multi-candidate inference methods in `recent_methods/`. Report generators and result templates are excluded.

The runner requires the existing Conda environment and LeRobot setup under `~/vla_workspace`. It writes evaluation outputs under `v3/outputs/` and rejects existing results. Example fixed-horizon evaluation:

```bash
bash v3/run_one.sh fixed10 fixed 10 5 0 libero_object 0 2000 10 /path/to/smolvla_libero
```

For adaptive evaluation, first run a pilot in `probe` mode and use `scripts/calibrate.py` to obtain the threshold; then hold it fixed for evaluation. CLI arguments are documented at the start of `run_one.sh`.

```bash
python -m pytest v3/tests/test_controller.py
```
