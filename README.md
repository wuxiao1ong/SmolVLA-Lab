# SmolVLA-Lab

Experiments on SmolVLA fine-tuning and inference strategies, evaluated on LIBERO.

## Contents

- `v4_*.py` through `v9_*.py`: preparation, training, validation, evaluation, auditing, and reporting scripts from successive experiment iterations.
- `recovery_*.py`: recovery training, evaluation, and analysis utilities.
- `diagnosis_eval.py`, `close_cd_diagnosis.py`, and `run_cd_diagnosis.sh`: diagnosis workflows.
- `recent_methods/`: inference experiments, including adaptive action chunking, candidate reranking, and controller tests. See its README for the experimental protocol.
- `v*_results/`: archived source copies associated with individual experiment versions.

## Environment and usage

These scripts were developed for a local Windows/WSL workflow using Python, PyTorch, LeRobot, SmolVLA, and LIBERO. They contain experiment-specific absolute paths, checkpoint references, and dataset locations. Review and adapt those values before running a script; this repository is a collection of experiment scripts rather than an installable package.

Refer to `recent_methods/README.md` for the environment versions and protocol used by that experiment. Other iterations may use different settings; inspect the relevant training and evaluation scripts.

Only source code and README documentation are tracked. Model weights, datasets, generated results, logs, videos, images, and Python caches remain local.
