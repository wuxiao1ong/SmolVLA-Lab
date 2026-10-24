#!/usr/bin/env bash
set -euo pipefail
source /home/xiaolongwu/vla_workspace/miniforge3/etc/profile.d/conda.sh
conda activate /home/xiaolongwu/vla_workspace/env
export HF_HOME=/home/xiaolongwu/vla_workspace/hf_cache
export HF_LEROBOT_HOME=/home/xiaolongwu/vla_workspace/hf_lerobot
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false
cd /home/xiaolongwu/vla_workspace/lerobot
mkdir -p /home/xiaolongwu/vla_workspace/outputs/eval_official_20261020_rename_map /home/xiaolongwu/vla_workspace/logs
printf 'N\n' | lerobot-eval --policy.path=lerobot/smolvla_libero --policy.device=cuda --env.type=libero --env.task=libero_object --env.task_ids=[0] --eval.batch_size=1 --eval.n_episodes=1 --env.max_parallel_tasks=1 --rename_map='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}' --output_dir=/home/xiaolongwu/vla_workspace/outputs/eval_official_20261020_rename_map 2>&1 | tee /home/xiaolongwu/vla_workspace/logs/eval_official_20261020_rename_map.log
test -s /home/xiaolongwu/vla_workspace/outputs/eval_official_20261020_rename_map/eval_info.json
find /home/xiaolongwu/vla_workspace/outputs/eval_official_20261020_rename_map -type f \( -name '*.mp4' -o -name '*.json' \) -print
echo OFFICIAL_EVAL_DIR=/home/xiaolongwu/vla_workspace/outputs/eval_official_20261020_rename_map | tee /home/xiaolongwu/vla_workspace/audit/official_eval_dir.txt
