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
test ! -e /home/xiaolongwu/vla_workspace/outputs/train_smoke_20261020_190101
lerobot-train --policy.path=lerobot/smolvla_base --policy.device=cuda --policy.freeze_vision_encoder=true --policy.train_expert_only=true --policy.empty_cameras=1 --policy.push_to_hub=false --rename_map='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}' --dataset.repo_id=lerobot/libero --dataset.episodes=[0,1,2,3,4,5,6,7] --dataset.video_backend=pyav --batch_size=1 --num_workers=0 --steps=20 --save_freq=20 --log_freq=5 --env_eval_freq=0 --wandb.enable=false --output_dir=/home/xiaolongwu/vla_workspace/outputs/train_smoke_20261020_190101 --job_name=smolvla_5060_8gb_smoke 2>&1 | tee /home/xiaolongwu/vla_workspace/logs/train_smoke_20261020_190101_retry.log
find /home/xiaolongwu/vla_workspace/outputs/train_smoke_20261020_190101/checkpoints -type f \( -name '*.safetensors' -o -name 'config.json' \) -print
echo SMOKE_TRAIN_DIR=/home/xiaolongwu/vla_workspace/outputs/train_smoke_20261020_190101 | tee /home/xiaolongwu/vla_workspace/audit/smoke_train_dir.txt
