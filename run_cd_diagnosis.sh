#!/bin/bash
set -euo pipefail
export PYTHONPATH=/home/xiaolongwu/vla_workspace/lerobot/src
root=/home/xiaolongwu/vla_workspace/outputs/cd_diagnosis_20261022
mkdir -p "$root"
py=/home/xiaolongwu/vla_workspace/env/bin/python
a=/home/xiaolongwu/vla_workspace/outputs/base_single_task_v5/A_formal2000_bs24_w2_20261021_171218/train/checkpoints/002000/pretrained_model
g=/home/xiaolongwu/vla_workspace/outputs/two_vlm_layers_v8/G13_14_formal500_bs24_w2_20261021_224918/train/checkpoints/000500/pretrained_model
h=/home/xiaolongwu/vla_workspace/outputs/two_vlm_layers_v8/H14_formal500_bs24_w2_20261021_230718/train/checkpoints/000500/pretrained_model
"$py" /mnt/c/VLATest/diagnosis_eval.py G "$g" - > "$root/G.log" 2>&1
"$py" /mnt/c/VLATest/diagnosis_eval.py G_reset13 "$g" "$a" > "$root/G_reset13.log" 2>&1
"$py" /mnt/c/VLATest/diagnosis_eval.py H "$h" - > "$root/H.log" 2>&1
"$py" /mnt/c/VLATest/diagnosis_eval.py A "$a" - > "$root/A.log" 2>&1
