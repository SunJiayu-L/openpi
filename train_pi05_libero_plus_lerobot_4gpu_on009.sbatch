#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:4
#SBATCH --nodelist=gnho009
#SBATCH -J pi05_lp4g_009
#SBATCH -o /storage/yukaichengLab/lishiwen/jiayusun/openpi/trainlogs/pi05_lp4g_009_%j.log
#SBATCH -e /storage/yukaichengLab/lishiwen/jiayusun/openpi/trainlogs/pi05_lp4g_009_%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G

set -euo pipefail

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi
mkdir -p trainlogs wandb_pi05_libero_plus_lerobot_4gpu

export HF_HOME=~/.cache/huggingface
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export WANDB_DIR=/storage/yukaichengLab/lishiwen/jiayusun/openpi/wandb_pi05_libero_plus_lerobot_4gpu
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

CONFIG=pi05_libero_plus_lerobot
EXP=from_libero_plus_lerobot_4gpu_on009

echo "[INFO] job=${SLURM_JOB_ID} node=${SLURMD_NODENAME} start=$(date)"
echo "[INFO] step1 compute norm stats: ${CONFIG}"
uv run scripts/compute_norm_stats.py --config-name "${CONFIG}"

echo "[INFO] step2 train: ${CONFIG}, exp=${EXP}"
uv run scripts/train.py "${CONFIG}" \
  --exp-name "${EXP}" \
  --overwrite \
  --fsdp-devices 4

echo "[INFO] done $(date)"
