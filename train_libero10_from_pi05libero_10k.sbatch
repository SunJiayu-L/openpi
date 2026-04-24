#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:2
#SBATCH --nodelist=gnho031
#SBATCH -J train_l10_from_pi05libero_10k
#SBATCH -o trainlogs/train_l10_from_pi05libero_10k_%j.log
#SBATCH -e trainlogs/train_l10_from_pi05libero_10k_%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G

set -eo pipefail

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi
mkdir -p trainlogs

export TF_CPP_MIN_LOG_LEVEL=3
export HF_HUB_OFFLINE=1
export HF_HOME=~/.cache/huggingface
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export WANDB_MODE=offline
# 强制使用空闲的 GPU 5,6（绕过 SLURM 分配到被占用的 GPU 0-4）
export CUDA_VISIBLE_DEVICES=5,6

uv run scripts/train.py \
    pi05_libero_10_from_pi05libero_10k \
    --exp-name ft_from_pi05libero_10k \
    --overwrite \
    --num-train-steps 30000

echo "Done: $(date)"
