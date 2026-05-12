#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:2
#SBATCH --nodelist=gnho031
#SBATCH -J ft_l10mix50_2k
#SBATCH -o trainlogs/ft_l10_mix50_from_15k_%j.log
#SBATCH -e trainlogs/ft_l10_mix50_from_15k_%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G

# FT from pi05_libero/my_experiment/15000
# Dataset: 50/50 mix libero_10 vs (spatial+object+goal)
# Steps: 2000, batch_size=32, 2-GPU FSDP, save every 1k

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
export WANDB_DIR=/storage/yukaichengLab/lishiwen/jiayusun/openpi/wandb_offline
mkdir -p "$WANDB_DIR"

uv run scripts/train.py \
    pi05_libero_l10_mix50_from_15k \
    --exp-name ft_l10_mix50_from_15k \
    --overwrite \
    --fsdp-devices 2

echo "Done: $(date)"
