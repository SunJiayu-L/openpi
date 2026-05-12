#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:2
#SBATCH --nodelist=gnho031
#SBATCH -J ft_l10t89
#SBATCH -o trainlogs/ft_l10_t89_from_l10mix50_2k_%j.log
#SBATCH -e trainlogs/ft_l10_t89_from_l10mix50_2k_%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G

# FT from l10_mix50/1999 (best 96.6% avg)
# Dataset: libero_10 task 8 (moka pots, 29 eps) + task 9 (mug microwave, 34 eps) = 63 eps
# Steps: 500, batch_size=32, save at step 300 and 500

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
    pi05_libero_10_t89_from_l10mix50_2k \
    --exp-name ft_l10_t89_from_l10mix50_2k \
    --overwrite \
    --fsdp-devices 2

echo "Done: $(date)"
