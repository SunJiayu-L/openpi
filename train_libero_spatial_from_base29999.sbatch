#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:4
#SBATCH --nodelist=gnho009
#SBATCH -J train_lspa_base29999
#SBATCH -o trainlogs/train_lspa_base29999_%j.log
#SBATCH -e trainlogs/train_lspa_base29999_%j.err
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

uv run scripts/train.py \
    pi05_libero_spatial_from_base29999 \
    --exp-name ft_from_base29999 \
    --overwrite \
    --num-train-steps 30000

echo "Done: $(date)"
