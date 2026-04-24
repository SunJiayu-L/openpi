#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:2
#SBATCH --nodelist=gnho031
#SBATCH -J jiayu_spatial_031
#SBATCH -o trainlogs/train_spatial_031_%j.log
#SBATCH -e trainlogs/train_spatial_031_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

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

.venv/bin/python -u scripts/train.py \
    pi05_libero_spatial_from_pi05libero_10k \
    --exp-name ft_from_pi05libero_10k \
    --overwrite \
    --num-train-steps 30000

echo "Done: $(date)"
