#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:4
#SBATCH --nodelist=gnho034
#SBATCH -J train_libero_merged
#SBATCH -o trainlogs/train_libero_merged_%j.log
#SBATCH -e trainlogs/train_libero_merged_%j.err
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
    pi05_libero_from_merged_4task_gd \
    --exp-name retrain_from_merged_30k \
    --overwrite \
    --num-train-steps 30000

echo "Done: $(date)"
