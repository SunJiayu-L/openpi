#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:2
#SBATCH --nodelist=gnho031
#SBATCH -J ft_l10_1k
#SBATCH -o trainlogs/train_libero10_from_4task_3sog500_5k_%j.log
#SBATCH -e trainlogs/train_libero10_from_4task_3sog500_5k_%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G

# Fine-tune from ft_from_3task_sog_iter500/4999 on libero_10 only, 1k steps, 2-GPU FSDP, batch_size=32

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
    pi05_libero_10_from_4task_3sog500_5k \
    --exp-name ft_libero10_from_4task_3sog500_5k \
    --overwrite \
    --fsdp-devices 2

echo "Done: $(date)"
