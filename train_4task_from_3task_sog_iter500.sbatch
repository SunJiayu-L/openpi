#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:2
#SBATCH --nodelist=gnho031
#SBATCH -J ft_3sog500_5k
#SBATCH -o trainlogs/ft_4task_from_3task_sog_iter500_%j.log
#SBATCH -e trainlogs/ft_4task_from_3task_sog_iter500_%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G

# Fine-tune from wudi_mllm/3task_sog_mean_iter500 on all 4 suites, 5k steps, 2-GPU FSDP, batch_size=32

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
    pi05_libero_4task_from_3task_sog_iter500 \
    --exp-name ft_from_3task_sog_iter500 \
    --overwrite \
    --fsdp-devices 2

echo "Done: $(date)"
