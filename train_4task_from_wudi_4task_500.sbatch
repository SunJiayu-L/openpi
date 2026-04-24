#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:2
#SBATCH --nodelist=gnho031
#SBATCH -J ft_4task_wudi_500
#SBATCH -o trainlogs/ft_4task_from_wudi_4task_500_%j.log
#SBATCH -e trainlogs/ft_4task_from_wudi_4task_500_%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G

# Fine-tune from wudi_mllm/4task_from10k_iter500 on all 4 suites, 10k steps

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
    pi05_libero_4task_from_wudi_4task_500 \
    --exp-name ft_from_wudi_4task_500 \
    --overwrite \
    --fsdp-devices 2

echo "Done: $(date)"
