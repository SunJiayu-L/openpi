#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:2
#SBATCH --nodelist=gnho018
#SBATCH -J train_10_spatial_wudi_5k
#SBATCH -o trainlogs/train_10_spatial_wudi_5k_%j.log
#SBATCH -e trainlogs/train_10_spatial_wudi_5k_%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G

# Fine-tune pi0.5 from WUDI-MLLM 5k merged checkpoint on libero_10 + libero_spatial only.
# Norm stats from pi05_libero/my_experiment/10000. batch_size=32, 2 GPUs.

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
    pi05_libero_10_spatial_from_wudi_mllm_5k \
    --exp-name ft_from_wudi_mllm_5k \
    --resume \
    --fsdp-devices 2

echo "Done: $(date)"
echo ""
echo "To sync wandb: source proxy_on.sh && .venv/bin/wandb sync wandb_offline/wandb/offline-run-*"
