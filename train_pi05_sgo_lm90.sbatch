#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:8
#SBATCH --nodelist=gnho034
#SBATCH -J pi05_sgo_lm90
#SBATCH -o trainlogs/pi05_sgo_lm90.%j.log
#SBATCH -e trainlogs/pi05_sgo_lm90.%j.err
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G

set -eo pipefail

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi
mkdir -p trainlogs

export HF_HOME=~/.cache/huggingface
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled

CONFIG="pi05_libero_sgo_lm90"
EXP_NAME="sgo_lm90_bs64_warmup1k"

# Step 1: 归一化统计（已在 job 32585 中完成，跳过）
# uv run scripts/compute_norm_stats.py --config-name ${CONFIG}

# Step 2: 训练
echo "=== Starting training ==="
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py ${CONFIG} \
    --exp-name=${EXP_NAME} \
    --overwrite \
    --fsdp-devices 8

echo "End time: $(date)"
