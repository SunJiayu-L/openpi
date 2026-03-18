#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:8
#SBATCH --nodelist=gnho034
#SBATCH -J pi0_libero_train
#SBATCH -o trainlogs/pi0_libero_train.%j.log
#SBATCH -e trainlogs/pi0_libero_train.%j.err
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi

# Offline mode
export HF_HOME=~/.cache/huggingface
export HF_HUB_OFFLINE=1

# Disable wandb (no external network on GPU nodes)
export WANDB_MODE=disabled

# ============================================================
# Step 1: Compute norm stats for pi0_libero
# ============================================================
echo "Computing normalization statistics for pi0_libero..."
uv run scripts/compute_norm_stats.py --config-name pi0_libero

# ============================================================
# Step 2: Train pi0 on all 40 LIBERO tasks (full fine-tuning)
# ============================================================
echo "Starting training..."
uv run scripts/train.py pi0_libero \
    --exp-name=my_experiment \
    --overwrite \
    --fsdp-devices 8

echo "End time: $(date)"
