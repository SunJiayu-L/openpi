#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:8
#SBATCH --nodelist=gnho034
#SBATCH -J pi05_libero_no10_train
#SBATCH -o trainlogs/pi05_libero_no10_train.%j.log
#SBATCH -e trainlogs/pi05_libero_no10_train.%j.err
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
# Step 1: Compute norm stats for the split subset (36 train tasks)
# ============================================================
echo "Computing normalization statistics for pi05_libero_no10..."
uv run scripts/compute_norm_stats.py --config-name pi05_libero_no10

# ============================================================
# Step 2: Train pi0.5 on 36 tasks (first task of each suite held out)
# ============================================================
echo "Starting training..."
uv run scripts/train.py pi05_libero_no10 \
    --exp-name=split_experiment \
    --fsdp-devices 8

echo "End time: $(date)"
