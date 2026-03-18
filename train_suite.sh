#!/bin/bash
# Usage: sbatch --nodelist=<node> --export=CONFIG=<config_name> train_suite.sh
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:8
#SBATCH -J suite_train
#SBATCH -o trainlogs/train_%x_%j.log
#SBATCH -e trainlogs/train_%x_%j.err
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G

set -eo pipefail

if [ -z "$CONFIG" ]; then
    echo "ERROR: CONFIG not set. Use --export=CONFIG=<config_name>"
    exit 1
fi

# Update job name for clarity in squeue
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Config: $CONFIG"
echo "Start time: $(date)"

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi

export HF_HOME=~/.cache/huggingface
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled

# Step 1: Norm stats
echo "Computing normalization statistics for $CONFIG..."
uv run scripts/compute_norm_stats.py --config-name $CONFIG

# Step 2: Train
echo "Starting training for $CONFIG..."
uv run scripts/train.py $CONFIG \
    --exp-name=my_experiment \
    --fsdp-devices 8

echo "Config: $CONFIG"
echo "End time: $(date)"
