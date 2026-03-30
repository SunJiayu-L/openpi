#!/bin/bash
# Usage: sbatch [--nodelist=<node>] --export=CONFIG=<config_name> train_suite_4gpu.sh
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:4
#SBATCH -J suite_train_4g
#SBATCH -o trainlogs/train_%x_%j.log
#SBATCH -e trainlogs/train_%x_%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G

set -eo pipefail

if [ -z "$CONFIG" ]; then
    echo "ERROR: CONFIG not set. Use --export=CONFIG=<config_name>"
    exit 1
fi

EXP_NAME="${EXP_NAME:-my_experiment}"
OVERWRITE="${OVERWRITE:-0}"
OVERWRITE_FLAG=""
if [ "$OVERWRITE" = "1" ]; then
    OVERWRITE_FLAG="--overwrite"
fi

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Config: $CONFIG"
echo "Exp name: $EXP_NAME"
echo "Overwrite: $OVERWRITE"
echo "Start time: $(date)"

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi

export HF_HOME=~/.cache/huggingface
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled

echo "Computing normalization statistics for $CONFIG..."
uv run scripts/compute_norm_stats.py --config-name "$CONFIG"

echo "Starting training for $CONFIG on 4 GPUs..."
uv run scripts/train.py "$CONFIG" \
    --exp-name="$EXP_NAME" \
    $OVERWRITE_FLAG \
    --fsdp-devices 4

echo "Config: $CONFIG"
echo "End time: $(date)"
