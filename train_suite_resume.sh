#!/bin/bash
# Resume training from latest checkpoint with fewer GPUs.
# Usage: sbatch --nodelist=<node> --export=CONFIG=<config_name> train_suite_resume.sh
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:8
#SBATCH -J suite_resume
#SBATCH -o trainlogs/resume_%x_%j.log
#SBATCH -e trainlogs/resume_%x_%j.err
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G

set -eo pipefail

if [ -z "$CONFIG" ]; then
    echo "ERROR: CONFIG not set. Use --export=CONFIG=<config_name>"
    exit 1
fi

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Config: $CONFIG (RESUME)"
echo "Start time: $(date)"

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi

export HF_HOME=~/.cache/huggingface
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled

# Norm stats already computed, skip recomputation.
# Resume training from latest checkpoint.
echo "Resuming training for $CONFIG..."
uv run scripts/train.py $CONFIG \
    --exp-name=my_experiment \
    --fsdp-devices 8 \
    --resume

echo "Config: $CONFIG"
echo "End time: $(date)"
