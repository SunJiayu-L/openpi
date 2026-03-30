#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:4
#SBATCH --exclude=gnho009
#SBATCH -J pi0_lora_train
#SBATCH -o trainlogs/lora_train_%x.%j.log
#SBATCH -e trainlogs/lora_train_%x.%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

set -eo pipefail

CONFIG="pi0_libero_low_mem_finetune"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Config: $CONFIG"
echo "Start time: $(date)"

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi

export HF_HOME=~/.cache/huggingface
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled

echo "Step 1: Computing norm stats..."
uv run scripts/compute_norm_stats.py --config-name $CONFIG

echo "Step 2: Training LoRA..."
uv run scripts/train.py $CONFIG \
    --exp-name=my_experiment \
    --fsdp-devices 4

echo "Config: $CONFIG"
echo "End time: $(date)"
