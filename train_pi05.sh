#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:8
#SBATCH --nodelist=gnho034
#SBATCH -J pi05_libero_train
#SBATCH -o trainlogs/pi05_libero_train.%j.log
#SBATCH -e trainlogs/pi05_libero_train.%j.err
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi

# Offline mode: compute node has no internet access
export HF_HOME=~/.cache/huggingface
export HF_HUB_OFFLINE=1

# Wandb offline mode (compute node has no internet), sync later
export WANDB_MODE=offline

# JAX memory allocation
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

uv run scripts/train.py pi05_libero --exp-name=my_experiment --fsdp_devices=8 --overwrite

echo "End time: $(date)"
