#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gnho009
#SBATCH -J compute_norm_stats
#SBATCH -o trainlogs/compute_norm_stats.%j.log
#SBATCH -e trainlogs/compute_norm_stats.%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi

# Ensure HF datasets cache goes to ~/.cache/huggingface
export HF_HOME=~/.cache/huggingface
# Offline mode: compute node has no internet access
export HF_HUB_OFFLINE=1

uv run scripts/compute_norm_stats.py --config-name pi05_libero

echo "End time: $(date)"