#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:4
#SBATCH --nodelist=gnho034
#SBATCH -J wudi_2task_mean_500
#SBATCH -o trainlogs/wudi_2task_mean_iter500_%j.log
#SBATCH -e trainlogs/wudi_2task_mean_iter500_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G

# WUDI merge (init=mean): 2task (libero_10 + libero_spatial), iter=500
# 初始化从 sum 改为 mean

set -eo pipefail
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi
mkdir -p trainlogs checkpoints/wudi_mllm/2task_mean_iter500 wandb_offline

BASE=/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints
export WANDB_MODE=offline
export WANDB_DIR=/storage/yukaichengLab/lishiwen/jiayusun/openpi/wandb_offline
export TF_CPP_MIN_LOG_LEVEL=3
export XLA_PYTHON_CLIENT_PREALLOCATE=false

.venv/bin/python -u scripts/wudi_merge.py \
    --base   $BASE/pi05_libero/my_experiment/10000 \
    --ft     $BASE/from10k/pi05_libero_10_from_pi05libero_10k/ft_from_pi05libero_10k/20000 \
             $BASE/from10k/pi05_libero_spatial_from_pi05libero_10k/ft_from_pi05libero_10k/20000 \
    --output $BASE/wudi_mllm/2task_mean_iter500 \
    --scope  llm_only \
    --iter   500 \
    --scaling 1.0 \
    --device cuda

echo "Done: $(date)"
