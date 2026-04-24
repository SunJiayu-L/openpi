#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:4
#SBATCH --nodelist=gnho009
#SBATCH -J jiayu_wudi_4task_500
#SBATCH -o trainlogs/wudi_4task_from10k_iter500_%j.log
#SBATCH -e trainlogs/wudi_4task_from10k_iter500_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G

# WUDI merge: 4 suites (libero_10 + goal + object + spatial) all from 10k base, 500 iters
# Base: pi05_libero/my_experiment/10000
# FTs:  from10k/pi05_libero_{10,goal,object,spatial}_from_pi05libero_10k/ft_from_pi05libero_10k/20000
# Scope: llm_only (Gemma attn+FFN only), scaling=1.0

set -eo pipefail

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi
mkdir -p trainlogs checkpoints/wudi_mllm/4task_from10k_iter500 wandb_offline

BASE=/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints

export WANDB_MODE=offline
export WANDB_DIR=/storage/yukaichengLab/lishiwen/jiayusun/openpi/wandb_offline
mkdir -p "$WANDB_DIR"

export TF_CPP_MIN_LOG_LEVEL=3
export XLA_PYTHON_CLIENT_PREALLOCATE=false

.venv/bin/python -u scripts/wudi_merge.py \
    --base   $BASE/pi05_libero/my_experiment/10000 \
    --ft     $BASE/from10k/pi05_libero_10_from_pi05libero_10k/ft_from_pi05libero_10k/20000 \
             $BASE/from10k/pi05_libero_goal_from_pi05libero_10k/ft_from_pi05libero_10k/20000 \
             $BASE/from10k/pi05_libero_object_from_pi05libero_10k/ft_from_pi05libero_10k/20000 \
             $BASE/from10k/pi05_libero_spatial_from_pi05libero_10k/ft_from_pi05libero_10k/20000 \
    --output $BASE/wudi_mllm/4task_from10k_iter500 \
    --scope  llm_only \
    --iter   500 \
    --scaling 1.0 \
    --device cuda

echo "Done: $(date)"
