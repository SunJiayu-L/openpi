#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --nodelist=gnho031
#SBATCH -J ties_4task_mr08
#SBATCH -o trainlogs/ties_4task_mr08_%j.log
#SBATCH -e trainlogs/ties_4task_mr08_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G

# TIES merge: 4task, mask_rate=0.8, scaling=1.0 (paper defaults)

set -eo pipefail
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi
mkdir -p trainlogs checkpoints/ties_mllm/4task_mr08_s10

BASE=/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints
export WANDB_MODE=offline
export WANDB_DIR=/storage/yukaichengLab/lishiwen/jiayusun/openpi/wandb_offline
export TF_CPP_MIN_LOG_LEVEL=3
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""

.venv/bin/python -u scripts/ties_merge.py \
    --base   $BASE/pi05_libero/my_experiment/10000 \
    --ft     $BASE/from10k/pi05_libero_10_from_pi05libero_10k/ft_from_pi05libero_10k/20000 \
             $BASE/from10k/pi05_libero_goal_from_pi05libero_10k/ft_from_pi05libero_10k/20000 \
             $BASE/from10k/pi05_libero_object_from_pi05libero_10k/ft_from_pi05libero_10k/20000 \
             $BASE/from10k/pi05_libero_spatial_from_pi05libero_10k/ft_from_pi05libero_10k/20000 \
    --output $BASE/ties_mllm/4task_mr08_s10 \
    --scope  llm_only \
    --mask-rate 0.8 \
    --scaling 1.0 \
    --scaling2 1.0 \
    --device cpu

# symlink assets (reuse WUDI's assets, they share the same pi05_libero tokenizer/norm_stats)
ln -sfn $BASE/wudi_mllm/10_spatial_from10k_iter5k/assets \
        $BASE/ties_mllm/4task_mr08_s10/assets
echo "Assets symlinked."
echo "Done: $(date)"
