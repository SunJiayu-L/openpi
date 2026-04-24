#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:4
#SBATCH --nodelist=gnho009
#SBATCH -J wudi_4task_mean_300
#SBATCH -o trainlogs/wudi_4task_mean_iter300_%j.log
#SBATCH -e trainlogs/wudi_4task_mean_iter300_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G

# WUDI merge (init=mean, iter=300): 4task

set -eo pipefail
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi
mkdir -p trainlogs checkpoints/wudi_mllm/4task_mean_iter300

BASE=/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints
export WANDB_MODE=offline
export WANDB_DIR=/storage/yukaichengLab/lishiwen/jiayusun/openpi/wandb_offline
export TF_CPP_MIN_LOG_LEVEL=3
export XLA_PYTHON_CLIENT_PREALLOCATE=false

.venv/bin/python -u scripts/wudi_merge.py \
    --base   $BASE/pi05_libero/my_experiment/10000 \
    --ft     $BASE/from10k/pi05_libero_10_from_pi05libero_10k/ft_from_pi05libero_10k/20000 \
             $BASE/from10k/pi05_libero_goal_from_pi05libero_10k/ft_from_pi05libero_10k/20000 \
             $BASE/from10k/pi05_libero_object_from_pi05libero_10k/ft_from_pi05libero_10k/20000 \
             $BASE/from10k/pi05_libero_spatial_from_pi05libero_10k/ft_from_pi05libero_10k/20000 \
    --output $BASE/wudi_mllm/4task_mean_iter300 \
    --scope  llm_only \
    --iter   300 \
    --scaling 1.0 \
    --device cuda

# symlink assets
ln -sfn $BASE/wudi_mllm/10_spatial_from10k_iter5k/assets \
        $BASE/wudi_mllm/4task_mean_iter300/assets
echo "Assets symlinked."
echo "Done: $(date)"
