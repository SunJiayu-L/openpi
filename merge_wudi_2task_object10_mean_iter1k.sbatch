#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:4
#SBATCH --nodelist=gnho009
#SBATCH -J wudi_2t_obj10_1k
#SBATCH -o trainlogs/wudi_2task_object10_mean_iter1k_%j.log
#SBATCH -e trainlogs/wudi_2task_object10_mean_iter1k_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G

set -eo pipefail
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi
mkdir -p trainlogs checkpoints/wudi_mllm/2task_object10_mean_iter1k wandb_offline

BASE=/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints
export WANDB_MODE=offline
export WANDB_DIR=/storage/yukaichengLab/lishiwen/jiayusun/openpi/wandb_offline
export TF_CPP_MIN_LOG_LEVEL=3
export XLA_PYTHON_CLIENT_PREALLOCATE=false

.venv/bin/python -u scripts/wudi_merge.py \
    --base   $BASE/pi05_libero/my_experiment/10000 \
    --ft     $BASE/from10k/pi05_libero_object_from_pi05libero_10k/ft_from_pi05libero_10k/20000 \
             $BASE/from10k/pi05_libero_10_from_pi05libero_10k/ft_from_pi05libero_10k/20000 \
    --output $BASE/wudi_mllm/2task_object10_mean_iter1k \
    --scope  llm_only \
    --iter   1000 \
    --scaling 1.0 \
    --scaling2 1.0 \
    --device cuda

ln -sfn $BASE/wudi_mllm/2task_spatial10_iter5k/assets \
        $BASE/wudi_mllm/2task_object10_mean_iter1k/assets
echo "Done: $(date)"
