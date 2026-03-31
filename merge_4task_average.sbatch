#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:1
#SBATCH --exclude=gnho009
#SBATCH -J merge_4task
#SBATCH -o trainlogs/merge_4task_%j.log
#SBATCH -e trainlogs/merge_4task_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -eo pipefail

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi
mkdir -p trainlogs checkpoints/pi05_libero_4task_merge

BASE=/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints

export TF_CPP_MIN_LOG_LEVEL=3
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

.venv/bin/python -u scripts/arithmetic.py \
    --config pi05_libero \
    --data-path libero_val.pkl \
    --checkpoints \
        $BASE/pi05_libero_spatial/my_experiment/29999 \
        $BASE/pi05_libero_object/my_experiment/29999 \
        $BASE/pi05_libero_goal/my_experiment/29999 \
        $BASE/pi05_libero_10/my_experiment/29999 \
    --output /storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero_4task_merge \
    --optimize_method average \
    --gpu_ids 0

echo "Done: $(date)"
