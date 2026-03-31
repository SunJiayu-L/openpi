#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:0
#SBATCH --exclude=gnho009
#SBATCH -J dump_libero_val
#SBATCH -o trainlogs/dump_libero_val_%j.log
#SBATCH -e trainlogs/dump_libero_val_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -eo pipefail

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi
mkdir -p trainlogs

export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""
export TF_CPP_MIN_LOG_LEVEL=3

.venv/bin/python -u scripts/dump_data.py \
    --dataset pi05_libero \
    --output libero_val.pkl \
    --num_batches 50 \
    --batch_size 32

echo "Done: $(date)"
