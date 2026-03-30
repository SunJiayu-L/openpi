#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:0
#SBATCH --exclude=gnho009
#SBATCH -J convert_lm90
#SBATCH -o trainlogs/convert_lm90_%j.log
#SBATCH -e trainlogs/convert_lm90_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

set -eo pipefail

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi
mkdir -p trainlogs

# 不使用 GPU，仅做数据格式转换
export CUDA_VISIBLE_DEVICES=""
export TF_CPP_MIN_LOG_LEVEL=3

.venv/bin/python -u examples/libero/convert_libero_data_to_lerobot.py \
    --data_dir /storage/yukaichengLab/lishiwen/jiayusun/embodied_features_and_demos_libero

echo "Done: $(date)"
