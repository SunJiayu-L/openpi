#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:4
#SBATCH --nodelist=gnho031
#SBATCH -J jiayu_wudi_mllm_10_spatial
#SBATCH -o trainlogs/wudi_mllm_10_spatial_%j.log
#SBATCH -e trainlogs/wudi_mllm_10_spatial_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G

# WUDI merge (MLLMerging exclusion rules): libero_10 + libero_spatial from 10k base
# Merges 10 Attention+FFN params only; norm/bias/vision/embedder kept at base.
# iter=1000, scaling=1.0, scope=llm_only.

set -eo pipefail

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi
mkdir -p trainlogs checkpoints/wudi_mllm/10_spatial_from10k

BASE=/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints

# Wandb 离线模式：计算节点无外网，结束后在登录节点上传
export WANDB_MODE=offline
export WANDB_DIR=/storage/yukaichengLab/lishiwen/jiayusun/openpi/wandb_offline
mkdir -p "$WANDB_DIR"

export TF_CPP_MIN_LOG_LEVEL=3
export XLA_PYTHON_CLIENT_PREALLOCATE=false

.venv/bin/python -u scripts/wudi_merge.py \
    --base   $BASE/pi05_libero/my_experiment/10000 \
    --ft     $BASE/from10k/pi05_libero_10_from_pi05libero_10k/ft_from_pi05libero_10k/20000 \
             $BASE/from10k/pi05_libero_spatial_from_pi05libero_10k/ft_from_pi05libero_10k/20000 \
    --output $BASE/wudi_mllm/10_spatial_from10k_iter5k \
    --scope  llm_only \
    --iter   5000 \
    --scaling 1.0 \
    --device cuda

echo "Done: $(date)"
echo ""
echo "To sync wandb logs, run on login node:"
echo "  cd /storage/yukaichengLab/lishiwen/jiayusun/openpi"
echo "  source proxy_on.sh && .venv/bin/wandb sync wandb_offline/wandb/offline-run-*"
