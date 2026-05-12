#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:2
#SBATCH --nodelist=gnho031
#SBATCH -J avg_t89_300_x07
#SBATCH -o trainlogs/avg_l10t89_step300_x07_base15k_x03_%j.log
#SBATCH -e trainlogs/avg_l10t89_step300_x07_base15k_x03_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G

# Simple weighted average (no WUDI optimization):
#   merged = 0.7 × l10t89_step300 + 0.3 × pi05_libero@15k
#
# 实现方式: --iter 0 + --init-weights 0.7
#   merged = scaling2*base + scaling * 0.7*(ft - base)
#          = 1.0*base15k + 1.0*0.7*(ckpt300 - base15k)
#          = 0.3*base15k + 0.7*ckpt300

set -eo pipefail
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi
mkdir -p trainlogs checkpoints/avg_merge/l10t89_step300_x07_base15k_x03 wandb_offline

BASE=/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints
export WANDB_MODE=offline
export WANDB_DIR=/storage/yukaichengLab/lishiwen/jiayusun/openpi/wandb_offline
export TF_CPP_MIN_LOG_LEVEL=3
export XLA_PYTHON_CLIENT_PREALLOCATE=false

.venv/bin/python -u scripts/wudi_merge.py \
    --base   $BASE/pi05_libero/my_experiment/15000 \
    --ft     $BASE/pi05_libero_10_t89_from_l10mix50_2k/ft_l10_t89_from_l10mix50_2k/300 \
    --output $BASE/avg_merge/l10t89_step300_x07_base15k_x03 \
    --init-weights 0.7 \
    --scope  llm_only \
    --iter   0 \
    --scaling  1.0 \
    --scaling2 1.0 \
    --device cuda

# 复制 assets（norm stats），评测时需要
cp -r $BASE/pi05_libero_10_t89_from_l10mix50_2k/ft_l10_t89_from_l10mix50_2k/300/assets \
      $BASE/avg_merge/l10t89_step300_x07_base15k_x03/assets

echo "Done: $(date)"
