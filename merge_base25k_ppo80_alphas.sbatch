#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:0
#SBATCH --nodelist=gnho031
#SBATCH -J merge_base_ppo80
#SBATCH -o /storage/yukaichengLab/lishiwen/jiayusun/openpi/trainlogs/merge_base_ppo80.%j.log
#SBATCH -e /storage/yukaichengLab/lishiwen/jiayusun/openpi/trainlogs/merge_base_ppo80.%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Task Arithmetic 融合 base_25k 与 PPO step80：
#   theta* = theta_base + alpha * (theta_PPO - theta_base)
# 扫 alpha = 0.3 / 0.5 / 0.7 / 1.0；alpha=1.0 等价于 PPO 本身（基线对照）。

set -eo pipefail
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi

BASE="checkpoints/pt_converted/base_25k"
FT="checkpoints/pt_rl/ppo_step80"

for ALPHA in 0.3 0.5 0.7 1.0; do
    TAG="base25k_ppo80_a${ALPHA//./}"
    OUT="checkpoints/merged/${TAG}"
    echo ""
    echo "=== alpha=${ALPHA} → ${OUT} | $(date) ==="
    uv run python scripts/task_arithmetic_pt.py \
        --base   "$BASE" \
        --ft     "$FT" \
        --output "$OUT" \
        --alpha  "$ALPHA" \
        --cast-dtype bfloat16
done

echo ""
echo "All merges done: $(date)"
ls -lh checkpoints/merged/
