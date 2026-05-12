#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:0
#SBATCH --nodelist=gnho031
#SBATCH -J merge_a09
#SBATCH -o /storage/yukaichengLab/lishiwen/jiayusun/openpi/trainlogs/merge_a09.%j.log
#SBATCH -e /storage/yukaichengLab/lishiwen/jiayusun/openpi/trainlogs/merge_a09.%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -eo pipefail
cd /storage/yukaichengLab/lishiwen/jiayusun/openpi
uv run python scripts/task_arithmetic_pt.py \
    --base   checkpoints/pt_converted/base_25k \
    --ft     checkpoints/pt_rl/ppo_step80 \
    --output checkpoints/merged/base25k_ppo80_a09 \
    --alpha  0.9 \
    --cast-dtype bfloat16
ls -lh checkpoints/merged/base25k_ppo80_a09/
