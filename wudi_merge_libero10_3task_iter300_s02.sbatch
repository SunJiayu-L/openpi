#!/bin/bash
# Merge 3 task-specific checkpoints with WUDI.
# base: pi05_libero_RETRAIN_base step 10000
# fts : task0_r2@29999, task6@29999, task9_r2@29000
# params: iter=1000, scaling=1.0, offline wandb + unified loss curve

#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH -J wudi_l10_3t_i1k_curve
#SBATCH -o /storage/yukaichengLab/lishiwen/jiayusun/openpi/trainlogs/wudi_l10_3t_i1k_curve_%j.log
#SBATCH -e /storage/yukaichengLab/lishiwen/jiayusun/openpi/trainlogs/wudi_l10_3t_i1k_curve_%j.err

set -euo pipefail

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi
mkdir -p trainlogs

BASE="/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero_RETRAIN_base/pi05_libero_RETRAIN_base/10000/params"
FT0="/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero10_task0_retrain/my_experiment_r2/29999/params"
FT6="/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero10_task6_retrain/my_experiment/29999/params"
FT9="/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero10_task9_retrain/my_experiment_r2/29000/params"
OUT="checkpoints/merged/wudi_libero10_3task_iter1000_s1_curve"

echo "[WUDI-3TASK] Job: ${SLURM_JOB_ID} Node: ${SLURMD_NODENAME}"
echo "[WUDI-3TASK] BASE: ${BASE}"
echo "[WUDI-3TASK] FT: ${FT0} ${FT6} ${FT9}"
echo "[WUDI-3TASK] OUT: ${OUT}"
echo "[WUDI-3TASK] Start: $(date)"

export WANDB_MODE=offline
export WANDB_DIR="/storage/yukaichengLab/lishiwen/jiayusun/openpi/wandb"
echo "[WUDI-3TASK] WANDB_MODE=${WANDB_MODE} WANDB_DIR=${WANDB_DIR}"

rm -rf "${OUT}"
mkdir -p "${OUT}"

uv run python scripts/wudi_merge.py \
  --base   "${BASE}" \
  --ft     "${FT0}" "${FT6}" "${FT9}" \
  --output "${OUT}" \
  --scope  llm_only \
  --iter   1000 \
  --scaling 1.0 \
  --scaling2 1.0 \
  --curve-every 1 \
  --device cuda

# keep task0 assets layout for eval config compatibility
cp -r "/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero10_task0_retrain/my_experiment_r2/29999/assets" "${OUT}/assets"

echo "[WUDI-3TASK] End: $(date)"
