#!/bin/bash
# Single-FT WUDI merge:
#   base: pi05_libero_RETRAIN_base@10000
#   ft  : pi05_sgo_lm90_mix50_task9@29999
#   cfg : scaling=1.0, scaling2=1.0, wandb offline logging

#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=gnho031,gnho034
#SBATCH --exclude=gnho009
#SBATCH --mem=96G
#SBATCH --cpus-per-task=16
#SBATCH -J wudi_1ft_t9_s1
#SBATCH -o /storage/yukaichengLab/lishiwen/jiayusun/openpi/trainlogs/wudi_1ft_t9_s1_%j.log
#SBATCH -e /storage/yukaichengLab/lishiwen/jiayusun/openpi/trainlogs/wudi_1ft_t9_s1_%j.err

set -euo pipefail

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi
mkdir -p trainlogs

BASE="/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero_RETRAIN_base/pi05_libero_RETRAIN_base/10000/params"
FT="/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_sgo_lm90_mix50_task9/my_experiment/29999/params"
FT_ASSETS="/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_sgo_lm90_mix50_task9/my_experiment/29999/assets/pi05_sgo_lm90_mix50_task9/norm_stats.json"
OUT="/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/merged/wudi_singleft_mix50_task9_s1_i300"

echo "[WUDI-1FT] Job: ${SLURM_JOB_ID} Node: ${SLURMD_NODENAME}"
echo "[WUDI-1FT] BASE: ${BASE}"
echo "[WUDI-1FT] FT:   ${FT}"
echo "[WUDI-1FT] OUT:  ${OUT}"
echo "[WUDI-1FT] Start: $(date)"

if [ ! -d "${BASE}" ]; then
  echo "[WUDI-1FT] ERROR: base not found: ${BASE}"
  exit 1
fi
if [ ! -d "${FT}" ]; then
  echo "[WUDI-1FT] ERROR: ft not found: ${FT}"
  exit 1
fi
if [ ! -f "${FT_ASSETS}" ]; then
  echo "[WUDI-1FT] ERROR: norm stats not found: ${FT_ASSETS}"
  exit 1
fi

export WANDB_MODE=offline
export WANDB_DIR="/storage/yukaichengLab/lishiwen/jiayusun/openpi/wandb"
echo "[WUDI-1FT] WANDB_MODE=${WANDB_MODE} WANDB_DIR=${WANDB_DIR}"

rm -rf "${OUT}"
mkdir -p "${OUT}"

uv run python scripts/wudi_merge.py \
  --base   "${BASE}" \
  --ft     "${FT}" \
  --output "${OUT}" \
  --scope  llm_only \
  --iter   300 \
  --scaling 1.0 \
  --scaling2 1.0 \
  --curve-every 1 \
  --device cuda

mkdir -p "${OUT}/assets/pi05_sgo_lm90_mix50_task9"
cp -f "${FT_ASSETS}" "${OUT}/assets/pi05_sgo_lm90_mix50_task9/norm_stats.json"

echo "[WUDI-1FT] End: $(date)"
