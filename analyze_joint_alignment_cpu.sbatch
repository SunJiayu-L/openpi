#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH -J joint_align
#SBATCH -o trainlogs/joint_align.%j.out
#SBATCH -e trainlogs/joint_align.%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

set -euo pipefail
cd /storage/yukaichengLab/lishiwen/jiayusun/openpi
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python scripts/analyze_joint_target_alignment.py \
  --scope llm_only \
  --output-dir docs/analysis/joint_target_alignment_2026-04-20

python scripts/analyze_joint_repair_map.py \
  --scope llm_only \
  --output-dir docs/analysis/joint_target_alignment_2026-04-20 \
  --drop-csv docs/analysis/from10k_llm_only_ffn_layerwise_retain_rate/layerwise_drop_summary_r10.csv

