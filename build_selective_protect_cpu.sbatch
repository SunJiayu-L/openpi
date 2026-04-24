#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH -J build_protect
#SBATCH -o trainlogs/build_protect.%j.out
#SBATCH -e trainlogs/build_protect.%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

set -euo pipefail
cd /storage/yukaichengLab/lishiwen/jiayusun/openpi
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python scripts/gen_selective_merge_from_mean4.py \
  --scope llm_only \
  --output-root checkpoints/ablation_selective_protect

