#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH -J retain_cpu
#SBATCH -o trainlogs/retain_cpu.%j.out
#SBATCH -e trainlogs/retain_cpu.%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

set -euo pipefail
cd /storage/yukaichengLab/lishiwen/jiayusun/openpi
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python scripts/analyze_task_vector_retain_rate.py \
  --scope llm_only \
  --output-dir docs/analysis/from10k_llm_only_retain_rate
