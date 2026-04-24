#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH -J ffn_layer_rr
#SBATCH -o trainlogs/ffn_layer_rr.%j.out
#SBATCH -e trainlogs/ffn_layer_rr.%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

set -euo pipefail
cd /storage/yukaichengLab/lishiwen/jiayusun/openpi
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python scripts/analyze_task_vector_layerwise_ffn_retain_rate.py \
  --scope llm_only \
  --output-dir docs/analysis/from10k_llm_only_ffn_layerwise_retain_rate \
  --topk 3

