#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH -J wudi_merge
#SBATCH -o trainlogs/wudi_%x.%j.log
#SBATCH -e trainlogs/wudi_%x.%j.err

# Usage:
#   sbatch wudi_merge.sbatch expert1_only  ft1/params ft2/params  output_name
#   sbatch wudi_merge.sbatch both_experts  ft1/params ft2/params  output_name
#
# For both_experts increase memory:
#   #SBATCH --mem=128G

SCOPE=${1:-expert1_only}
FT1=${2:-checkpoints/pi05_libero/my_experiment/29999/params}
FT2=${3:-checkpoints/pi05_libero_goal/my_experiment/29999/params}
OUTPUT_NAME=${4:-wudi_merged}

BASE=/storage/yukaichengLab/lishiwen/jiayusun/openpi_pt/pi05_model/pi05_base/params
OUTPUT=checkpoints/merged/${OUTPUT_NAME}

echo "Job ID  : ${SLURM_JOB_ID}"
echo "Node    : ${SLURM_NODELIST}"
echo "Scope   : ${SCOPE}"
echo "FT1     : ${FT1}"
echo "FT2     : ${FT2}"
echo "Output  : ${OUTPUT}"
echo "Start   : $(date)"

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi

mkdir -p trainlogs

# Keep JAX off the GPU so PyTorch (WUDI optimizer) can use it fully
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.01

echo ""
echo "=== 开始 WUDI 融合 ==="

uv run python scripts/wudi_merge.py \
    --base   "${BASE}" \
    --ft     "${FT1}" "${FT2}" \
    --output "${OUTPUT}" \
    --scope  "${SCOPE}" \
    --iter   300 \
    --scaling 1.0 \
    --device  cuda

echo ""
echo "=== 融合完成: $(date) ==="
echo "输出路径: ${OUTPUT}"

if [ -d "${OUTPUT}" ] && [ "$(ls -A ${OUTPUT})" ]; then
    echo "输出目录非空，checkpoint 已生成。"
else
    echo "警告：输出目录为空或不存在！"
    exit 1
fi
