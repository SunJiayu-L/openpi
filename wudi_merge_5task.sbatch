#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:2
#SBATCH --mem=160G
#SBATCH --cpus-per-task=8
#SBATCH -J wudi5_merge
#SBATCH -o trainlogs/wudi5_%x_%j.log
#SBATCH -e trainlogs/wudi5_%x_%j.err

# Usage:
#   sbatch wudi_merge_5task.sbatch <STEP> [SCALING]
# STEP: 5000 | 15000 | 29999   (29999 counts as "30k")
# SCALING: float, default 1.0
#
# Merges 5 fine-tuned models (libero / spatial / object / goal / 10)
# using WUDI SVD de-centering, expert1_only scope.

STEP=${1:-29999}
SCALING=${2:-1.0}
SCALING2=${3:-1.0}

# Map step → friendly name for output dir
case "${STEP}" in
  5000)   TAG="5k"  ;;
  15000)  TAG="15k" ;;
  29999)  TAG="30k" ;;
  *)      TAG="${STEP}" ;;
esac

# Output name
STAG=$(echo "${SCALING}" | sed 's/\.//g')
STAG2=$(echo "${SCALING2}" | sed 's/\.//g')
if [ "${SCALING2}" = "1.0" ]; then
  OUTPUT_NAME=wudi_4task_${TAG}_s${STAG}
else
  OUTPUT_NAME=wudi_4task_${TAG}_s${STAG}_b${STAG2}
fi

# Base = pi05_base (预训练模型，不含任何 LIBERO 知识)
BASE=/storage/yukaichengLab/lishiwen/jiayusun/openpi_pt/pi05_model/pi05_base/params
OUTPUT=checkpoints/merged/${OUTPUT_NAME}

# Assets source (norm_stats etc.)
ASSETS_SRC=checkpoints/pi05_libero_goal/my_experiment/${STEP}/assets

echo "Job ID  : ${SLURM_JOB_ID}"
echo "Node    : ${SLURM_NODELIST}"
echo "Step    : ${STEP}  (tag: ${TAG})"
echo "Scaling : ${SCALING}  (scaling2/base=${SCALING2})"
echo "Output  : ${OUTPUT}"
echo "Start   : $(date)"

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi
mkdir -p trainlogs

# Keep JAX off the GPU so PyTorch (WUDI optimizer) can use it fully
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.01

# wandb
export WANDB_PROJECT=wudi-merge-pi05

echo ""
echo "=== 开始 WUDI 5-model 融合 (step=${STEP}) ==="

uv run python scripts/wudi_merge.py \
    --base   "${BASE}" \
    --ft \
        checkpoints/pi05_libero_spatial/my_experiment/${STEP}/params \
        checkpoints/pi05_libero_object/my_experiment/${STEP}/params \
        checkpoints/pi05_libero_goal/my_experiment/${STEP}/params \
        checkpoints/pi05_libero_10/my_experiment/${STEP}/params \
    --output  "${OUTPUT}" \
    --scope  llm_only \
    --iter   300 \
    --scaling  ${SCALING} \
    --scaling2 ${SCALING2} \
    --device  cuda

echo ""
echo "=== 融合完成: $(date) ==="

if [ ! -d "${OUTPUT}/params" ]; then
    echo "错误：融合失败，输出目录不存在！"
    exit 1
fi

# Copy assets (norm_stats.json etc.) from source checkpoint
echo "复制 assets from ${ASSETS_SRC}..."
if [ -d "${ASSETS_SRC}" ]; then
    cp -r "${ASSETS_SRC}" "${OUTPUT}/assets"
    echo "assets 复制完成。"
else
    echo "警告：assets 源目录不存在: ${ASSETS_SRC}"
fi

echo ""
echo "输出路径: ${OUTPUT}"
echo "完成。"
