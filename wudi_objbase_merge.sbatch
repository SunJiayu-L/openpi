#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:2
#SBATCH --mem=160G
#SBATCH --cpus-per-task=8
#SBATCH -J wudi_objbase
#SBATCH -o trainlogs/wudi_objbase_%j.log
#SBATCH -e trainlogs/wudi_objbase_%j.err
#SBATCH --exclude=gnho009

# Base = pi05_libero_object (30k)
# ft   = spatial / goal / libero_10 (30k)
# scope = llm_only, scaling=0.1, iter=300

STEP=29999
SCALING=0.1
SCALING2=1.0
OUTPUT_NAME=wudi_objbase_30k_s01

BASE=checkpoints/pi05_libero_object/my_experiment/${STEP}/params
OUTPUT=checkpoints/merged/${OUTPUT_NAME}
ASSETS_SRC=checkpoints/pi05_libero_goal/my_experiment/${STEP}/assets

echo "Job ID  : ${SLURM_JOB_ID}"
echo "Node    : ${SLURM_NODELIST}"
echo "Base    : ${BASE}"
echo "ft      : spatial / goal / libero_10 (step=${STEP})"
echo "Scaling : ${SCALING}  (scaling2=${SCALING2})"
echo "Output  : ${OUTPUT}"
echo "Start   : $(date)"

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi
mkdir -p trainlogs

export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.01
export WANDB_MODE=disabled

echo ""
echo "=== 开始 WUDI 融合 (base=pi05_libero_object) ==="

uv run python scripts/wudi_merge.py \
    --base   "${BASE}" \
    --ft \
        checkpoints/pi05_libero_spatial/my_experiment/${STEP}/params \
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
