#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:2
#SBATCH --exclude=gnho009
#SBATCH --mem=160G
#SBATCH --cpus-per-task=8
#SBATCH -J wudi_test_merge
#SBATCH -o trainlogs/wudi_test_merge_%j.log
#SBATCH -e trainlogs/wudi_test_merge_%j.err

# Test merge: pi05_base + pi05_libero 5k, scaling=1, scaling2=1

BASE=/storage/yukaichengLab/lishiwen/jiayusun/openpi_pt/pi05_model/pi05_base/params
FT=checkpoints/pi05_libero/my_experiment/5000/params
OUTPUT=checkpoints/merged/wudi_test_pi05base_5k_iter5
ASSETS_SRC=checkpoints/pi05_libero/my_experiment/5000/assets


echo "Job ID  : ${SLURM_JOB_ID}"
echo "Node    : ${SLURM_NODELIST}"
echo "Base    : ${BASE}"
echo "FT      : ${FT}"
echo "Output  : ${OUTPUT}"
echo "Start   : $(date)"

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi
mkdir -p trainlogs

export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.01
export WANDB_MODE=disabled

echo ""
echo "=== 开始 WUDI 融合 ==="

uv run python scripts/wudi_merge.py \
    --base   "${BASE}" \
    --ft     "${FT}" \
    --output "${OUTPUT}" \
    --scope  llm_only \
    --iter   5 \
    --scaling  1.0 \
    --scaling2 1.0 \
    --device   cuda

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
