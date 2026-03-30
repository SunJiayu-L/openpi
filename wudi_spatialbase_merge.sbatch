#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:2
#SBATCH --exclude=gnho009
#SBATCH --mem=160G
#SBATCH --cpus-per-task=8
#SBATCH -J wudi_spatialbase
#SBATCH -o trainlogs/wudi_spatialbase_%j.log
#SBATCH -e trainlogs/wudi_spatialbase_%j.err

BASE=checkpoints/pi05_libero_spatial/my_experiment/29999/params
OUTPUT=checkpoints/merged/wudi_pi05_spatialbase_iter300_s02

echo "Job ID  : ${SLURM_JOB_ID}"
echo "Node    : ${SLURM_NODELIST}"
echo "Base    : ${BASE}"
echo "Output  : ${OUTPUT}"
echo "Start   : $(date)"

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi
mkdir -p trainlogs

export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.01
export WANDB_MODE=disabled

echo ""
echo "=== 开始 WUDI 融合 (base=pi05_spatial, ft=object+goal, scaling=0.2) ==="

uv run python scripts/wudi_merge.py \
    --base   "${BASE}" \
    --ft \
        checkpoints/pi05_libero_object/my_experiment/29999/params \
        checkpoints/pi05_libero_goal/my_experiment/29999/params \
    --output  "${OUTPUT}" \
    --scope   llm_only \
    --iter    300 \
    --scaling  0.2 \
    --scaling2 1.0 \
    --device   cuda

echo ""
echo "=== 融合完成: $(date) ==="

if [ ! -d "${OUTPUT}/params" ]; then
    echo "错误：融合失败！"
    exit 1
fi

echo "复制 assets (from pi05_libero_spatial)..."
cp -r checkpoints/pi05_libero_spatial/my_experiment/29999/assets "${OUTPUT}/assets"
echo "完成。输出: ${OUTPUT}"
