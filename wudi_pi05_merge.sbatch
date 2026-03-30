#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:2
#SBATCH --exclude=gnho009
#SBATCH --mem=160G
#SBATCH --cpus-per-task=8
#SBATCH -J wudi_pi05_merge
#SBATCH -o trainlogs/wudi_pi05_merge_%j.log
#SBATCH -e trainlogs/wudi_pi05_merge_%j.err

BASE=/storage/yukaichengLab/lishiwen/jiayusun/openpi_pt/pi05_model/pi05_base/params
OUTPUT=checkpoints/merged/wudi_pi05_5task_iter300_s1

echo "Job ID  : ${SLURM_JOB_ID}"
echo "Node    : ${SLURM_NODELIST}"
echo "Output  : ${OUTPUT}"
echo "Start   : $(date)"

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi
mkdir -p trainlogs

export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.01
export WANDB_MODE=offline

echo ""
echo "=== 开始 WUDI 融合 (pi05, 5 tasks, iter=300, scaling=1) ==="

uv run python scripts/wudi_merge.py \
    --base   "${BASE}" \
    --ft \
        checkpoints/pi05_libero/my_experiment/29999/params \
        checkpoints/pi05_libero_10/my_experiment/29999/params \
        checkpoints/pi05_libero_object/my_experiment/29999/params \
        checkpoints/pi05_libero_spatial/my_experiment/29999/params \
        checkpoints/pi05_libero_goal/my_experiment/29999/params \
    --output  "${OUTPUT}" \
    --scope   llm_only \
    --iter    300 \
    --scaling  1.0 \
    --scaling2 1.0 \
    --device   cuda

echo ""
echo "=== 融合完成: $(date) ==="

if [ ! -d "${OUTPUT}/params" ]; then
    echo "错误：融合失败！"
    exit 1
fi

echo "复制 assets (from pi05_libero)..."
cp -r checkpoints/pi05_libero/my_experiment/29999/assets "${OUTPUT}/assets"
echo "完成。输出: ${OUTPUT}"
