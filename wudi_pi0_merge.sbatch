#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:2
#SBATCH --exclude=gnho009
#SBATCH --mem=160G
#SBATCH --cpus-per-task=8
#SBATCH -J wudi_pi0_merge
#SBATCH -o trainlogs/wudi_pi0_merge_%j.log
#SBATCH -e trainlogs/wudi_pi0_merge_%j.err

BASE=/storage/yukaichengLab/lishiwen/.cache/openpi/openpi-assets/checkpoints/pi0_base/params
OUTPUT=checkpoints/merged/wudi_pi0_5task_iter300

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
echo "=== 开始 WUDI 融合 (pi0, 5 tasks, iter=300) ==="

uv run python scripts/wudi_merge.py \
    --base   "${BASE}" \
    --ft \
        checkpoints/pi0_libero/my_experiment/29999/params \
        checkpoints/pi0_libero_10/my_experiment/29999/params \
        checkpoints/pi0_libero_object/my_experiment/29999/params \
        checkpoints/pi0_libero_spatial/my_experiment/29999/params \
        checkpoints/pi0_libero_goal/my_experiment/29999/params \
    --output  "${OUTPUT}" \
    --scope   llm_only \
    --iter    300 \
    --scaling  0.1 \
    --scaling2 1.0 \
    --device   cuda

echo ""
echo "=== 融合完成: $(date) ==="

if [ ! -d "${OUTPUT}/params" ]; then
    echo "错误：融合失败！"
    exit 1
fi

echo "复制 assets..."
cp -r checkpoints/pi0_libero/my_experiment/29999/assets "${OUTPUT}/assets"
echo "完成。输出: ${OUTPUT}"
