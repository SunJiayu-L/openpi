#!/bin/bash
# 评测 4 个单 suite 微调模型在所有 LIBERO suite 上的表现
# 4 个任务并行，每个占 1 GPU，使用不同端口

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi

MODELS=("pi05_libero_object" "pi05_libero_spatial" "pi05_libero_10" "pi05_libero_goal")
SHORT=("obj" "spat" "l10" "goal")
PORTS=(8200 8201 8202 8203)

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    TAG="${SHORT[$i]}"
    PORT="${PORTS[$i]}"

    JOB_ID=$(sbatch --parsable eval_finetuned_model.sh "$MODEL" "$TAG" "$PORT")
    echo "Submitted ${MODEL} (port ${PORT}) as Job ${JOB_ID}"
done

echo "All 4 jobs submitted in parallel on gnho031."
