#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:4
#SBATCH --nodelist=gnho034
#SBATCH -J jiayu_2h_merge_gd_frozen_large
#SBATCH -o trainlogs/merge_gd_frozen_large_%j.log
#SBATCH -e trainlogs/merge_gd_frozen_large_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G

set -eo pipefail

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi
mkdir -p trainlogs checkpoints/pi05_libero_4task_merge_gd_frozen_large wandb_offline

BASE=/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints

export TF_CPP_MIN_LOG_LEVEL=3
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.98
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

.venv/bin/python -u scripts/arithmetic.py \
    --config pi05_libero \
    --data-path /storage/yukaichengLab/lishiwen/jiayusun/openpi/pkl/libero_val_balanced_10pct.pkl \
    --checkpoints \
        $BASE/pi05_libero_spatial/my_experiment/29999 \
        $BASE/pi05_libero_object/my_experiment/29999 \
        $BASE/pi05_libero_goal/my_experiment/29999 \
        $BASE/pi05_libero_10/my_experiment/29999 \
    --output /storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero_4task_merge_gd_frozen_large \
    --optimize_method gradient_descent \
    --num_iterations 856 \
    --learning_rate 0.1 \
    --freeze_action \
    --action_src $BASE/pi05_libero_10/my_experiment/29999 \
    --wandb_project openpi_merge \
    --wandb_run_name 4task_gd_frozen_large \
    --wandb_dir /storage/yukaichengLab/lishiwen/jiayusun/openpi/wandb_offline

echo "Done: $(date)"
echo ""
echo "To sync wandb logs, run on login node:"
echo "  cd /storage/yukaichengLab/lishiwen/jiayusun/openpi"
echo "  source proxy_on.sh && .venv/bin/wandb sync wandb_offline/wandb/offline-run-*"
