#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gnho034
#SBATCH -J merge_eval
#SBATCH -o trainlogs/merge_eval.%j.log
#SBATCH -e trainlogs/merge_eval.%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

set -eo pipefail

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi

export HF_HOME=~/.cache/huggingface
export HF_HUB_OFFLINE=1
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export PYOPENGL_PLATFORM=egl

LIBERO_VENV=examples/libero/.venv
NUM_TRIALS=20
PORT=8100

CKPT_OBJ="checkpoints/pi05_libero_object/my_experiment/29999"
CKPT_GOAL="checkpoints/pi05_libero_goal/my_experiment/29999"
CKPT_BASE="/storage/yukaichengLab/lishiwen/.cache/openpi/openpi-assets/checkpoints/pi05_base"

SUITES=("libero_10" "libero_goal" "libero_object" "libero_spatial")

echo "############################################################"
echo "# Merging: object(0.4) + goal(0.4) + base(0.2)"
echo "############################################################"

# Start merged policy server
SAVE_PATH="checkpoints/merged/pi05_obj04_goal04_base02"

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/merging_experiments.py \
    --port $PORT \
    --config pi05_libero \
    --merging_fn linear_interpolation \
    --merging_fn_kwargs '{"model_mixing_coefficients": [0.4, 0.4, 0.2]}' \
    --checkpoint_dirs "$CKPT_OBJ" "$CKPT_GOAL" "$CKPT_BASE" \
    --save-path "$SAVE_PATH" &
SERVER_PID=$!

# Wait for server
echo "Waiting for merged policy server to start..."
for i in $(seq 1 600); do
    if curl -s http://localhost:${PORT}/healthz > /dev/null 2>&1; then
        echo "Server ready! (waited ${i}s)"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: Server died"
        exit 1
    fi
    sleep 1
done

# Evaluate all 4 suites
source "$LIBERO_VENV/bin/activate"
export PYTHONPATH=$PWD/third_party/libero:$PYTHONPATH

for SUITE in "${SUITES[@]}"; do
    VIDEO_OUT="data/libero/videos/merged_obj04_goal04_base02/${SUITE}"
    echo ""
    echo "=========================================="
    echo "Evaluating ${SUITE} — all 10 tasks, ${NUM_TRIALS} trials each"
    echo "Start: $(date)"
    echo "=========================================="

    python examples/libero/main.py \
        --args.task-suite-name "$SUITE" \
        --args.num-trials-per-task "$NUM_TRIALS" \
        --args.video-out-path "$VIDEO_OUT" \
        --args.host 0.0.0.0 \
        --args.port $PORT

    echo "Finished ${SUITE} at $(date)"
done

deactivate

kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo ""
echo "############################################################"
echo "Merge + Eval complete!"
echo "End time: $(date)"
echo "############################################################"
