#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gnho031
#SBATCH -J pi05_eval_full_all
#SBATCH -o trainlogs/pi05_libero_full_alltask.%j.log
#SBATCH -e trainlogs/pi05_libero_full_alltask.%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

set -eo pipefail

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi

# Offline mode
export HF_HOME=~/.cache/huggingface
export HF_HUB_OFFLINE=1

# Headless GPU rendering
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export PYOPENGL_PLATFORM=egl

LIBERO_VENV=examples/libero/.venv
NUM_TRIALS=20
PORT=8000

SUITES=("libero_10" "libero_goal" "libero_object" "libero_spatial")
STEPS=(5000 10000 15000 20000 25000 29999)

for STEP in "${STEPS[@]}"; do
    CKPT_DIR="checkpoints/pi05_libero/my_experiment/${STEP}"
    echo ""
    echo "############################################################"
    echo "# Checkpoint step: ${STEP} (full 40-task model, all tasks)"
    echo "############################################################"

    # Start policy server
    echo "Starting policy server with checkpoint: $CKPT_DIR"
    uv run scripts/serve_policy.py \
        --port $PORT \
        policy:checkpoint \
        --policy.config pi05_libero \
        --policy.dir "$CKPT_DIR" &
    SERVER_PID=$!

    # Wait for server
    echo "Waiting for policy server to start..."
    for i in $(seq 1 300); do
        if curl -s http://localhost:${PORT}/healthz > /dev/null 2>&1; then
            echo "Policy server is ready! (waited ${i}s)"
            break
        fi
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "ERROR: Policy server process died"
            exit 1
        fi
        sleep 1
    done

    # Evaluate all tasks in all 4 suites (10 tasks each, 20 trials per task)
    source "$LIBERO_VENV/bin/activate"
    export PYTHONPATH=$PWD/third_party/libero:$PYTHONPATH

    for SUITE in "${SUITES[@]}"; do
        VIDEO_OUT="data/libero/videos/full_alltask_step${STEP}/${SUITE}"
        echo ""
        echo "=========================================="
        echo "[Step ${STEP}] ${SUITE} — all 10 tasks, ${NUM_TRIALS} trials each"
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

    # Stop server before next checkpoint
    echo "Stopping policy server..."
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    echo "[Step ${STEP}] Done."
done

echo ""
echo "############################################################"
echo "All evaluations (6 checkpoints x 4 suites x 10 tasks x 20 trials) complete!"
echo "End time: $(date)"
echo "############################################################"
