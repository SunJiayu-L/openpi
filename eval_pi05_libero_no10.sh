#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gnho031
#SBATCH -J pi05_eval_split
#SBATCH -o trainlogs/pi05_libero_split_eval.%j.log
#SBATCH -e trainlogs/pi05_libero_split_eval.%j.err
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
NUM_TRIALS=50
PORT=8000

# Correct mapping: dataset task_index -> suite + suite_task_id
# task_index 0  -> libero_10     suite_task 4
# task_index 10 -> libero_goal   suite_task 8
# task_index 20 -> libero_object suite_task 9
# task_index 30 -> libero_spatial suite_task 6
SUITES=("libero_10" "libero_goal" "libero_object" "libero_spatial")
SUITE_TASK_IDS=(4 8 9 6)
DS_TASK_INDICES=(0 10 20 30)

STEPS=(10000 15000)

for STEP in "${STEPS[@]}"; do
    CKPT_DIR="checkpoints/pi05_libero_no10/split_experiment/${STEP}"
    echo ""
    echo "############################################################"
    echo "# Checkpoint step: ${STEP}"
    echo "############################################################"

    # Start policy server
    echo "Starting policy server with checkpoint: $CKPT_DIR"
    uv run scripts/serve_policy.py \
        --port $PORT \
        policy:checkpoint \
        --policy.config pi05_libero_no10 \
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

    # Evaluate 4 held-out test tasks
    source "$LIBERO_VENV/bin/activate"
    export PYTHONPATH=$PWD/third_party/libero:$PYTHONPATH

    for idx in 0 1 2 3; do
        SUITE=${SUITES[$idx]}
        TASK_ID=${SUITE_TASK_IDS[$idx]}
        DS_IDX=${DS_TASK_INDICES[$idx]}
        VIDEO_OUT="data/libero/videos/split_eval_step${STEP}/${SUITE}_ds${DS_IDX}"
        echo ""
        echo "=========================================="
        echo "[Step ${STEP}] ${SUITE} suite_task ${TASK_ID} (dataset task_index ${DS_IDX})"
        echo "  ${NUM_TRIALS} trials"
        echo "Start: $(date)"
        echo "=========================================="

        python examples/libero/main.py \
            --args.task-suite-name "$SUITE" \
            --args.num-trials-per-task "$NUM_TRIALS" \
            --args.video-out-path "$VIDEO_OUT" \
            --args.host 0.0.0.0 \
            --args.port $PORT \
            --args.task-ids $TASK_ID

        echo "Finished ${SUITE} suite_task ${TASK_ID} at $(date)"
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
echo "All evaluations (5k, 10k, 14k) complete!"
echo "End time: $(date)"
echo "############################################################"
