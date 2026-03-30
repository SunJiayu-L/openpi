#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:1
#SBATCH --exclude=gnho009
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH -J wudi_test_eval
#SBATCH -o trainlogs/wudi_test_eval_%j.log
#SBATCH -e trainlogs/wudi_test_eval_%j.err

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
PORT=8500
CKPT_DIR="checkpoints/merged/wudi_test_pi05base_5k_iter5"
CONFIG="pi05_libero"
SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")

echo ""
echo "############################################################"
echo "# Checkpoint : ${CKPT_DIR}"
echo "# Config     : ${CONFIG}"
echo "# Suites     : ${SUITES[*]}"
echo "# Trials/task: ${NUM_TRIALS}"
echo "############################################################"

# Start policy server
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/serve_policy.py \
    --env LIBERO \
    --port $PORT \
    policy:checkpoint \
    --policy.config "$CONFIG" \
    --policy.dir "$CKPT_DIR" &
SERVER_PID=$!

echo "Waiting for server (PID: $SERVER_PID)..."
for i in $(seq 1 600); do
    if curl -s http://localhost:${PORT}/healthz > /dev/null 2>&1; then
        echo "Server ready! (waited ${i}s)"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: Policy server died"
        exit 1
    fi
    sleep 1
done

source "$LIBERO_VENV/bin/activate"
export PYTHONPATH=$PWD/third_party/libero:$PYTHONPATH

for SUITE in "${SUITES[@]}"; do
    VIDEO_OUT="data/libero/videos/wudi_test_pi05base_5k/${SUITE}"
    echo ""
    echo "=========================================="
    echo "Suite: $SUITE  (${NUM_TRIALS} trials/task)"
    echo "Start: $(date)"
    echo "=========================================="

    python examples/libero/main.py \
        --args.task-suite-name "$SUITE" \
        --args.num-trials-per-task "$NUM_TRIALS" \
        --args.video-out-path "$VIDEO_OUT" \
        --args.host 0.0.0.0 \
        --args.port $PORT || true

    echo "Finished $SUITE at $(date)"
done

deactivate

kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo ""
echo "############################################################"
echo "All evaluations complete!"
echo "End time: $(date)"
echo "############################################################"
