#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:1
#SBATCH -J wudi_merged_eval
#SBATCH -o trainlogs/wudi_merged_eval.%j.log
#SBATCH -e trainlogs/wudi_merged_eval.%j.err
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
PORT=8000

CKPT_DIR="checkpoints/merged/wudi_e1_libero_goal"
CONFIG="pi05_libero_goal"
SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")

echo ""
echo "############################################################"
echo "# Checkpoint : ${CKPT_DIR}"
echo "# Config     : ${CONFIG}"
echo "# Suites     : ${SUITES[*]}"
echo "# Trials/task: ${NUM_TRIALS}"
echo "############################################################"

# Start policy server
echo "Starting policy server..."
uv run scripts/serve_policy.py \
    --port $PORT \
    policy:checkpoint \
    --policy.config "$CONFIG" \
    --policy.dir "$CKPT_DIR" &
SERVER_PID=$!
echo "Policy server PID: $SERVER_PID"

# Wait for server ready
echo "Waiting for policy server..."
for i in $(seq 1 300); do
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

# Evaluate all suites
source "$LIBERO_VENV/bin/activate"
export PYTHONPATH=$PWD/third_party/libero:$PYTHONPATH

for SUITE in "${SUITES[@]}"; do
    VIDEO_OUT="data/libero/videos/wudi_merged/${SUITE}"
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
        --args.port $PORT

    echo "Finished $SUITE at $(date)"
done

deactivate

# Cleanup
echo ""
echo "Stopping policy server..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo ""
echo "############################################################"
echo "All evaluations complete! (4 suites x 10 tasks x ${NUM_TRIALS} trials)"
echo "End time: $(date)"
echo "Videos: data/libero/videos/wudi_merged/"
echo "############################################################"
