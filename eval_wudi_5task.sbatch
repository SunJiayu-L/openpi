#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:1
#SBATCH -J wudi5_eval
#SBATCH -o trainlogs/wudi5_eval_%j.log
#SBATCH -e trainlogs/wudi5_eval_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# Usage:
#   sbatch eval_wudi_5task.sbatch <CKPT_NAME>
# CKPT_NAME: e.g. wudi_5task_30k, wudi_5task_30k_s01
#
# Evaluates checkpoints/merged/<CKPT_NAME> on all 4 LIBERO suites,
# 20 trials per task.

CKPT_NAME=${1:-wudi_5task_30k}

CKPT_DIR="checkpoints/merged/${CKPT_NAME}"
CONFIG="pi05_libero"
SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")
NUM_TRIALS=20
PORT=8000

LIBERO_VENV=examples/libero/.venv

echo "Job ID    : ${SLURM_JOB_ID}"
echo "Node      : ${SLURM_NODENAME}"
echo "Ckpt Name : ${CKPT_NAME}"
echo "Checkpoint: ${CKPT_DIR}"
echo "Config    : ${CONFIG}"
echo "Suites    : ${SUITES[*]}"
echo "Trials    : ${NUM_TRIALS}"
echo "Start     : $(date)"

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi

export HF_HOME=~/.cache/huggingface
export HF_HUB_OFFLINE=1
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export PYOPENGL_PLATFORM=egl

mkdir -p trainlogs

# Verify checkpoint exists
if [ ! -d "${CKPT_DIR}/params" ]; then
    echo "ERROR: Checkpoint not found at ${CKPT_DIR}/params"
    exit 1
fi

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
    VIDEO_OUT="data/libero/videos/${CKPT_NAME}/${SUITE}"
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
echo "All evaluations complete!"
echo "Ckpt: ${CKPT_NAME}  |  4 suites x 10 tasks x ${NUM_TRIALS} trials"
echo "End time: $(date)"
echo "Videos: data/libero/videos/wudi_5task_${TAG}/"
echo "############################################################"
