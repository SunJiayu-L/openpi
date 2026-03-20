#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gnho031
#SBATCH -J eval_ft
#SBATCH -o trainlogs/eval_ft_%x.%j.log
#SBATCH -e trainlogs/eval_ft_%x.%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G

set -eo pipefail

MODEL_CONFIG="$1"   # e.g. pi05_libero_object
TAG="$2"            # e.g. obj
PORT_ARG="$3"       # e.g. 8200 (each parallel job needs a unique port)

if [ -z "$MODEL_CONFIG" ] || [ -z "$TAG" ]; then
    echo "Usage: sbatch eval_finetuned_model.sh <config_name> <tag> [port]"
    exit 1
fi

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Model: $MODEL_CONFIG (tag: $TAG)"
echo "Start time: $(date)"

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi

export HF_HOME=~/.cache/huggingface
export HF_HUB_OFFLINE=1
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export PYOPENGL_PLATFORM=egl

LIBERO_VENV=examples/libero/.venv
NUM_TRIALS=20
PORT=${PORT_ARG:-8200}

CKPT_ROOT="checkpoints/${MODEL_CONFIG}/my_experiment"
STEPS=(5000 15000 29999)
SUITES=("libero_10" "libero_goal" "libero_object" "libero_spatial")

for STEP in "${STEPS[@]}"; do
    CKPT_DIR="${CKPT_ROOT}/${STEP}"

    if [ ! -d "$CKPT_DIR" ]; then
        echo "WARNING: Checkpoint ${CKPT_DIR} not found, skipping."
        continue
    fi

    echo ""
    echo "============================================================"
    echo "  Model: ${MODEL_CONFIG}  Step: ${STEP}"
    echo "  Checkpoint: ${CKPT_DIR}"
    echo "  Start: $(date)"
    echo "============================================================"

    # Start policy server
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/serve_policy.py \
        --env LIBERO \
        --port $PORT \
        policy:checkpoint \
        --policy.config "$MODEL_CONFIG" \
        --policy.dir "$CKPT_DIR" &
    SERVER_PID=$!

    echo "Waiting for server (PID: $SERVER_PID)..."
    SERVER_READY=0
    for i in $(seq 1 600); do
        if curl -s http://localhost:${PORT}/healthz > /dev/null 2>&1; then
            echo "Server ready! (waited ${i}s)"
            SERVER_READY=1
            break
        fi
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "ERROR: Server died for ${MODEL_CONFIG} step ${STEP}"
            break
        fi
        sleep 1
    done

    if [ "$SERVER_READY" -eq 0 ]; then
        echo "ERROR: Server failed to start, skipping step ${STEP}"
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
        continue
    fi

    # Evaluate all suites
    source "$LIBERO_VENV/bin/activate"
    export PYTHONPATH=$PWD/third_party/libero:$PYTHONPATH

    for SUITE in "${SUITES[@]}"; do
        VIDEO_OUT="data/libero/videos/eval_ft_${TAG}_step${STEP}/${SUITE}"
        echo ""
        echo "  ── ${SUITE} (${NUM_TRIALS} trials/task) ──"
        echo "  Start: $(date)"

        python examples/libero/main.py \
            --args.task-suite-name "$SUITE" \
            --args.num-trials-per-task "$NUM_TRIALS" \
            --args.video-out-path "$VIDEO_OUT" \
            --args.host 0.0.0.0 \
            --args.port $PORT

        echo "  Finished ${SUITE} at $(date)"
    done

    deactivate

    # Kill server for this step
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    echo "Server stopped for step ${STEP}"
done

echo ""
echo "============================================================"
echo "All evaluations complete for ${MODEL_CONFIG}!"
echo "End time: $(date)"
echo "============================================================"
