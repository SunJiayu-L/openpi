#!/bin/bash

#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:2
#SBATCH --nodelist=gnho034
#SBATCH -J eval_wudi_4task_500
#SBATCH -o /storage/yukaichengLab/lishiwen/jiayusun/openpi/trainlogs/eval_wudi_4task_500.%j.log
#SBATCH -e /storage/yukaichengLab/lishiwen/jiayusun/openpi/trainlogs/eval_wudi_4task_500.%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# Evaluate WUDI-MLLM 4task 500 iter merged checkpoint
# Checkpoint: checkpoints/wudi_mllm/4task_from10k_iter500
# GPU0 for simulator (MuJoCo EGL), GPU1 for policy (JAX)

set -eo pipefail
export PYTHONFAULTHANDLER=1

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi
mkdir -p trainlogs

export HF_HOME=~/.cache/huggingface
export HF_HUB_OFFLINE=1

CKPT_DIR="checkpoints/wudi_mllm/4task_from10k_iter500"
echo "Starting policy server with checkpoint: $CKPT_DIR"

CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/serve_policy.py \
    --port 8000 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$CKPT_DIR" &
SERVER_PID=$!
echo "Policy server PID: $SERVER_PID"

echo "Waiting for policy server to start..."
for i in $(seq 1 300); do
    if curl -s http://localhost:8000/healthz > /dev/null 2>&1; then
        echo "Policy server is ready! (waited ${i}s)"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: Policy server process died"
        exit 1
    fi
    sleep 1
done

LIBERO_VENV=examples/libero/.venv
source "$LIBERO_VENV/bin/activate"
export PYTHONPATH=$PWD/third_party/libero:$PYTHONPATH

NUM_TRIALS=50
SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")

for SUITE in "${SUITES[@]}"; do
    VIDEO_OUT="data/libero/videos/wudi_4task_500/${SUITE}"
    mkdir -p "$VIDEO_OUT"
    echo ""
    echo "=========================================="
    echo "Evaluating: $SUITE (${NUM_TRIALS} trials per task)"
    echo "Start: $(date)"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYOPENGL_PLATFORM=egl \
    python -X faulthandler examples/libero/main.py \
        --args.task-suite-name "$SUITE" \
        --args.num-trials-per-task "$NUM_TRIALS" \
        --args.video-out-path "$VIDEO_OUT" \
        --args.host 0.0.0.0 \
        --args.port 8000

    echo "Finished $SUITE at $(date)"
done

deactivate

echo ""
echo "Stopping policy server..."
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null

echo "=========================================="
echo "All evaluations complete!"
echo "End time: $(date)"
echo "=========================================="
