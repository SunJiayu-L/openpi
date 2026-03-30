#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:2
#SBATCH --nodelist=gnho031
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH -J wudi_eval_suite
#SBATCH -o trainlogs/wudi_eval_%x_%j.log
#SBATCH -e trainlogs/wudi_eval_%x_%j.err

# Usage: sbatch --job-name=<suite> wudi_eval_suite.sbatch <suite> <ckpt_dir> <config>
SUITE="$1"
CKPT_DIR="${2:-checkpoints/merged/wudi_test_pi05base_5k_s1}"
CONFIG="${3:-pi05_libero}"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Suite: $SUITE"
echo "Checkpoint: $CKPT_DIR"
echo "Start: $(date)"

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi

export HF_HOME=~/.cache/huggingface
export HF_HUB_OFFLINE=1
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export PYOPENGL_PLATFORM=egl

LIBERO_VENV=examples/libero/.venv
NUM_TRIALS=20
PORT=850${SLURM_JOB_ID: -1}  # unique port per job

echo "Port: $PORT"

# GPU1: policy server (JAX), GPU0: simulator (EGL).
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/serve_policy.py \
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

VIDEO_OUT="data/libero/videos/wudi_test_pi05base_5k/${SUITE}"
echo ""
echo "=========================================="
echo "Suite: $SUITE  (${NUM_TRIALS} trials/task)"
echo "Start: $(date)"
echo "=========================================="

CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python examples/libero/main.py \
    --args.task-suite-name "$SUITE" \
    --args.num-trials-per-task "$NUM_TRIALS" \
    --args.host 0.0.0.0 \
    --args.port $PORT || true

deactivate

kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo ""
echo "Done: $SUITE at $(date)"
