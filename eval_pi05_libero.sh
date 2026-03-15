#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gnho034
#SBATCH -J pi05_libero_eval
#SBATCH -o trainlogs/pi05_libero_eval.%j.log
#SBATCH -e trainlogs/pi05_libero_eval.%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

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

# ============================================================
# Step 1: Start policy server in background
# ============================================================
CKPT_DIR="checkpoints/pi05_libero/my_experiment/29999"
echo "Starting policy server with checkpoint: $CKPT_DIR"

uv run scripts/serve_policy.py \
    --port 8000 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$CKPT_DIR" &
SERVER_PID=$!
echo "Policy server PID: $SERVER_PID"

# Wait for server to be ready
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

# ============================================================
# Step 2: Run LIBERO evaluation
# ============================================================
LIBERO_VENV=examples/libero/.venv
source "$LIBERO_VENV/bin/activate"
export PYTHONPATH=$PWD/third_party/libero:$PYTHONPATH

TASK_SUITE="libero_spatial"
NUM_TRIALS=3
VIDEO_OUT="data/libero/videos"

echo "Running LIBERO eval: suite=$TASK_SUITE, trials=$NUM_TRIALS"
python examples/libero/main.py \
    --args.task-suite-name "$TASK_SUITE" \
    --args.num-trials-per-task "$NUM_TRIALS" \
    --args.video-out-path "$VIDEO_OUT" \
    --args.host 0.0.0.0 \
    --args.port 8000

deactivate

# ============================================================
# Cleanup
# ============================================================
echo "Stopping policy server..."
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null

echo "End time: $(date)"
echo "Videos saved to: $VIDEO_OUT"
