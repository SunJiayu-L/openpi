#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:2
#SBATCH --nodelist=gnho031
#SBATCH -J eval_t8_300_499
#SBATCH -o /storage/yukaichengLab/lishiwen/jiayusun/openpi/trainlogs/eval_l10_t8_from_avg_x07.%j.log
#SBATCH -e /storage/yukaichengLab/lishiwen/jiayusun/openpi/trainlogs/eval_l10_t8_from_avg_x07.%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# Eval: ft_l10_t8_from_avg_x07 / step 300 + step 499
# Suite: libero_10 only
# Runs both checkpoints sequentially in one job

set -eo pipefail
export PYTHONFAULTHANDLER=1
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi
mkdir -p trainlogs
export HF_HOME=~/.cache/huggingface
export HF_HUB_OFFLINE=1

BASE=checkpoints/pi05_libero_10_t8_from_avg_x07/ft_l10_t8_from_avg_x07

source examples/libero/.venv/bin/activate
export PYTHONPATH=$PWD/third_party/libero:$PYTHONPATH

# ── step 300 ──────────────────────────────────────────────
TAG="l10_t8_from_avg_x07_step300"
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/serve_policy.py \
    --port 8022 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$BASE/300" &
SERVER_PID=$!

for i in $(seq 1 300); do
    if curl -s http://localhost:8022/healthz > /dev/null 2>&1; then echo "Server ready (${i}s)"; break; fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then echo "ERROR: server died"; exit 1; fi
    sleep 1
done

mkdir -p "data/libero/videos/${TAG}/libero_10"
echo "=== [step300] Evaluating libero_10 | $(date) ==="
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYOPENGL_PLATFORM=egl \
python -X faulthandler examples/libero/main.py \
    --args.task-suite-name libero_10 \
    --args.num-trials-per-task 50 \
    --args.video-out-path "data/libero/videos/${TAG}/libero_10" \
    --args.host 0.0.0.0 --args.port 8022
echo "=== [step300] Done: $(date) ==="

kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null || true
sleep 5

# ── step 499 ──────────────────────────────────────────────
TAG="l10_t8_from_avg_x07_step499"
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/serve_policy.py \
    --port 8023 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$BASE/499" &
SERVER_PID=$!

for i in $(seq 1 300); do
    if curl -s http://localhost:8023/healthz > /dev/null 2>&1; then echo "Server ready (${i}s)"; break; fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then echo "ERROR: server died"; exit 1; fi
    sleep 1
done

mkdir -p "data/libero/videos/${TAG}/libero_10"
echo "=== [step499] Evaluating libero_10 | $(date) ==="
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYOPENGL_PLATFORM=egl \
python -X faulthandler examples/libero/main.py \
    --args.task-suite-name libero_10 \
    --args.num-trials-per-task 50 \
    --args.video-out-path "data/libero/videos/${TAG}/libero_10" \
    --args.host 0.0.0.0 --args.port 8023
echo "=== [step499] Done: $(date) ==="

kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null || true

deactivate
echo "All done: $(date)"
