#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:2
#SBATCH --nodelist=gnho031
#SBATCH -J eval_t8x200
#SBATCH -o /storage/yukaichengLab/lishiwen/jiayusun/openpi/trainlogs/eval_4task_wudi1k_9k_task8x200.%j.log
#SBATCH -e /storage/yukaichengLab/lishiwen/jiayusun/openpi/trainlogs/eval_4task_wudi1k_9k_task8x200.%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# Eval: ft_from_wudi_4task_1k/9000
# Suite: libero_10, task 8 only ("put both moka pots on the stove")
# Trials: 200 (for reliable estimate of true success rate)

set -eo pipefail
export PYTHONFAULTHANDLER=1
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi
mkdir -p trainlogs
export HF_HOME=~/.cache/huggingface
export HF_HUB_OFFLINE=1

CKPT_DIR="checkpoints/pi05_libero_4task_from_wudi_4task_1k/ft_from_wudi_4task_1k/9000"
TAG="4task_wudi1k_9k_task8x200"
PORT=8025

CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/serve_policy.py \
    --port $PORT \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$CKPT_DIR" &
SERVER_PID=$!

for i in $(seq 1 300); do
    if curl -s http://localhost:${PORT}/healthz > /dev/null 2>&1; then echo "Server ready (${i}s)"; break; fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then echo "ERROR: server died"; exit 1; fi
    sleep 1
done

source examples/libero/.venv/bin/activate
export PYTHONPATH=$PWD/third_party/libero:$PYTHONPATH

mkdir -p "data/libero/videos/${TAG}/libero_10"
echo "=== Evaluating libero_10 task 8 x200 | $(date) ==="
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYOPENGL_PLATFORM=egl \
python -X faulthandler examples/libero/main.py \
    --args.task-suite-name libero_10 \
    --args.task-ids 8 \
    --args.num-trials-per-task 200 \
    --args.video-out-path "data/libero/videos/${TAG}/libero_10" \
    --args.host 0.0.0.0 --args.port $PORT
echo "=== Done: $(date) ==="

deactivate
kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null || true
echo "All done: $(date)"
