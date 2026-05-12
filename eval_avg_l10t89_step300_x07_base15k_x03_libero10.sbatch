#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:2
#SBATCH --nodelist=gnho031
#SBATCH -J eval_avg_t89_x07
#SBATCH -o /storage/yukaichengLab/lishiwen/jiayusun/openpi/trainlogs/eval_avg_l10t89_step300_x07_base15k_x03.%j.log
#SBATCH -e /storage/yukaichengLab/lishiwen/jiayusun/openpi/trainlogs/eval_avg_l10t89_step300_x07_base15k_x03.%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

set -eo pipefail
export PYTHONFAULTHANDLER=1
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi
mkdir -p trainlogs
export HF_HOME=~/.cache/huggingface
export HF_HUB_OFFLINE=1

CKPT_DIR="checkpoints/avg_merge/l10t89_step300_x07_base15k_x03"
TAG="avg_l10t89_step300_x07_base15k_x03"

CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/serve_policy.py \
    --port 8021 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$CKPT_DIR" &
SERVER_PID=$!

for i in $(seq 1 300); do
    if curl -s http://localhost:8021/healthz > /dev/null 2>&1; then echo "Server ready (${i}s)"; break; fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then echo "ERROR: server died"; exit 1; fi
    sleep 1
done

source examples/libero/.venv/bin/activate
export PYTHONPATH=$PWD/third_party/libero:$PYTHONPATH

mkdir -p "data/libero/videos/${TAG}/libero_10"
echo "=== Evaluating: libero_10 | $(date) ==="
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYOPENGL_PLATFORM=egl \
python -X faulthandler examples/libero/main.py \
    --args.task-suite-name libero_10 \
    --args.num-trials-per-task 50 \
    --args.video-out-path "data/libero/videos/${TAG}/libero_10" \
    --args.host 0.0.0.0 --args.port 8021
echo "Finished libero_10 at $(date)"

deactivate
kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null
echo "All done: $(date)"
