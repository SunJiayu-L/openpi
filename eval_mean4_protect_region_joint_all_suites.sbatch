#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:2
#SBATCH --nodelist=gnho034
#SBATCH -J eval_m4_reg_joint
#SBATCH -o /storage/yukaichengLab/lishiwen/jiayusun/openpi/trainlogs/eval_m4_reg_joint.%j.log
#SBATCH -e /storage/yukaichengLab/lishiwen/jiayusun/openpi/trainlogs/eval_m4_reg_joint.%j.err
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

CKPT_DIR="checkpoints/ablation_selective_protect/mean4_protect_region_joint"
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/serve_policy.py \
    --port 8000 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$CKPT_DIR" &
SERVER_PID=$!

for i in $(seq 1 300); do
    if curl -s http://localhost:8000/healthz > /dev/null 2>&1; then echo "Server ready (${i}s)"; break; fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then echo "ERROR: server died"; exit 1; fi
    sleep 1
done

source examples/libero/.venv/bin/activate
export PYTHONPATH=$PWD/third_party/libero:$PYTHONPATH

for SUITE in libero_spatial libero_object libero_goal libero_10; do
    mkdir -p "data/libero/videos/mean4_protect_region_joint/${SUITE}"
    echo "=== Evaluating: $SUITE | $(date) ==="
    CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYOPENGL_PLATFORM=egl \
    python -X faulthandler examples/libero/main.py \
        --args.task-suite-name "$SUITE" \
        --args.num-trials-per-task 50 \
        --args.video-out-path "data/libero/videos/mean4_protect_region_joint/${SUITE}" \
        --args.host 0.0.0.0 --args.port 8000
    echo "Finished $SUITE at $(date)"
done

deactivate
kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null
echo "All done: $(date)"

