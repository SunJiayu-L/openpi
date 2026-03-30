#!/bin/bash
# Evaluate merged checkpoint on corresponding libero_10 tasks:
# task_id=4 (white mug left/right plates),
# task_id=8 (both moka pots on stove),
# task_id=5 (book -> caddy).

#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=gnho031,gnho034
#SBATCH --exclude=gnho009
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH -J eval_l10_3t_nv
#SBATCH -o /storage/yukaichengLab/lishiwen/jiayusun/openpi/trainlogs/eval_l10_3t_nv_%j.log
#SBATCH -e /storage/yukaichengLab/lishiwen/jiayusun/openpi/trainlogs/eval_l10_3t_nv_%j.err

set -euo pipefail

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi
mkdir -p trainlogs

PORT=$((30000 + (SLURM_JOB_ID % 20000)))
CKPT_DIR="/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/merged/wudi_libero10_3task_iter1000_s1_curve"
CONFIG="pi05_libero10_task0_retrain"
VIDEO_OUT="data/libero/videos/libero10_3task_wudi_${SLURM_JOB_ID}"

export HF_HOME=~/.cache/huggingface
export HF_HUB_OFFLINE=1
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export PYOPENGL_PLATFORM=egl
export PYTHONFAULTHANDLER=1

echo "[EVAL-L10-3TASK] Job: ${SLURM_JOB_ID} Node: ${SLURMD_NODENAME}"
echo "[EVAL-L10-3TASK] CKPT: ${CKPT_DIR}"
echo "[EVAL-L10-3TASK] task_ids: 4 8 5, trials/task=50"
echo "[EVAL-L10-3TASK] Start: $(date)"

# GPU1: policy server (JAX), GPU0: simulator (EGL).
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/serve_policy.py \
  --env LIBERO \
  --port "${PORT}" \
  policy:checkpoint \
  --policy.config "${CONFIG}" \
  --policy.dir "${CKPT_DIR}" &
SERVER_PID=$!

for i in $(seq 1 600); do
  if curl -s "http://localhost:${PORT}/healthz" >/dev/null 2>&1; then
    echo "[EVAL-L10-3TASK] server ready after ${i}s"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[EVAL-L10-3TASK] server died"
    exit 1
  fi
  sleep 1
done

source examples/libero/.venv/bin/activate
export PYTHONPATH=$PWD/third_party/libero:${PYTHONPATH:-}

CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python -X faulthandler examples/libero/main.py \
  --args.task-suite-name libero_10 \
  --args.task-ids 4 8 5 \
  --args.num-trials-per-task 50 \
  --args.host 0.0.0.0 \
  --args.port "${PORT}"

deactivate
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true

echo "[EVAL-L10-3TASK] End: $(date)"
