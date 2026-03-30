#!/bin/bash
# Evaluate merged task9 checkpoint on libero_10 task_id=5 (50 episodes).

#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:1
#SBATCH --exclude=gnho009
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH -J eval_t9_llm
#SBATCH -o /storage/yukaichengLab/lishiwen/jiayusun/openpi/trainlogs/eval_t9_llm_%j.log
#SBATCH -e /storage/yukaichengLab/lishiwen/jiayusun/openpi/trainlogs/eval_t9_llm_%j.err

set -euo pipefail

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi
mkdir -p trainlogs

SUITE="libero_10"
SUITE_TASK_ID=5
NUM_TRIALS=50
PORT=$((30000 + (SLURM_JOB_ID % 20000)))
CKPT_DIR="/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/merged/pi05_task9_llmbackbone_w90_base10_mmli"
CONFIG="pi05_sgo_lm90_mix50_task9"
VIDEO_OUT="data/libero/videos/task9_eval/llm_mmli_${SLURM_JOB_ID}"

export HF_HOME=~/.cache/huggingface
export HF_HUB_OFFLINE=1
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export PYOPENGL_PLATFORM=egl

echo "[EVAL-LLM-MMLI] Job: ${SLURM_JOB_ID}  Node: ${SLURMD_NODENAME}"
echo "[EVAL-LLM-MMLI] CKPT: ${CKPT_DIR}"
echo "[EVAL-LLM-MMLI] Suite: ${SUITE}  task_id: ${SUITE_TASK_ID}"
echo "[EVAL-LLM-MMLI] Start: $(date)"

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/serve_policy.py \
  --env LIBERO \
  --port "${PORT}" \
  policy:checkpoint \
  --policy.config "${CONFIG}" \
  --policy.dir "${CKPT_DIR}" &
SERVER_PID=$!

for i in $(seq 1 600); do
  if curl -s "http://localhost:${PORT}/healthz" >/dev/null 2>&1; then
    echo "[EVAL-LLM-MMLI] server ready after ${i}s"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[EVAL-LLM-MMLI] server died"
    exit 1
  fi
  sleep 1
done

source examples/libero/.venv/bin/activate
export PYTHONPATH=$PWD/third_party/libero:${PYTHONPATH:-}

python examples/libero/main.py \
  --args.task-suite-name "${SUITE}" \
  --args.task-ids "${SUITE_TASK_ID}" \
  --args.num-trials-per-task "${NUM_TRIALS}" \
  --args.video-out-path "${VIDEO_OUT}" \
  --args.host 0.0.0.0 \
  --args.port "${PORT}" || true

deactivate
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true

echo "[EVAL-LLM-MMLI] End: $(date)"
