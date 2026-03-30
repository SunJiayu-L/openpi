#!/bin/bash
# Evaluate merged checkpoint on all LIBERO suites, 20 trials per task.
# GPU1 for policy, GPU0 for simulator (to avoid core dump issue).

#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=gnho031,gnho034
#SBATCH --exclude=gnho009
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH -J eval_1ft_t9_all
#SBATCH -o /storage/yukaichengLab/lishiwen/jiayusun/openpi/trainlogs/eval_1ft_t9_all_%j.log
#SBATCH -e /storage/yukaichengLab/lishiwen/jiayusun/openpi/trainlogs/eval_1ft_t9_all_%j.err

set -euo pipefail

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi
mkdir -p trainlogs

PORT_BASE=$((33000 + (SLURM_JOB_ID % 2000) * 10))
CONFIG="pi05_sgo_lm90_mix50_task9"
CKPT_DIR="/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/merged/wudi_singleft_mix50_task9_s1_i300"
NUM_TRIALS=20
SUITES=("libero_10" "libero_goal" "libero_object" "libero_spatial")

export HF_HOME=~/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export PYTHONFAULTHANDLER=1

if [ ! -d "${CKPT_DIR}" ]; then
  echo "[EVAL-1FT-T9] ERROR: checkpoint dir not found: ${CKPT_DIR}"
  exit 1
fi

run_suite() {
  local suite="$1"
  local port="$2"

  echo "[${suite}] start at $(date), port=${port}"

  CUDA_VISIBLE_DEVICES=1 \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/serve_policy.py \
    --env LIBERO \
    --port "${port}" \
    policy:checkpoint \
    --policy.config "${CONFIG}" \
    --policy.dir "${CKPT_DIR}" &
  local server_pid=$!

  local ready=0
  for _ in $(seq 1 600); do
    if curl -s "http://localhost:${port}/healthz" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! kill -0 "${server_pid}" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done

  if [ "${ready}" -ne 1 ]; then
    echo "[${suite}] ERROR: policy server not ready"
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
    return 1
  fi

  source examples/libero/.venv/bin/activate
  export PYTHONPATH=$PWD/third_party/libero:${PYTHONPATH:-}

  CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
  python -X faulthandler examples/libero/main.py \
    --args.task-suite-name "${suite}" \
    --args.num-trials-per-task "${NUM_TRIALS}" \
    --args.host 0.0.0.0 \
    --args.port "${port}"

  deactivate
  kill "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" 2>/dev/null || true

  echo "[${suite}] done at $(date)"
}

echo "[EVAL-1FT-T9] Job: ${SLURM_JOB_ID} Node: ${SLURMD_NODENAME}"
echo "[EVAL-1FT-T9] CKPT: ${CKPT_DIR}"
echo "[EVAL-1FT-T9] suites: ${SUITES[*]}"
echo "[EVAL-1FT-T9] trials/task: ${NUM_TRIALS}"

idx=0
for suite in "${SUITES[@]}"; do
  port=$((PORT_BASE + idx))
  run_suite "${suite}" "${port}"
  idx=$((idx + 1))
done

echo "[EVAL-1FT-T9] all suites finished at $(date)"
