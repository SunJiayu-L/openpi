#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=gnho031,gnho034
#SBATCH --exclude=gnho009
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH -J eval_mix50_t9_all
#SBATCH -o /storage/yukaichengLab/lishiwen/jiayusun/openpi/trainlogs/eval_mix50_t9_all_%j.log
#SBATCH -e /storage/yukaichengLab/lishiwen/jiayusun/openpi/trainlogs/eval_mix50_t9_all_%j.err

set -euo pipefail

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi
mkdir -p trainlogs

PORT_BASE=$((32000 + (SLURM_JOB_ID % 2000) * 10))
CONFIG="pi05_sgo_lm90_mix50_task9"
CKPT_DIR="/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_sgo_lm90_mix50_task9/my_experiment/29999"
NUM_TRIALS=20
SUITES=("libero_10" "libero_goal" "libero_object" "libero_spatial")

export HF_HOME=~/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export PYTHONFAULTHANDLER=1

if [ ! -d "$CKPT_DIR" ]; then
  echo "ERROR: checkpoint not found: $CKPT_DIR"
  exit 1
fi

run_suite() {
  local suite="$1"
  local port="$2"

  echo "[$suite] start at $(date), port=${port}"

  CUDA_VISIBLE_DEVICES=3 \
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
    echo "[$suite] ERROR: policy server not ready"
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
    return 1
  fi

  source examples/libero/.venv/bin/activate
  export PYTHONPATH=$PWD/third_party/libero:${PYTHONPATH:-}

  CUDA_VISIBLE_DEVICES=2 MUJOCO_EGL_DEVICE_ID=2 \
  python -X faulthandler examples/libero/main.py \
    --args.task-suite-name "${suite}" \
    --args.num-trials-per-task "${NUM_TRIALS}" \
    --args.host 0.0.0.0 \
    --args.port "${port}"

  deactivate
  kill "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" 2>/dev/null || true

  echo "[$suite] done at $(date)"
}

echo "[EVAL-MIX50-T9-ALL] Job: ${SLURM_JOB_ID} Node: ${SLURMD_NODENAME}"
echo "[EVAL-MIX50-T9-ALL] CKPT: ${CKPT_DIR}"
echo "[EVAL-MIX50-T9-ALL] suites: ${SUITES[*]}"
echo "[EVAL-MIX50-T9-ALL] trials/task: ${NUM_TRIALS}"

idx=0
for suite in "${SUITES[@]}"; do
  port=$((PORT_BASE + idx))
  run_suite "${suite}" "${port}"
  idx=$((idx + 1))
done

echo "[EVAL-MIX50-T9-ALL] all suites finished at $(date)"
