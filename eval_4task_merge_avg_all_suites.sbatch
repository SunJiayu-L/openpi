#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:2
#SBATCH --nodelist=gnho034
#SBATCH -J eval_4task_merge_avg
#SBATCH -o trainlogs/eval_4task_merge_avg_%j.log
#SBATCH -e trainlogs/eval_4task_merge_avg_%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

set -euo pipefail
export PYTHONFAULTHANDLER=1

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi
mkdir -p trainlogs

export HF_HOME=~/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export PYTHONPATH=$PWD/third_party/libero:${PYTHONPATH:-}

LIBERO_VENV=examples/libero/.venv
CONFIG=pi05_libero
CKPT_DIR=/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero_4task_merge/0
NUM_TRIALS=20

if [ ! -d "$CKPT_DIR" ]; then
  echo "ERROR: checkpoint not found: $CKPT_DIR"
  exit 1
fi

# Copy norm_stats.json to the location serve_policy.py expects
ASSETS_DIR="${CKPT_DIR}/assets/physical-intelligence/libero"
mkdir -p "${ASSETS_DIR}"
if [ ! -f "${ASSETS_DIR}/norm_stats.json" ]; then
  cp /storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero_4task_merge/norm_stats.json \
     "${ASSETS_DIR}/norm_stats.json"
  echo "Copied norm_stats.json to ${ASSETS_DIR}/"
fi

# GPU1: policy server (JAX, no EGL)
# GPU0: simulator (MuJoCo EGL)
run_suite() {
  local suite="$1"
  local port="$2"
  local video_out="data/libero/videos/pi05_4task_merge_avg/${suite}"

  echo "[${suite}] launching policy server on GPU1, port ${port}"
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
      echo "[${suite}] ERROR: server process died"
      break
    fi
    sleep 1
  done

  if [ "${ready}" -ne 1 ]; then
    echo "[${suite}] ERROR: server not ready after 600s"
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
    return 1
  fi

  echo "[${suite}] server ready, starting eval on GPU0"
  source "${LIBERO_VENV}/bin/activate"
  CUDA_VISIBLE_DEVICES=0 \
  MUJOCO_EGL_DEVICE_ID=0 \
  python -X faulthandler examples/libero/main.py \
    --args.task-suite-name "${suite}" \
    --args.num-trials-per-task "${NUM_TRIALS}" \
    --args.video-out-path "${video_out}" \
    --args.host 0.0.0.0 \
    --args.port "${port}"
  deactivate

  kill "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" 2>/dev/null || true
  echo "[${suite}] done at $(date)"
}

run_suite libero_spatial 8620
run_suite libero_goal    8621
run_suite libero_object  8622
run_suite libero_10      8623

echo "All suites finished at $(date)"
