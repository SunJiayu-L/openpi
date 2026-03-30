#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:2
#SBATCH --nodelist=gnho034
#SBATCH -J eval_sgo_lm90_all
#SBATCH -o trainlogs/eval_sgo_lm90_all.%j.log
#SBATCH -e trainlogs/eval_sgo_lm90_all.%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

set -euo pipefail

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi

export HF_HOME=~/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export PYTHONPATH=$PWD/third_party/libero:${PYTHONPATH:-}

LIBERO_VENV=examples/libero/.venv
CONFIG=pi05_libero_sgo_lm90
CKPT_DIR=/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero_RETRAIN_base/pi05_libero_RETRAIN_base/29999
NUM_TRIALS=20

if [ ! -d "$CKPT_DIR" ]; then
  echo "ERROR: checkpoint not found: $CKPT_DIR"
  exit 1
fi

run_suite() {
  local suite="$1"
  local gpu="$2"
  local port="$3"
  local video_out="data/libero/videos/pi05_sgo_lm90_step29999/${suite}"

  echo "[${suite}] launch server on GPU ${gpu}, port ${port}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  MUJOCO_EGL_DEVICE_ID=0 \
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
    echo "[${suite}] ERROR: server not ready"
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
    return 1
  fi

  source "${LIBERO_VENV}/bin/activate"
  MUJOCO_EGL_DEVICE_ID=0 \
  python examples/libero/main.py \
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

# Round 1: two suites in parallel on two GPUs.
run_suite libero_10 0 8610 &
pid_a=$!
run_suite libero_goal 1 8611 &
pid_b=$!
wait "$pid_a"
wait "$pid_b"

# Round 2: two suites in parallel on two GPUs.
run_suite libero_object 0 8612 &
pid_c=$!
run_suite libero_spatial 1 8613 &
pid_d=$!
wait "$pid_c"
wait "$pid_d"

echo "All suites finished at $(date)"
