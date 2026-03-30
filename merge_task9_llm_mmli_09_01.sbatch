#!/bin/bash
# Use RETAIN_code multimodal_linear_interpolation:
# - vision: keep A (1.0, 0.0)
# - llm backbone: merge A/B (0.9, 0.1)
# - action expert + aux action heads: keep A (1.0, 0.0)
#
# A: finetuned task9 checkpoint
# B: base checkpoint (10k)

#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:1
#SBATCH --exclude=gnho009
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH -J merge_t9_llm
#SBATCH -o /storage/yukaichengLab/lishiwen/jiayusun/openpi/trainlogs/merge_t9_llm_%j.log
#SBATCH -e /storage/yukaichengLab/lishiwen/jiayusun/openpi/trainlogs/merge_t9_llm_%j.err

set -euo pipefail

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi
mkdir -p trainlogs

A="/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_sgo_lm90_mix50_task9/my_experiment/29999"
B="/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero_RETRAIN_base/pi05_libero_RETRAIN_base/10000"
OUT="/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/merged/pi05_task9_llmbackbone_w90_base10_mmli"

export PYTHONPATH="/storage/yukaichengLab/lishiwen/jiayusun/RETAIN_code/src:${PYTHONPATH:-}"
export JAX_PLATFORMS=cpu

echo "[MERGE-LLM-MMLI] Job: ${SLURM_JOB_ID} Node: ${SLURMD_NODENAME}"
echo "[MERGE-LLM-MMLI] A: ${A}"
echo "[MERGE-LLM-MMLI] B: ${B}"
echo "[MERGE-LLM-MMLI] OUT: ${OUT}"
echo "[MERGE-LLM-MMLI] Start: $(date)"

rm -rf "${OUT}"
mkdir -p "${OUT}"

/storage/yukaichengLab/lishiwen/jiayusun/openpi/.venv/bin/python - <<'PY'
import pathlib
from types import SimpleNamespace
import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import orbax.checkpoint as ocp

from openpi.models.pi0 import Pi0Config
import openpi.models.model as _model
from openpi.policies.model_merging import multimodal_linear_interpolation

A = pathlib.Path("/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_sgo_lm90_mix50_task9/my_experiment/29999")
B = pathlib.Path("/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero_RETRAIN_base/pi05_libero_RETRAIN_base/10000")
OUT = pathlib.Path("/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/merged/pi05_task9_llmbackbone_w90_base10_mmli/params")

train_config = SimpleNamespace(model=Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False))
model = multimodal_linear_interpolation(
    train_config,
    [A, B],
    {
        "vision_mixing_coefficients": [1.0, 0.0],
        "llm_mixing_coefficients": [0.9, 0.1],
        "action_expert_mixing_coefficients": [1.0, 0.0],
    },
)

state = nnx.state(model)
pure = state.to_pure_dict()
params = pure["params"] if "params" in pure else pure

# multimodal_linear_interpolation may leave some leaves abstract (ShapeDtypeStruct).
# Fill those leaves from checkpoint A so we can materialize a full checkpoint.
params_a = _model.restore_params(A / "params")
flat_m = traverse_util.flatten_dict(params)
flat_a = traverse_util.flatten_dict(params_a)
for k, v in list(flat_m.items()):
    if isinstance(v, jax.ShapeDtypeStruct):
        flat_m[k] = flat_a[k]
params = traverse_util.unflatten_dict(flat_m)

ocp.PyTreeCheckpointer().save(OUT, {"params": params})
print(f"saved merged params to: {OUT}")
PY

cp -r "${A}/assets" "${OUT}/assets"

echo "[MERGE-LLM-MMLI] End: $(date)"
