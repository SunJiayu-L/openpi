#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -J merge_t22w02_t24w08_vl
#SBATCH -o /storage/yukaichengLab/lishiwen/jiayusun/openpi/trainlogs/merge_t22w02_t24w08_vl_%j.log
#SBATCH -e /storage/yukaichengLab/lishiwen/jiayusun/openpi/trainlogs/merge_t22w02_t24w08_vl_%j.err

set -euo pipefail

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi
mkdir -p trainlogs

T22_STEP="/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero_obj_task22_ep45_from_libero10_5000_retrain_15/t22_from_libero10_5000_15steps/15"
T24_STEP="/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero_obj_task24_ep44_from_libero10_5000_retrain_15/t24_from_libero10_5000_15steps/15"
OUT_STEP="/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/merged/t22w02_t24w08_vl_only/0"

export JAX_PLATFORMS=cpu
export UV_CACHE_DIR=/tmp/${USER}/uv-cache
mkdir -p "${UV_CACHE_DIR}"

echo "[MERGE] Job: ${SLURM_JOB_ID}  Node: ${SLURMD_NODENAME}"
echo "[MERGE] T22(w=0.2): ${T22_STEP}"
echo "[MERGE] T24(w=0.8): ${T24_STEP}"
echo "[MERGE] OUT: ${OUT_STEP}"
echo "[MERGE] Start: $(date)"

rm -rf "$(dirname ${OUT_STEP})"
mkdir -p "${OUT_STEP}"

.venv/bin/python - <<'PY'
import pathlib
import numpy as np
import flax.traverse_util as traverse_util
import orbax.checkpoint as ocp
from openpi.models.model import restore_params

T22 = pathlib.Path('/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero_obj_task22_ep45_from_libero10_5000_retrain_15/t22_from_libero10_5000_15steps/15/params')
T24 = pathlib.Path('/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero_obj_task24_ep44_from_libero10_5000_retrain_15/t24_from_libero10_5000_15steps/15/params')
OUT = pathlib.Path('/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/merged/t22w02_t24w08_vl_only/0/params')

W22 = 0.2
W24 = 0.8

# Action keys: keep from T24 (不融合，直接取 T24)
ACTION_KEYS = {'action_in_proj', 'action_out_proj', 'time_mlp_in', 'time_mlp_out'}

print(f'Loading T22 params ...')
pa = restore_params(T22, restore_type=np.ndarray)
print(f'Loading T24 params ...')
pb = restore_params(T24, restore_type=np.ndarray)

fa = traverse_util.flatten_dict(pa)  # keys are tuples, e.g. ('PaliGemma', 'llm', ...)
fb = traverse_util.flatten_dict(pb)

if set(fa.keys()) != set(fb.keys()):
    only_a = sorted(set(fa.keys()) - set(fb.keys()))[:5]
    only_b = sorted(set(fb.keys()) - set(fa.keys()))[:5]
    raise RuntimeError(f'Key mismatch! only_in_t22={only_a}, only_in_t24={only_b}')

merged = {}
vl_count = 0
action_count = 0

for k in fa:
    if k[0] in ACTION_KEYS:
        # 不融合动作头，直接取 T24
        merged[k] = fb[k]
        action_count += 1
    else:
        # 线性插值视觉语言部分
        a = fa[k].astype(np.float32)
        b = fb[k].astype(np.float32)
        m = W22 * a + W24 * b
        merged[k] = m.astype(fa[k].dtype, copy=False)
        vl_count += 1

print(f'VL merged (0.2*T22 + 0.8*T24): {vl_count} tensors')
print(f'Action frozen (T24 only):       {action_count} tensors')
print(f'Total:                          {len(merged)} tensors')

nested = traverse_util.unflatten_dict(merged)
ckptr = ocp.PyTreeCheckpointer()
ckptr.save(OUT, {'params': nested})
print(f'Saved merged params to: {OUT}')
PY

# norm stats: 取 T24 的 assets（与动作头保持一致）
cp -r "${T24_STEP}/assets" "${OUT_STEP}/assets"
echo "[MERGE] Copied norm stats from T24 -> ${OUT_STEP}/assets"

echo "[MERGE] End: $(date)"
echo "[MERGE] Done: $(dirname ${OUT_STEP})"
