#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:0
#SBATCH --nodelist=gnho031
#SBATCH -J convert_rl_ckpts
#SBATCH -o /storage/yukaichengLab/lishiwen/jiayusun/openpi/trainlogs/convert_rl_ckpts.%j.log
#SBATCH -e /storage/yukaichengLab/lishiwen/jiayusun/openpi/trainlogs/convert_rl_ckpts.%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

# Convert RL PPO checkpoints (full_weights.pt) to model.safetensors
# Strips value_head.* keys so openpi serve_policy.py can load them
# Steps: 80, 100, 120

set -eo pipefail
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"

CKPT_BASE="/storage/yukaichengLab/lishiwen/jiayusun/RLinf/logs/20260503-15:42:22-libero_10_ppo_pi05_h800_4gpu/libero_10_ppo_pi05_from_base25k/checkpoints"
OUT_BASE="/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pt_rl"
NORM_STATS_SRC="/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pt_converted/base_25k/physical-intelligence/libero/norm_stats.json"

mkdir -p "$OUT_BASE"

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi

convert_one() {
    local step=$1
    local src="${CKPT_BASE}/global_step_${step}/actor/model_state_dict/full_weights.pt"
    local out_dir="${OUT_BASE}/ppo_step${step}"
    mkdir -p "${out_dir}/physical-intelligence/libero"

    echo "=== Converting step ${step} | $(date) ==="
    uv run python - <<PYEOF
import torch
import safetensors.torch

src = "${src}"
out = "${out_dir}/model.safetensors"

print(f"Loading {src} ...")
state_dict = torch.load(src, map_location="cpu", weights_only=False)

# Strip value head keys and tied embed_tokens duplicate (not in PI0Pytorch)
EXCLUDE = {"paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"}
filtered = {k: v for k, v in state_dict.items()
            if not k.startswith("value_head.") and k not in EXCLUDE}
removed = len(state_dict) - len(filtered)
print(f"Total keys: {len(state_dict)}, removed: {removed}, kept: {len(filtered)}")

print(f"Saving to {out} ...")
safetensors.torch.save_file(filtered, out)
print("Done.")
PYEOF

    # serve_policy.py 找 assets/ 子目录；RLinf 找根目录，两处都放
    mkdir -p "${out_dir}/assets/physical-intelligence/libero"
    cp "$NORM_STATS_SRC" "${out_dir}/assets/physical-intelligence/libero/norm_stats.json"
    cp "$NORM_STATS_SRC" "${out_dir}/physical-intelligence/libero/norm_stats.json"
    echo "Saved: ${out_dir}"
}

convert_one 80
convert_one 100
convert_one 120

echo "All conversions done: $(date)"
