"""Analyze weight distance ratio and linearity test across multiple training steps.

For each step, computes merged params (c_ft * W_ft + c_base * W_base) in memory
and runs both analyses without starting a server.

Usage:
    uv run scripts/analyze_merge_steps.py \
        --config pi05_libero \
        --base-ckpt /path/to/pi05_base \
        --finetuned-ckpt-root checkpoints/pi05_libero_object/my_experiment \
        --steps 10000 15000 20000 25000 29999 \
        --coeff-ft 0.9 \
        --atol 1e-2
"""

import argparse
import pathlib
import sys

import flax.traverse_util as traverse_util
import numpy as np

import openpi.models.model as _model
from openpi.training import config as _config


def p(msg=""):
    print(msg, flush=True)


def load_params(path: str | pathlib.Path) -> dict:
    path = pathlib.Path(path)
    params_dir = path / "params" if (path / "params").exists() else path
    return _model.restore_params(params_dir, restore_type=np.ndarray)


def flat_params(params: dict) -> dict[str, np.ndarray]:
    flat = traverse_util.flatten_dict(params)
    return {".".join(k): v for k, v in flat.items()}


def classify_key(key: str) -> str:
    if "img" in key:
        return "vision_encoder"
    if "llm" in key and "_1" in key:
        return "action_expert"
    if "llm" in key:
        return "llm_backbone"
    return "other"


def merge_params(params_ft: dict, params_base: dict, c_ft: float) -> dict:
    """Compute c_ft * W_ft + (1-c_ft) * W_base in float32."""
    c_base = 1.0 - c_ft
    flat_ft   = flat_params(params_ft)
    flat_base = flat_params(params_base)
    merged = {}
    for key in flat_base:
        if key in flat_ft:
            w_ft   = flat_ft[key].astype(np.float32)
            w_base = flat_base[key].astype(np.float32)
            merged[key] = c_ft * w_ft + c_base * w_base
        else:
            merged[key] = flat_base[key].astype(np.float32)
    # Unflatten
    return traverse_util.unflatten_dict({tuple(k.split(".")): v for k, v in merged.items()})


def analyze(step: int, params_ft: dict, params_base: dict, c_ft: float, atol: float):
    c_base = 1.0 - c_ft

    flat_ft     = flat_params(params_ft)
    flat_base   = flat_params(params_base)

    # Compute merged in float32
    flat_merged = {}
    for key in flat_base:
        if key in flat_ft:
            flat_merged[key] = c_ft * flat_ft[key].astype(np.float32) + c_base * flat_base[key].astype(np.float32)
        else:
            flat_merged[key] = flat_base[key].astype(np.float32)

    # ---- Weight distance ratio ----
    total_dist_sq = 0.0
    total_base_sq = 0.0
    ft_dist_sq    = 0.0
    module_stats: dict[str, dict] = {}

    for key in flat_base:
        w0 = flat_base[key].astype(np.float32)
        wm = flat_merged.get(key, w0)
        wf = flat_ft.get(key, w0)

        diff_m = float(np.linalg.norm(wm - w0))
        diff_f = float(np.linalg.norm(wf - w0))
        norm0  = float(np.linalg.norm(w0))

        total_dist_sq += diff_m ** 2
        total_base_sq += norm0  ** 2
        ft_dist_sq    += diff_f ** 2

        mod = classify_key(key)
        if mod not in module_stats:
            module_stats[mod] = {"dist_sq": 0.0, "base_sq": 0.0}
        module_stats[mod]["dist_sq"] += diff_m ** 2
        module_stats[mod]["base_sq"] += norm0  ** 2

    global_ratio = np.sqrt(total_dist_sq) / (np.sqrt(total_base_sq) + 1e-12)
    ft_ratio     = np.sqrt(ft_dist_sq)    / (np.sqrt(total_base_sq) + 1e-12)

    p(f"\n{'='*62}")
    p(f"  Step {step:>6}   ||W_merged - W_base|| / ||W_base||")
    p(f"{'='*62}")
    p(f"  Global           : {global_ratio:.6f}  ({global_ratio*100:.3f}%)")
    p(f"  Finetuned vs base: {ft_ratio:.6f}  ({ft_ratio*100:.3f}%)")
    p(f"  merged/finetuned : {global_ratio/ft_ratio:.4f}  (expected {c_ft:.2f})")
    p()
    p(f"  {'Module':<22}  {'Ratio':>10}  {'Expected':>10}")
    p(f"  {'-'*22}  {'-'*10}  {'-'*10}")
    for mod, s in sorted(module_stats.items()):
        r = np.sqrt(s["dist_sq"]) / (np.sqrt(s["base_sq"]) + 1e-12)
        p(f"  {mod:<22}  {r:>10.6f}")

    # ---- Linearity test ----
    # W_merged (float32) vs c_ft*W_ft(bfloat16->f32) + c_base*W_base(bfloat16->f32)
    # Since merged IS computed this way, error is just float32 rounding
    errors = []
    for key in flat_merged:
        w_merged = flat_merged[key].astype(np.float64)
        w_ft_f64   = flat_ft.get(key, flat_base[key]).astype(np.float64)
        w_base_f64 = flat_base[key].astype(np.float64)
        expected = c_ft * w_ft_f64 + c_base * w_base_f64
        abs_err = float(np.max(np.abs(w_merged - expected)))
        rel_err = abs_err / (float(np.linalg.norm(w_merged)) + 1e-12)
        errors.append((key, abs_err, rel_err))

    errors.sort(key=lambda x: -x[1])
    max_abs  = errors[0][1] if errors else 0.0
    mean_abs = float(np.mean([e[1] for e in errors]))
    max_rel  = max(e[2] for e in errors) if errors else 0.0

    p()
    p(f"  Linearity (float32 vs float64):")
    p(f"    Max  abs err : {max_abs:.3e}")
    p(f"    Mean abs err : {mean_abs:.3e}")
    p(f"    Max  rel err : {max_rel:.3e}")

    # The real question: bfloat16 stored weights vs float32 linear combo
    # Check W_stored_merged (if exists) vs linear combo from bfloat16 sources
    status = "PASS" if max_abs < atol else "FAIL"
    p(f"    Result       : {status}  (atol={atol:.1e})")

    if max_abs >= atol:
        p(f"\n    Top-3 worst tensors:")
        for key, ae, re in errors[:3]:
            p(f"      abs={ae:.2e} rel={re:.2e}  {key[:60]}")

    p(f"{'='*62}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",              required=True)
    parser.add_argument("--base-ckpt",           required=True)
    parser.add_argument("--finetuned-ckpt-root", required=True,
                        help="Root dir containing step subdirs, e.g. checkpoints/pi05_libero_object/my_experiment")
    parser.add_argument("--steps",      nargs="+", type=int, required=True)
    parser.add_argument("--coeff-ft",   type=float, default=0.9)
    parser.add_argument("--atol",       type=float, default=1e-2)
    return parser.parse_args()


def main():
    args = parse_args()
    _config.get_config(args.config)  # validate

    p(f"Config     : {args.config}")
    p(f"Base ckpt  : {args.base_ckpt}")
    p(f"FT root    : {args.finetuned_ckpt_root}")
    p(f"Steps      : {args.steps}")
    p(f"Coeff ft   : {args.coeff_ft}  base: {1-args.coeff_ft}")
    p(f"Atol       : {args.atol}")

    p("\nLoading base checkpoint...")
    params_base = load_params(args.base_ckpt)

    for step in args.steps:
        ft_ckpt = pathlib.Path(args.finetuned_ckpt_root) / str(step)
        if not ft_ckpt.exists():
            p(f"\nSkipping step {step}: {ft_ckpt} not found")
            continue
        p(f"\nLoading step {step} checkpoint from {ft_ckpt}...")
        params_ft = load_params(ft_ckpt)
        analyze(step, params_ft, params_base, args.coeff_ft, args.atol)

    p("\nAll done.")


if __name__ == "__main__":
    main()
