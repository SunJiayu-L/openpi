"""Analyze merged model weights.

Two analyses:

1. Weight distance ratio: ||W_merged - W_base|| / ||W_base||
   Computed globally and broken down by module (vision / llm / action_expert).

2. Linearity test (param-level): verify that the merged params satisfy
   W_merged = sum(c_i * W_i) within float32 tolerance.

Usage:
    # Weight distance ratio only:
    uv run scripts/analyze_merge.py \
        --config pi05_libero \
        --base-ckpt /path/to/pi05_base \
        --merged-ckpt checkpoints/merged/pi05_obj09_base01

    # With linearity test:
    uv run scripts/analyze_merge.py \
        --config pi05_libero \
        --base-ckpt /path/to/pi05_base \
        --merged-ckpt checkpoints/merged/pi05_obj09_base01 \
        --finetuned-ckpts checkpoints/pi05_libero_object/my_experiment/29999 \
        --coefficients 0.9 0.1 \
        --test-linearity
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


def load_params(ckpt_dir: str | pathlib.Path) -> dict:
    ckpt_dir = pathlib.Path(ckpt_dir)
    params_path = ckpt_dir / "params"
    if not params_path.exists():
        params_path = ckpt_dir
    p(f"  Loading from: {params_path}")
    return _model.restore_params(params_path, restore_type=np.ndarray)


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


# ---------------------------------------------------------------------------
# Analysis 1: Weight distance ratio
# ---------------------------------------------------------------------------

def weight_distance_ratio(params_merged, params_base, params_ref=None):
    flat_merged = flat_params(params_merged)
    flat_base   = flat_params(params_base)

    total_dist_sq = 0.0
    total_base_sq = 0.0
    module_stats: dict[str, dict] = {}

    for key in flat_base:
        if key not in flat_merged:
            continue
        w0 = flat_base[key].astype(np.float32)
        w  = flat_merged[key].astype(np.float32)

        diff_norm = float(np.linalg.norm(w - w0))
        base_norm = float(np.linalg.norm(w0))

        total_dist_sq += diff_norm ** 2
        total_base_sq += base_norm ** 2

        mod = classify_key(key)
        if mod not in module_stats:
            module_stats[mod] = {"dist_sq": 0.0, "base_sq": 0.0, "count": 0}
        module_stats[mod]["dist_sq"] += diff_norm ** 2
        module_stats[mod]["base_sq"] += base_norm ** 2
        module_stats[mod]["count"]   += 1

    global_ratio = np.sqrt(total_dist_sq) / (np.sqrt(total_base_sq) + 1e-12)

    p()
    p("=" * 62)
    p("WEIGHT DISTANCE RATIO   ||W_merged - W_base|| / ||W_base||")
    p("=" * 62)
    p(f"  Global:  {global_ratio:.6f}  ({global_ratio*100:.3f}%)")
    p()
    p(f"  {'Module':<22}  {'Ratio':>10}  {'Tensors':>10}")
    p(f"  {'-'*22}  {'-'*10}  {'-'*10}")
    for mod, s in sorted(module_stats.items()):
        r = np.sqrt(s["dist_sq"]) / (np.sqrt(s["base_sq"]) + 1e-12)
        p(f"  {mod:<22}  {r:>10.6f}  {s['count']:>10,}")
    p("=" * 62)

    if params_ref is not None:
        flat_ref = flat_params(params_ref)
        ft_dist_sq = sum(
            float(np.linalg.norm(
                flat_ref[k].astype(np.float32) - flat_base[k].astype(np.float32)
            )) ** 2
            for k in flat_base if k in flat_ref
        )
        ft_ratio = np.sqrt(ft_dist_sq) / (np.sqrt(total_base_sq) + 1e-12)
        p()
        p(f"  Finetuned vs base : {ft_ratio:.6f}  ({ft_ratio*100:.3f}%)")
        p(f"  Merged    vs base : {global_ratio:.6f}  ({global_ratio*100:.3f}%)")
        p(f"  merged/finetuned  : {global_ratio/ft_ratio:.4f}  (expected ≈ coeff of finetuned model)")


# ---------------------------------------------------------------------------
# Analysis 2: Param-level linearity test
# ---------------------------------------------------------------------------

def linearity_test(params_merged, params_list, coefficients, atol=1e-4):
    """Check W_merged == sum(c_i * W_i) within tolerance."""
    flat_all    = [flat_params(p) for p in params_list]
    flat_merged = flat_params(params_merged)

    p()
    p("=" * 62)
    p("LINEARITY TEST   W_merged = Σ c_i · W_i  (param level)")
    p("=" * 62)
    p(f"  Coefficients: {coefficients}")
    p(f"  Tolerance   : {atol:.1e}")

    errors = []
    for key in flat_merged:
        expected = sum(
            c * flat_all[i][key].astype(np.float64)
            for i, c in enumerate(coefficients)
            if key in flat_all[i]
        )
        actual = flat_merged[key].astype(np.float64)
        abs_err = float(np.max(np.abs(actual - expected)))
        rel_err = abs_err / (float(np.linalg.norm(actual)) + 1e-12)
        errors.append((key, abs_err, rel_err))

    errors.sort(key=lambda x: -x[1])
    max_abs  = errors[0][1] if errors else 0.0
    mean_abs = float(np.mean([e[1] for e in errors]))
    max_rel  = max(e[2] for e in errors) if errors else 0.0

    p()
    p(f"  Max  absolute error : {max_abs:.3e}")
    p(f"  Mean absolute error : {mean_abs:.3e}")
    p(f"  Max  relative error : {max_rel:.3e}")

    if max_abs < atol:
        p(f"\n  PASS  ✓  (max err {max_abs:.2e} < tol {atol:.1e})")
    else:
        p(f"\n  FAIL  ✗  (max err {max_abs:.2e} > tol {atol:.1e})")
        p("\n  Top-5 worst tensors:")
        for key, ae, re in errors[:5]:
            p(f"    abs={ae:.2e}  rel={re:.2e}  {key[:70]}")

    p("=" * 62)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",         required=True)
    parser.add_argument("--base-ckpt",      required=True)
    parser.add_argument("--merged-ckpt",    required=True)
    parser.add_argument("--finetuned-ckpts", nargs="*", default=[])
    parser.add_argument("--coefficients",    nargs="*", type=float, default=[])
    parser.add_argument("--test-linearity",  action="store_true")
    parser.add_argument("--atol",            type=float, default=1e-4)
    return parser.parse_args()


def main():
    args = parse_args()
    _config.get_config(args.config)  # validate config name early

    p("Loading base checkpoint...")
    params_base = load_params(args.base_ckpt)

    p("Loading merged checkpoint...")
    params_merged = load_params(args.merged_ckpt)

    params_ref = None
    if args.finetuned_ckpts:
        p("Loading finetuned checkpoint(s)...")
        params_ref = load_params(args.finetuned_ckpts[0])

    weight_distance_ratio(params_merged, params_base, params_ref)

    if args.test_linearity:
        if not args.finetuned_ckpts or not args.coefficients:
            p("ERROR: --test-linearity requires --finetuned-ckpts and --coefficients")
            sys.exit(1)
        all_params = [load_params(c) for c in args.finetuned_ckpts] + [load_params(args.base_ckpt)]
        linearity_test(params_merged, all_params, args.coefficients, atol=args.atol)


if __name__ == "__main__":
    main()
