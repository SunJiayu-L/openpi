#!/usr/bin/env python3
"""Layerwise FFN retain-rate diagnostics for from10k checkpoints.

Focus modules:
  - gating_einsum
  - linear

Metrics per (module, layer, task):
  Delta_t = W_t - W_base
  Delta_m2 = (Delta_10 + Delta_spatial) / 2
  Delta_m4 = (Delta_10 + Delta_spatial + Delta_object + Delta_goal) / 4

  r_t(Delta_m)        = <Delta_m, Delta_t> / ||Delta_t||^2
  cos(Delta_m,Delta_t)= <Delta_m, Delta_t> / (||Delta_m|| * ||Delta_t||)
  norm_ratio          = ||Delta_m|| / ||Delta_t||
  drop_r              = r_in_4task_mean - r_in_2task_mean
"""

from __future__ import annotations

import argparse
import csv
import logging
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import Dict

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from openpi.models.model import restore_params

logger = logging.getLogger(__name__)


TASKS = ("libero_10", "libero_spatial", "libero_object", "libero_goal")
MODULES = ("gating_einsum", "linear")
NUM_LAYERS = 18

_FROZEN_PREFIXES = (
    "PaliGemma/llm/embedder/",
    "action_in_proj/",
    "action_out_proj/",
    "time_mlp_in/",
    "time_mlp_out/",
)
_RE_GATE = re.compile(r".*/gating_einsum(_\d+)?$")
_RE_LINEAR = re.compile(r".*/linear(_\d+)?$")


def _is_frozen(key: str) -> bool:
    return any(key.startswith(p) for p in _FROZEN_PREFIXES)


def _is_vision(key: str) -> bool:
    return key.startswith("PaliGemma/img/")


def _is_expert1(key: str) -> bool:
    if not key.startswith("PaliGemma/llm/"):
        return False
    segs = key.split("/")
    expert_named = re.compile(
        r"^(attn|mlp|pre_attention_norm|pre_ffw_norm|final_norm|"
        r"q_einsum|kv_einsum|attn_vec_einsum|gating_einsum|linear)_(\d+)$"
    )
    for seg in segs:
        m = expert_named.fullmatch(seg)
        if m and int(m.group(2)) >= 1:
            return True
    return False


def _in_scope(key: str, scope: str) -> bool:
    if _is_frozen(key):
        return False
    if scope == "expert1_only":
        return _is_expert1(key)
    if scope == "both_experts":
        return True
    if scope == "llm_only":
        return not _is_vision(key)
    if scope == "lang_and_vision":
        return not _is_expert1(key)
    raise ValueError(f"Unknown scope: {scope!r}")


def _ffn_module_of_key(key: str) -> str | None:
    if _RE_GATE.fullmatch(key):
        return "gating_einsum"
    if _RE_LINEAR.fullmatch(key):
        return "linear"
    return None


def _region_of_layer(layer: int) -> str:
    if 0 <= layer <= 5:
        return "front"
    if 6 <= layer <= 11:
        return "middle"
    return "back"


def _safe_div(a: float, b: float) -> float:
    if b == 0.0:
        return float("nan")
    return a / b


def _load_flat(params_path: str | pathlib.Path) -> dict[str, np.ndarray]:
    import flax.traverse_util as traverse_util

    path = pathlib.Path(params_path).resolve()
    params_dir = path / "params" if (path / "params" / "_METADATA").exists() else path
    logger.info("Loading: %s", params_dir)
    params = restore_params(params_dir, restore_type=np.ndarray)
    flat = traverse_util.flatten_dict(params)
    return {"/".join(k): v for k, v in flat.items()}


def _validate_keys(base: dict[str, np.ndarray], models: Dict[str, dict[str, np.ndarray]]) -> None:
    base_keys = set(base.keys())
    for name, m in models.items():
        keys = set(m.keys())
        if keys != base_keys:
            miss = sorted(base_keys - keys)
            extra = sorted(keys - base_keys)
            raise ValueError(
                f"Key mismatch for {name}: missing={len(miss)}, extra={len(extra)}; "
                f"first_missing={miss[:3]}, first_extra={extra[:3]}"
            )


@dataclass
class PairAcc:
    dot2: float = 0.0
    dot4: float = 0.0
    norm_t: float = 0.0
    norm_m2: float = 0.0
    norm_m4: float = 0.0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Layerwise FFN retain-rate diagnostics")
    ap.add_argument("--base", type=str, default="checkpoints/pi05_libero/my_experiment/10000")
    ap.add_argument(
        "--ft-libero-10",
        type=str,
        default=(
            "checkpoints/from10k/pi05_libero_10_from_pi05libero_10k/"
            "ft_from_pi05libero_10k/20000"
        ),
    )
    ap.add_argument(
        "--ft-libero-spatial",
        type=str,
        default=(
            "checkpoints/from10k/pi05_libero_spatial_from_pi05libero_10k/"
            "ft_from_pi05libero_10k/20000"
        ),
    )
    ap.add_argument(
        "--ft-libero-goal",
        type=str,
        default=(
            "checkpoints/from10k/pi05_libero_goal_from_pi05libero_10k/"
            "ft_from_pi05libero_10k/20000"
        ),
    )
    ap.add_argument(
        "--ft-libero-object",
        type=str,
        default=(
            "checkpoints/from10k/pi05_libero_object_from_pi05libero_10k/"
            "ft_from_pi05libero_10k/20000"
        ),
    )
    ap.add_argument(
        "--scope",
        type=str,
        default="llm_only",
        choices=["expert1_only", "both_experts", "llm_only", "lang_and_vision"],
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default="docs/analysis/from10k_llm_only_ffn_layerwise_retain_rate",
    )
    ap.add_argument("--topk", type=int, default=3, help="top-k layers per module by most negative drop_r10")
    ap.add_argument("--log-level", type=str, default="INFO")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s %(message)s")

    base = _load_flat(args.base)
    models: Dict[str, dict[str, np.ndarray]] = {
        "libero_10": _load_flat(args.ft_libero_10),
        "libero_spatial": _load_flat(args.ft_libero_spatial),
        "libero_object": _load_flat(args.ft_libero_object),
        "libero_goal": _load_flat(args.ft_libero_goal),
    }
    _validate_keys(base, models)

    # Accumulators per (module, layer, task)
    accs: dict[tuple[str, int, str], PairAcc] = {}
    for m in MODULES:
        for l in range(NUM_LAYERS):
            for t in TASKS:
                accs[(m, l, t)] = PairAcc()

    matched_keys = 0
    per_module_key_count = {m: 0 for m in MODULES}
    for key in sorted(base.keys()):
        if not _in_scope(key, args.scope):
            continue
        mod = _ffn_module_of_key(key)
        if mod is None:
            continue

        b = np.asarray(base[key], dtype=np.float64)
        if b.ndim < 1 or b.shape[0] != NUM_LAYERS:
            continue

        matched_keys += 1
        per_module_key_count[mod] += 1

        tvec = {
            t: (np.asarray(models[t][key], dtype=np.float64) - b)
            for t in TASKS
        }
        m2 = 0.5 * (tvec["libero_10"] + tvec["libero_spatial"])
        m4 = 0.25 * (tvec["libero_10"] + tvec["libero_spatial"] + tvec["libero_object"] + tvec["libero_goal"])

        for l in range(NUM_LAYERS):
            m2l = m2[l].ravel()
            m4l = m4[l].ravel()
            for t in TASKS:
                tl = tvec[t][l].ravel()
                a = accs[(mod, l, t)]
                a.dot2 += float(np.dot(m2l, tl))
                a.dot4 += float(np.dot(m4l, tl))
                a.norm_t += float(np.dot(tl, tl))
                a.norm_m2 += float(np.dot(m2l, m2l))
                a.norm_m4 += float(np.dot(m4l, m4l))

    logger.info("Scope=%s, matched FFN keys=%d", args.scope, matched_keys)
    for m in MODULES:
        logger.info("Module %-16s matched keys=%d", m, per_module_key_count[m])

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) layerwise full table
    rows: list[dict[str, str]] = []
    for m in MODULES:
        for l in range(NUM_LAYERS):
            for t in TASKS:
                a = accs[(m, l, t)]
                r2 = _safe_div(a.dot2, a.norm_t)
                r4 = _safe_div(a.dot4, a.norm_t)
                drop = r4 - r2
                c2 = _safe_div(a.dot2, np.sqrt(a.norm_m2) * np.sqrt(a.norm_t))
                c4 = _safe_div(a.dot4, np.sqrt(a.norm_m4) * np.sqrt(a.norm_t))
                nr2 = _safe_div(np.sqrt(a.norm_m2), np.sqrt(a.norm_t))
                nr4 = _safe_div(np.sqrt(a.norm_m4), np.sqrt(a.norm_t))
                rows.append(
                    {
                        "module": m,
                        "layer": str(l),
                        "task": t,
                        "r_in_2task_mean": f"{r2:.12f}",
                        "r_in_4task_mean": f"{r4:.12f}",
                        "drop_r": f"{drop:.12f}",
                        "cos_with_2task_mean": f"{c2:.12f}",
                        "cos_with_4task_mean": f"{c4:.12f}",
                        "norm_ratio_2task_mean": f"{nr2:.12f}",
                        "norm_ratio_4task_mean": f"{nr4:.12f}",
                    }
                )
    with (out_dir / "layerwise_retain_rate_ffn.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "module",
            "layer",
            "task",
            "r_in_2task_mean",
            "r_in_4task_mean",
            "drop_r",
            "cos_with_2task_mean",
            "cos_with_4task_mean",
            "norm_ratio_2task_mean",
            "norm_ratio_4task_mean",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # 2) r10 drop summary by layer + ranking
    r10_rows = []
    for m in MODULES:
        module_rows = []
        for l in range(NUM_LAYERS):
            a = accs[(m, l, "libero_10")]
            r2 = _safe_div(a.dot2, a.norm_t)
            r4 = _safe_div(a.dot4, a.norm_t)
            module_rows.append(
                {
                    "module": m,
                    "layer": l,
                    "r10_2task": r2,
                    "r10_4task": r4,
                    "drop_r10": r4 - r2,
                }
            )
        module_rows_sorted = sorted(module_rows, key=lambda x: x["drop_r10"])
        for rank, rr in enumerate(module_rows_sorted, start=1):
            r10_rows.append(
                {
                    "module": rr["module"],
                    "layer": str(rr["layer"]),
                    "r10_2task": f"{rr['r10_2task']:.12f}",
                    "r10_4task": f"{rr['r10_4task']:.12f}",
                    "drop_r10": f"{rr['drop_r10']:.12f}",
                    "rank_by_drop": str(rank),
                }
            )
    with (out_dir / "layerwise_drop_summary_r10.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["module", "layer", "r10_2task", "r10_4task", "drop_r10", "rank_by_drop"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(r10_rows)

    # 3) region summary for r10
    region_rows = []
    for m in MODULES:
        for region in ("front", "middle", "back"):
            vals = []
            for l in range(NUM_LAYERS):
                if _region_of_layer(l) != region:
                    continue
                a = accs[(m, l, "libero_10")]
                r2 = _safe_div(a.dot2, a.norm_t)
                r4 = _safe_div(a.dot4, a.norm_t)
                vals.append(r4 - r2)
            arr = np.asarray(vals, dtype=np.float64)
            region_rows.append(
                {
                    "module": m,
                    "region": region,
                    "mean_drop_r10": f"{float(np.mean(arr)):.12f}",
                    "min_drop_r10": f"{float(np.min(arr)):.12f}",
                    "max_drop_r10": f"{float(np.max(arr)):.12f}",
                    "num_layers": str(arr.size),
                }
            )
    with (out_dir / "region_summary_r10.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["module", "region", "mean_drop_r10", "min_drop_r10", "max_drop_r10", "num_layers"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(region_rows)

    # 4) candidate layers to protect: top-k most negative drop per module
    candidate_rows = []
    for m in MODULES:
        m_rows = [r for r in r10_rows if r["module"] == m]
        m_rows = sorted(m_rows, key=lambda x: float(x["drop_r10"]))
        for rr in m_rows[: args.topk]:
            l = int(rr["layer"])
            candidate_rows.append(
                {
                    "module": m,
                    "layer": rr["layer"],
                    "drop_r10": rr["drop_r10"],
                    "region": _region_of_layer(l),
                }
            )
    with (out_dir / "candidate_protected_layers.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["module", "layer", "drop_r10", "region"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(candidate_rows)

    print("\nSaved outputs:")
    print(f"  {out_dir / 'layerwise_retain_rate_ffn.csv'}")
    print(f"  {out_dir / 'layerwise_drop_summary_r10.csv'}")
    print(f"  {out_dir / 'region_summary_r10.csv'}")
    print(f"  {out_dir / 'candidate_protected_layers.csv'}")


if __name__ == "__main__":
    main()

