#!/usr/bin/env python3
"""Analyze repair map: Delta_repair = W_joint - W_mean4_iter0.

Outputs:
  - repair_module_norm.csv
  - repair_ffn_layer_norm.csv
  - repair_vs_drop_correlation.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import pathlib
import re
import sys
from typing import Dict

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from openpi.models.model import restore_params

logger = logging.getLogger(__name__)

MODULES = ("q_einsum", "kv_einsum", "attn_vec_einsum", "gating_einsum", "linear")
NUM_LAYERS = 18

_FROZEN_PREFIXES = (
    "PaliGemma/llm/embedder/",
    "action_in_proj/",
    "action_out_proj/",
    "time_mlp_in/",
    "time_mlp_out/",
)
_RE_Q = re.compile(r".*/q_einsum(_\d+)?/w$")
_RE_KV = re.compile(r".*/kv_einsum(_\d+)?/w$")
_RE_AV = re.compile(r".*/attn_vec_einsum(_\d+)?/w$")
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


def _module_of_key(key: str) -> str | None:
    if _RE_KV.fullmatch(key):
        return "kv_einsum"
    if _RE_Q.fullmatch(key):
        return "q_einsum"
    if _RE_AV.fullmatch(key):
        return "attn_vec_einsum"
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


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    x0 = x - x.mean()
    y0 = y - y.mean()
    denom = np.sqrt(np.dot(x0, x0) * np.dot(y0, y0))
    if denom == 0:
        return float("nan")
    return float(np.dot(x0, y0) / denom)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson(_rankdata(x), _rankdata(y))


def _load_flat(params_path: str | pathlib.Path) -> dict[str, np.ndarray]:
    import flax.traverse_util as traverse_util

    p = pathlib.Path(params_path).resolve()
    params_dir = p / "params" if (p / "params" / "_METADATA").exists() else p
    logger.info("Loading: %s", params_dir)
    params = restore_params(params_dir, restore_type=np.ndarray)
    flat = traverse_util.flatten_dict(params)
    return {"/".join(k): v for k, v in flat.items()}


def _validate_keys(reference: dict[str, np.ndarray], others: Dict[str, dict[str, np.ndarray]]) -> None:
    ref_keys = set(reference.keys())
    for name, m in others.items():
        keys = set(m.keys())
        if keys != ref_keys:
            miss = sorted(ref_keys - keys)
            extra = sorted(keys - ref_keys)
            raise ValueError(
                f"Key mismatch for {name}: missing={len(miss)} extra={len(extra)} "
                f"first_missing={miss[:3]} first_extra={extra[:3]}"
            )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Joint repair map analysis")
    ap.add_argument("--joint", type=str, default="checkpoints/pi05_libero/my_experiment/29999")
    ap.add_argument("--mean4-iter0", type=str, default="checkpoints/wudi_mllm/4task_mean_iter0")
    ap.add_argument(
        "--drop-csv",
        type=str,
        default="docs/analysis/from10k_llm_only_ffn_layerwise_retain_rate/layerwise_drop_summary_r10.csv",
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
        default="docs/analysis/joint_target_alignment_2026-04-20",
    )
    ap.add_argument("--log-level", type=str, default="INFO")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s %(message)s")

    joint = _load_flat(args.joint)
    mean0 = _load_flat(args.mean4_iter0)
    _validate_keys(joint, {"mean4_iter0": mean0})

    mod_ss = {m: 0.0 for m in MODULES}
    layer_ss = {(m, l): 0.0 for m in ("gating_einsum", "linear") for l in range(NUM_LAYERS)}

    for key in sorted(joint.keys()):
        if not _in_scope(key, args.scope):
            continue
        mod = _module_of_key(key)
        if mod is None:
            continue
        d = np.asarray(joint[key], dtype=np.float64) - np.asarray(mean0[key], dtype=np.float64)
        v = d.ravel()
        ss = float(np.dot(v, v))
        mod_ss[mod] += ss
        if mod in ("gating_einsum", "linear") and d.ndim >= 1 and d.shape[0] == NUM_LAYERS:
            for l in range(NUM_LAYERS):
                vl = d[l].ravel()
                layer_ss[(mod, l)] += float(np.dot(vl, vl))

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_ss = sum(mod_ss.values())
    with (out_dir / "repair_module_norm.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["module", "l2_norm", "share_of_total"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for mod in MODULES:
            l2 = np.sqrt(mod_ss[mod])
            share = 0.0 if total_ss == 0 else mod_ss[mod] / total_ss
            w.writerow({"module": mod, "l2_norm": f"{l2:.12f}", "share_of_total": f"{share:.12f}"})

    with (out_dir / "repair_ffn_layer_norm.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["module", "layer", "region", "l2_norm"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for mod in ("gating_einsum", "linear"):
            for l in range(NUM_LAYERS):
                w.writerow(
                    {
                        "module": mod,
                        "layer": str(l),
                        "region": _region_of_layer(l),
                        "l2_norm": f"{np.sqrt(layer_ss[(mod, l)]):.12f}",
                    }
                )

    # Correlation with existing drop table
    drop_rows = []
    with pathlib.Path(args.drop_csv).open("r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            drop_rows.append(r)

    corr_rows = []
    x_all = []
    y_all = []
    for mod in ("gating_einsum", "linear"):
        xs = []
        ys = []
        for l in range(NUM_LAYERS):
            repair_l2 = np.sqrt(layer_ss[(mod, l)])
            # use -drop_r10 so larger means worse retained compression
            match = next((r for r in drop_rows if r["module"] == mod and int(r["layer"]) == l), None)
            if match is None:
                continue
            damage = -float(match["drop_r10"])
            xs.append(repair_l2)
            ys.append(damage)
            x_all.append(repair_l2)
            y_all.append(damage)
        x = np.asarray(xs, dtype=np.float64)
        y = np.asarray(ys, dtype=np.float64)
        corr_rows.append(
            {
                "scope": mod,
                "n": str(x.size),
                "pearson": f"{_pearson(x, y):.12f}",
                "spearman": f"{_spearman(x, y):.12f}",
            }
        )

    xa = np.asarray(x_all, dtype=np.float64)
    ya = np.asarray(y_all, dtype=np.float64)
    corr_rows.append(
        {
            "scope": "ffn_all",
            "n": str(xa.size),
            "pearson": f"{_pearson(xa, ya):.12f}",
            "spearman": f"{_spearman(xa, ya):.12f}",
        }
    )

    with (out_dir / "repair_vs_drop_correlation.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["scope", "n", "pearson", "spearman"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(corr_rows)

    print("Saved:")
    print(f"  {out_dir / 'repair_module_norm.csv'}")
    print(f"  {out_dir / 'repair_ffn_layer_norm.csv'}")
    print(f"  {out_dir / 'repair_vs_drop_correlation.csv'}")


if __name__ == "__main__":
    main()

