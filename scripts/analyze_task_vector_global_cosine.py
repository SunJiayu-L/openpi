#!/usr/bin/env python3
"""Compute global cosine similarities of task vectors for from10k 4-task checkpoints.

Definition:
  Delta_t = W_t - W_base
  cos(Delta_i, Delta_j) = <Delta_i, Delta_j> / (||Delta_i|| * ||Delta_j||)

Scope filtering follows scripts/wudi_merge.py `_in_scope` for consistency with WUDI merge.
This script computes global cosine over all in-scope params under a fixed scope
(default: llm_only), then outputs:
  - global_cosine_6pairs.csv
  - global_cosine_matrix_4x4.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import logging
import pathlib
import re
import sys
from typing import Dict, Iterable

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from openpi.models.model import restore_params

logger = logging.getLogger(__name__)


_FROZEN_PREFIXES = (
    "PaliGemma/llm/embedder/",
    "action_in_proj/",
    "action_out_proj/",
    "time_mlp_in/",
    "time_mlp_out/",
)


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


def _iter_scope_keys(keys: Iterable[str], scope: str) -> list[str]:
    selected = [k for k in keys if _in_scope(k, scope)]
    selected.sort()
    return selected


def _compute_cosines(
    base: dict[str, np.ndarray],
    models: Dict[str, dict[str, np.ndarray]],
    scope_keys: list[str],
) -> dict[tuple[str, str], float]:
    names = list(models.keys())
    norms = {n: 0.0 for n in names}
    dots = {(a, b): 0.0 for a, b in itertools.combinations_with_replacement(names, 2)}

    for k in scope_keys:
        b = np.asarray(base[k], dtype=np.float64).ravel()
        deltas: dict[str, np.ndarray] = {}
        for n in names:
            deltas[n] = np.asarray(models[n][k], dtype=np.float64).ravel() - b

        for n in names:
            v = deltas[n]
            norms[n] += float(np.dot(v, v))

        for a, bname in itertools.combinations_with_replacement(names, 2):
            va = deltas[a]
            vb = deltas[bname]
            dots[(a, bname)] += float(np.dot(va, vb))

    cos: dict[tuple[str, str], float] = {}
    for a, bname in itertools.combinations_with_replacement(names, 2):
        denom = np.sqrt(norms[a]) * np.sqrt(norms[bname])
        if denom == 0:
            val = float("nan")
        else:
            val = dots[(a, bname)] / denom
        cos[(a, bname)] = val
        cos[(bname, a)] = val

    return cos


def _write_outputs(
    output_dir: pathlib.Path,
    names: list[str],
    cos: dict[tuple[str, str], float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_order = [
        ("libero_10", "libero_spatial"),
        ("libero_10", "libero_object"),
        ("libero_10", "libero_goal"),
        ("libero_spatial", "libero_object"),
        ("libero_spatial", "libero_goal"),
        ("libero_object", "libero_goal"),
    ]

    pair_csv = output_dir / "global_cosine_6pairs.csv"
    with pair_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["task_i", "task_j", "cosine"])
        for i, j in pair_order:
            w.writerow([i, j, f"{cos[(i, j)]:.12f}"])

    matrix_csv = output_dir / "global_cosine_matrix_4x4.csv"
    with matrix_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["task"] + names)
        for i in names:
            w.writerow([i] + [f"{cos[(i, j)]:.12f}" for j in names])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Global cosine analysis for from10k task vectors")
    ap.add_argument(
        "--base",
        type=str,
        default="checkpoints/pi05_libero/my_experiment/10000",
        help="Base checkpoint (step dir or params dir)",
    )
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
        help="Scope filter, aligned with scripts/wudi_merge.py",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default="docs/analysis/from10k_llm_only_global_cosine",
    )
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
    scope_keys = _iter_scope_keys(base.keys(), args.scope)
    if not scope_keys:
        raise ValueError(f"No parameters selected under scope={args.scope}")

    logger.info("Scope=%s, selected params=%d / %d", args.scope, len(scope_keys), len(base))

    names = ["libero_10", "libero_spatial", "libero_object", "libero_goal"]
    cos = _compute_cosines(base, models, scope_keys)

    out = pathlib.Path(args.output_dir)
    _write_outputs(out, names, cos)

    print("\\nGlobal cosine (focus 6 pairs):")
    focus = [
        ("libero_10", "libero_spatial"),
        ("libero_10", "libero_object"),
        ("libero_10", "libero_goal"),
        ("libero_spatial", "libero_object"),
        ("libero_spatial", "libero_goal"),
        ("libero_object", "libero_goal"),
    ]
    for i, j in focus:
        print(f"  cos({i}, {j}) = {cos[(i, j)]:.8f}")

    print(f"\\nSaved: {out / 'global_cosine_6pairs.csv'}")
    print(f"Saved: {out / 'global_cosine_matrix_4x4.csv'}")


if __name__ == "__main__":
    main()
