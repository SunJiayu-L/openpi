#!/usr/bin/env python3
"""Analyze alignment of model deltas to joint-SFT delta (base@10k reference).

Outputs:
  - geometry_alignment_to_joint_global.csv
  - geometry_alignment_to_joint_by_module.csv
  - geometry_alignment_to_joint_ffn_layer.csv
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
TASK_MODELS = ("mean4_iter0", "mean4_iter100", "ft_from_500", "ft_from_1k")
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


def _cos(dot: float, n1: float, n2: float) -> float:
    d = np.sqrt(n1) * np.sqrt(n2)
    return float("nan") if d == 0 else dot / d


def _rel_l2(n_diff: float, n_ref: float) -> float:
    d = np.sqrt(n_ref)
    return float("nan") if d == 0 else np.sqrt(n_diff) / d


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Joint target alignment analysis")
    ap.add_argument("--base", type=str, default="checkpoints/pi05_libero/my_experiment/10000")
    ap.add_argument("--joint", type=str, default="checkpoints/pi05_libero/my_experiment/29999")
    ap.add_argument("--mean4-iter0", type=str, default="checkpoints/wudi_mllm/4task_mean_iter0")
    ap.add_argument("--mean4-iter100", type=str, default="checkpoints/wudi_mllm/4task_mean_iter100")
    ap.add_argument(
        "--ft-from-500",
        type=str,
        default="checkpoints/pi05_libero_4task_from_wudi_4task_500/ft_from_wudi_4task_500/9999",
    )
    ap.add_argument(
        "--ft-from-1k",
        type=str,
        default="checkpoints/pi05_libero_4task_from_wudi_4task_1k/ft_from_wudi_4task_1k/9000",
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

    base = _load_flat(args.base)
    joint = _load_flat(args.joint)
    model_paths = {
        "mean4_iter0": args.mean4_iter0,
        "mean4_iter100": args.mean4_iter100,
        "ft_from_500": args.ft_from_500,
        "ft_from_1k": args.ft_from_1k,
    }
    _validate_keys(base, {"joint": joint})

    selected_keys = [k for k in sorted(base.keys()) if _in_scope(k, args.scope)]
    logger.info("Scope=%s selected keys=%d", args.scope, len(selected_keys))

    # Precompute joint norms once.
    g_n_joint = 0.0
    m_n_joint = {mm: 0.0 for mm in MODULES}
    ffn_mods = ("gating_einsum", "linear")
    l_n_joint = {(mod, l): 0.0 for mod in ffn_mods for l in range(NUM_LAYERS)}
    for key in selected_keys:
        b = np.asarray(base[key], dtype=np.float64)
        d_joint = np.asarray(joint[key], dtype=np.float64) - b
        vj = d_joint.ravel()
        g_n_joint += float(np.dot(vj, vj))

        mod = _module_of_key(key)
        if mod is not None:
            m_n_joint[mod] += float(np.dot(vj, vj))
        if mod in ffn_mods and d_joint.ndim >= 1 and d_joint.shape[0] == NUM_LAYERS:
            for l in range(NUM_LAYERS):
                vjl = d_joint[l].ravel()
                l_n_joint[(mod, l)] += float(np.dot(vjl, vjl))

    # Accumulate per model one by one to avoid OOM.
    global_rows = []
    module_rows = []
    layer_rows = []
    for model_name in TASK_MODELS:
        logger.info("Processing model=%s", model_name)
        mflat = _load_flat(model_paths[model_name])
        _validate_keys(base, {model_name: mflat})

        g_dot = 0.0
        g_n_model = 0.0
        g_n_diff = 0.0
        m_dot = {mm: 0.0 for mm in MODULES}
        m_n_model = {mm: 0.0 for mm in MODULES}
        m_n_diff = {mm: 0.0 for mm in MODULES}
        l_dot = {(mod, l): 0.0 for mod in ffn_mods for l in range(NUM_LAYERS)}
        l_n_model = {(mod, l): 0.0 for mod in ffn_mods for l in range(NUM_LAYERS)}
        l_n_diff = {(mod, l): 0.0 for mod in ffn_mods for l in range(NUM_LAYERS)}

        for key in selected_keys:
            b = np.asarray(base[key], dtype=np.float64)
            d_joint = np.asarray(joint[key], dtype=np.float64) - b
            d_m = np.asarray(mflat[key], dtype=np.float64) - b
            vj = d_joint.ravel()
            vm = d_m.ravel()
            diff = vm - vj
            g_dot += float(np.dot(vm, vj))
            g_n_model += float(np.dot(vm, vm))
            g_n_diff += float(np.dot(diff, diff))

            mod = _module_of_key(key)
            if mod is not None:
                m_dot[mod] += float(np.dot(vm, vj))
                m_n_model[mod] += float(np.dot(vm, vm))
                m_n_diff[mod] += float(np.dot(diff, diff))

            if mod in ffn_mods and d_joint.ndim >= 1 and d_joint.shape[0] == NUM_LAYERS:
                for l in range(NUM_LAYERS):
                    vjl = d_joint[l].ravel()
                    vml = d_m[l].ravel()
                    dfl = vml - vjl
                    l_dot[(mod, l)] += float(np.dot(vml, vjl))
                    l_n_model[(mod, l)] += float(np.dot(vml, vml))
                    l_n_diff[(mod, l)] += float(np.dot(dfl, dfl))

        global_rows.append(
            {
                "model": model_name,
                "cos_to_joint": f"{_cos(g_dot, g_n_model, g_n_joint):.12f}",
                "rel_l2_to_joint": f"{_rel_l2(g_n_diff, g_n_joint):.12f}",
                "delta_norm": f"{np.sqrt(g_n_model):.12f}",
                "joint_delta_norm": f"{np.sqrt(g_n_joint):.12f}",
            }
        )
        for mod in MODULES:
            module_rows.append(
                {
                    "model": model_name,
                    "module": mod,
                    "cos_to_joint": f"{_cos(m_dot[mod], m_n_model[mod], m_n_joint[mod]):.12f}",
                    "rel_l2_to_joint": f"{_rel_l2(m_n_diff[mod], m_n_joint[mod]):.12f}",
                    "delta_norm": f"{np.sqrt(m_n_model[mod]):.12f}",
                    "joint_delta_norm": f"{np.sqrt(m_n_joint[mod]):.12f}",
                }
            )
        for mod in ffn_mods:
            for l in range(NUM_LAYERS):
                layer_rows.append(
                    {
                        "model": model_name,
                        "module": mod,
                        "layer": str(l),
                        "cos_to_joint": f"{_cos(l_dot[(mod, l)], l_n_model[(mod, l)], l_n_joint[(mod, l)]):.12f}",
                        "rel_l2_to_joint": f"{_rel_l2(l_n_diff[(mod, l)], l_n_joint[(mod, l)]):.12f}",
                        "delta_norm": f"{np.sqrt(l_n_model[(mod, l)]):.12f}",
                        "joint_delta_norm": f"{np.sqrt(l_n_joint[(mod, l)]):.12f}",
                    }
                )

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Global table
    with (out_dir / "geometry_alignment_to_joint_global.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["model", "cos_to_joint", "rel_l2_to_joint", "delta_norm", "joint_delta_norm"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(global_rows)

    # Module table
    with (out_dir / "geometry_alignment_to_joint_by_module.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["model", "module", "cos_to_joint", "rel_l2_to_joint", "delta_norm", "joint_delta_norm"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(module_rows)

    # FFN layer table
    with (out_dir / "geometry_alignment_to_joint_ffn_layer.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "model",
            "module",
            "layer",
            "cos_to_joint",
            "rel_l2_to_joint",
            "delta_norm",
            "joint_delta_norm",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(layer_rows)

    print("Saved:")
    print(f"  {out_dir / 'geometry_alignment_to_joint_global.csv'}")
    print(f"  {out_dir / 'geometry_alignment_to_joint_by_module.csv'}")
    print(f"  {out_dir / 'geometry_alignment_to_joint_ffn_layer.csv'}")


if __name__ == "__main__":
    main()
