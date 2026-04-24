#!/usr/bin/env python3
"""Generate selective-protected checkpoints from 4task mean iter0.

Two strategy families:
  - *_joint: protected layers are replaced by joint checkpoint layers
  - *_base:  protected layers are replaced by base checkpoint layers

Presets:
  - minimal: gating={6,10,11}, linear={1,2,4}
  - region : gating=6..11,     linear=1..11
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import re
import shutil
import sys
from typing import Dict

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
_RE_GATE = re.compile(r".*/gating_einsum(_\d+)?$")
_RE_LINEAR = re.compile(r".*/linear(_\d+)?$")
_NUM_LAYERS = 18


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


def _load_flat(params_path: str | pathlib.Path) -> dict[str, np.ndarray]:
    import flax.traverse_util as traverse_util

    p = pathlib.Path(params_path).resolve()
    params_dir = p / "params" if (p / "params" / "_METADATA").exists() else p
    logger.info("Loading: %s", params_dir)
    params = restore_params(params_dir, restore_type=np.ndarray)
    flat = traverse_util.flatten_dict(params)
    return {"/".join(k): v for k, v in flat.items()}


def _save_flat(flat_params: dict[str, np.ndarray], output_dir: str | pathlib.Path) -> None:
    import flax.traverse_util as traverse_util
    import orbax.checkpoint as ocp

    output = pathlib.Path(output_dir).resolve() / "params"
    if output.exists():
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    tuple_flat = {tuple(k.split("/")): v for k, v in flat_params.items()}
    nested = traverse_util.unflatten_dict(tuple_flat)
    ocp.PyTreeCheckpointer().save(output, {"params": nested})
    logger.info("Saved: %s", output)


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
    ap = argparse.ArgumentParser(description="Generate selective protected checkpoints from mean4")
    ap.add_argument("--mean4", type=str, default="checkpoints/wudi_mllm/4task_mean_iter0")
    ap.add_argument("--base", type=str, default="checkpoints/pi05_libero/my_experiment/10000")
    ap.add_argument("--joint", type=str, default="checkpoints/pi05_libero/my_experiment/29999")
    ap.add_argument(
        "--output-root",
        type=str,
        default="checkpoints/ablation_selective_protect",
    )
    ap.add_argument(
        "--scope",
        type=str,
        default="llm_only",
        choices=["expert1_only", "both_experts", "llm_only", "lang_and_vision"],
    )
    ap.add_argument("--log-level", type=str, default="INFO")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s %(message)s")

    mean4 = _load_flat(args.mean4)
    base = _load_flat(args.base)
    joint = _load_flat(args.joint)
    _validate_keys(mean4, {"base": base, "joint": joint})

    presets = {
        "minimal": {
            "gating_einsum": {6, 10, 11},
            "linear": {1, 2, 4},
        },
        "region": {
            "gating_einsum": set(range(6, 12)),
            "linear": set(range(1, 12)),
        },
    }
    sources = {"joint": joint, "base": base}

    output_root = pathlib.Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for preset_name, layer_map in presets.items():
        for source_name, source_flat in sources.items():
            out_name = f"mean4_protect_{preset_name}_{source_name}"
            out_dir = output_root / out_name

            out = dict(mean4)  # shallow copy: unchanged arrays are shared
            changed_keys = 0
            changed_slices = 0

            for key in sorted(mean4.keys()):
                if not _in_scope(key, args.scope):
                    continue
                mod = _ffn_module_of_key(key)
                if mod is None:
                    continue
                arr = np.asarray(mean4[key])
                src = np.asarray(source_flat[key])
                if arr.shape != src.shape or arr.ndim < 1 or arr.shape[0] != _NUM_LAYERS:
                    continue

                target_layers = layer_map[mod]
                if not target_layers:
                    continue

                new_arr = arr.copy()
                for l in target_layers:
                    new_arr[l] = src[l]
                    changed_slices += 1
                out[key] = new_arr
                changed_keys += 1

            logger.info(
                "Building %s: changed_keys=%d changed_slices=%d",
                out_name,
                changed_keys,
                changed_slices,
            )
            _save_flat(out, out_dir)


if __name__ == "__main__":
    main()

