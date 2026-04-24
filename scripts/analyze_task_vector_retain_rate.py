#!/usr/bin/env python3
"""Compute retain-rate diagnostics for 2-task/4-task means on from10k checkpoints.

Metrics:
  Delta_t = W_t - W_base
  Delta_m2 = (Delta_10 + Delta_spatial) / 2
  Delta_m4 = (Delta_10 + Delta_spatial + Delta_object + Delta_goal) / 4

  r_t(Delta_m)        = <Delta_m, Delta_t> / ||Delta_t||^2
  cos(Delta_m,Delta_t)= <Delta_m, Delta_t> / (||Delta_m|| * ||Delta_t||)
  norm_ratio          = ||Delta_m|| / ||Delta_t||

Outputs:
  docs/analysis/from10k_llm_only_retain_rate/retain_rate_mean_2task_4task.csv
  docs/analysis/from10k_llm_only_retain_rate/retain_rate_by_module.csv
  docs/analysis/from10k_llm_only_retain_rate/retain_rate_drop_summary.csv
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


_FROZEN_PREFIXES = (
    "PaliGemma/llm/embedder/",
    "action_in_proj/",
    "action_out_proj/",
    "time_mlp_in/",
    "time_mlp_out/",
)

_MOD_Q = re.compile(r".*/q_einsum(_\d+)?/w$")
_MOD_KV = re.compile(r".*/kv_einsum(_\d+)?/w$")
_MOD_AV = re.compile(r".*/attn_vec_einsum(_\d+)?/w$")
_MOD_GATE = re.compile(r".*/gating_einsum(_\d+)?$")
_MOD_LIN = re.compile(r".*/linear(_\d+)?$")

TASKS = ("libero_10", "libero_spatial", "libero_object", "libero_goal")


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
    if _MOD_KV.fullmatch(key):
        return "kv_einsum"
    if _MOD_Q.fullmatch(key):
        return "q_einsum"
    if _MOD_AV.fullmatch(key):
        return "attn_vec_einsum"
    if _MOD_GATE.fullmatch(key):
        return "gating_einsum"
    if _MOD_LIN.fullmatch(key):
        return "linear"
    return None


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
class Acc:
    dot2: dict[str, float]
    dot4: dict[str, float]
    norm_t: dict[str, float]
    norm_m2: float
    norm_m4: float

    @classmethod
    def create(cls) -> "Acc":
        return cls(
            dot2={t: 0.0 for t in TASKS},
            dot4={t: 0.0 for t in TASKS},
            norm_t={t: 0.0 for t in TASKS},
            norm_m2=0.0,
            norm_m4=0.0,
        )


def _update_acc(acc: Acc, d: dict[str, np.ndarray]) -> None:
    d2 = 0.5 * (d["libero_10"] + d["libero_spatial"])
    d4 = 0.25 * (d["libero_10"] + d["libero_spatial"] + d["libero_object"] + d["libero_goal"])

    acc.norm_m2 += float(np.dot(d2, d2))
    acc.norm_m4 += float(np.dot(d4, d4))
    for t in TASKS:
        dt = d[t]
        acc.norm_t[t] += float(np.dot(dt, dt))
        acc.dot2[t] += float(np.dot(d2, dt))
        acc.dot4[t] += float(np.dot(d4, dt))


def _safe_div(a: float, b: float) -> float:
    if b == 0.0:
        return float("nan")
    return a / b


def _to_rows(scope_acc: Acc, with_2task_mask: dict[str, bool]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for t in TASKS:
        n_t = np.sqrt(scope_acc.norm_t[t])
        n_m2 = np.sqrt(scope_acc.norm_m2)
        n_m4 = np.sqrt(scope_acc.norm_m4)

        if with_2task_mask[t]:
            r2 = _safe_div(scope_acc.dot2[t], scope_acc.norm_t[t])
            c2 = _safe_div(scope_acc.dot2[t], n_m2 * n_t)
            nr2 = _safe_div(n_m2, n_t)
            r2_s = f"{r2:.12f}"
            c2_s = f"{c2:.12f}"
            nr2_s = f"{nr2:.12f}"
        else:
            r2_s = ""
            c2_s = ""
            nr2_s = ""

        r4 = _safe_div(scope_acc.dot4[t], scope_acc.norm_t[t])
        c4 = _safe_div(scope_acc.dot4[t], n_m4 * n_t)
        nr4 = _safe_div(n_m4, n_t)

        rows.append(
            {
                "task": t,
                "r_in_2task_mean": r2_s,
                "r_in_4task_mean": f"{r4:.12f}",
                "cos_with_2task_mean": c2_s,
                "cos_with_4task_mean": f"{c4:.12f}",
                "norm_ratio_2task_mean": nr2_s,
                "norm_ratio_4task_mean": f"{nr4:.12f}",
            }
        )
    return rows


def _write_csv(path: pathlib.Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Retain-rate diagnostics for from10k checkpoints")
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
        default="docs/analysis/from10k_llm_only_retain_rate",
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

    global_acc = Acc.create()
    module_accs = {
        "q_einsum": Acc.create(),
        "kv_einsum": Acc.create(),
        "attn_vec_einsum": Acc.create(),
        "gating_einsum": Acc.create(),
        "linear": Acc.create(),
    }

    all_keys = sorted(base.keys())
    selected = 0
    selected_by_module = {m: 0 for m in module_accs}
    for key in all_keys:
        if not _in_scope(key, args.scope):
            continue
        selected += 1
        b = np.asarray(base[key], dtype=np.float64).ravel()
        d = {
            t: (np.asarray(models[t][key], dtype=np.float64).ravel() - b)
            for t in TASKS
        }
        _update_acc(global_acc, d)
        mod = _module_of_key(key)
        if mod is not None:
            selected_by_module[mod] += 1
            _update_acc(module_accs[mod], d)

    logger.info("Scope=%s, selected params=%d / %d", args.scope, selected, len(all_keys))
    for m, c in selected_by_module.items():
        logger.info("Module %-16s keys=%d", m, c)

    out_dir = pathlib.Path(args.output_dir)
    common_fields = [
        "task",
        "r_in_2task_mean",
        "r_in_4task_mean",
        "cos_with_2task_mean",
        "cos_with_4task_mean",
        "norm_ratio_2task_mean",
        "norm_ratio_4task_mean",
    ]

    with_2task = {
        "libero_10": True,
        "libero_spatial": True,
        "libero_object": False,
        "libero_goal": False,
    }

    global_rows = _to_rows(global_acc, with_2task_mask=with_2task)
    _write_csv(out_dir / "retain_rate_mean_2task_4task.csv", global_rows, common_fields)

    mod_rows: list[dict[str, str]] = []
    drop_rows: list[dict[str, str]] = []
    for mod in ("q_einsum", "kv_einsum", "attn_vec_einsum", "gating_einsum", "linear"):
        rows = _to_rows(module_accs[mod], with_2task_mask=with_2task)
        for r in rows:
            rr = {"module": mod}
            rr.update(r)
            mod_rows.append(rr)

        r10_2 = rows[0]["r_in_2task_mean"]
        r10_4 = rows[0]["r_in_4task_mean"]
        drop = (
            float(r10_4) - float(r10_2)
            if r10_2 != "" and r10_4 != ""
            else float("nan")
        )
        drop_rows.append(
            {
                "module": mod,
                "r10_2task": r10_2,
                "r10_4task": r10_4,
                "drop_r10": f"{drop:.12f}",
            }
        )

    _write_csv(
        out_dir / "retain_rate_by_module.csv",
        mod_rows,
        ["module"] + common_fields,
    )
    _write_csv(
        out_dir / "retain_rate_drop_summary.csv",
        drop_rows,
        ["module", "r10_2task", "r10_4task", "drop_r10"],
    )

    print("\nGlobal retain-rate summary (focus tasks):")
    for r in global_rows:
        t = r["task"]
        print(
            f"  {t:14s} r2={r['r_in_2task_mean'] or '-':>14}  "
            f"r4={r['r_in_4task_mean']:>14}  "
            f"cos2={r['cos_with_2task_mean'] or '-':>14}  "
            f"cos4={r['cos_with_4task_mean']:>14}"
        )
    print(f"\nSaved: {out_dir / 'retain_rate_mean_2task_4task.csv'}")
    print(f"Saved: {out_dir / 'retain_rate_by_module.csv'}")
    print(f"Saved: {out_dir / 'retain_rate_drop_summary.csv'}")


if __name__ == "__main__":
    main()

