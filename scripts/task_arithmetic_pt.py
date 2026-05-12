#!/usr/bin/env python3
"""Task Arithmetic merging for PyTorch safetensors checkpoints.

公式: theta_star = theta_SFT + alpha * (theta_RL - theta_SFT)
                = theta_SFT + alpha * Delta_RL

函数式风格参照 MLLMerging/InternVL/internvl_chat/model_merging.py 中的
TaskVector + task_arithmetic 实现，但直接对 safetensors state_dict 操作，
不需要实例化完整的 nn.Module。

Usage:
    python scripts/task_arithmetic_pt.py \\
        --base   checkpoints/pt_converted/base_25k \\
        --ft     checkpoints/pt_rl/ppo_step80 \\
        --output checkpoints/merged/base25k_ppo80_a05 \\
        --alpha  0.5
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import re
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Param name filter
# ---------------------------------------------------------------------------

# RLinf PPO 训练 pi0.5 时的可训练参数白名单。
# 来源：RLinf train_expert_only=True 策略 + 实测 base_25k vs PPO step80 的张量 diff（208/812 个 key 改变）。
# - 198 keys: paligemma_with_expert.gemma_expert.model.layers.*
# -   2 keys: paligemma_with_expert.gemma_expert.model.norm.*
# -   2 keys: action_in_proj.{weight,bias}
# -   2 keys: action_out_proj.{weight,bias}
# -   4 keys: time_mlp_{in,out}.{weight,bias}
# 其余 604 个 paligemma backbone 参数在 RL 中冻结，不参与融合。
PI05_PPO_TRAINABLE_PREFIXES: tuple[str, ...] = (
    "paligemma_with_expert.gemma_expert.model.layers.",
    "paligemma_with_expert.gemma_expert.model.norm.",
    "action_in_proj.",
    "action_out_proj.",
    "time_mlp_in.",
    "time_mlp_out.",
)


def get_param_names_to_merge(
    input_param_names: list[str],
    exclude_param_names_regex: list[str] | None = None,
    include_param_names_prefixes: tuple[str, ...] | None = None,
) -> list[str]:
    """筛选参与融合的参数名。

    顺序: 先 include 白名单（若指定）→ 再 exclude 正则黑名单。
    白名单为空 → 不限制（保留 model_merging.py 的原行为）。
    """
    exclude_param_names_regex = exclude_param_names_regex or []
    out = []
    for name in input_param_names:
        if include_param_names_prefixes and not any(name.startswith(p) for p in include_param_names_prefixes):
            continue
        if any(re.match(p, name) for p in exclude_param_names_regex):
            continue
        out.append(name)
    return out


# ---------------------------------------------------------------------------
# TaskVector (state_dict-based, 对齐 model_merging.py 的接口)
# ---------------------------------------------------------------------------

class TaskVector:
    """任务向量: tau = theta_finetuned - theta_pretrained。

    与 MLLMerging 版本的接口一致，差异仅在于这里直接接受 state_dict（dict[str, Tensor]）
    而不是 nn.Module，从而避免实例化大模型。
    """

    def __init__(
        self,
        pretrained_state_dict: dict[str, torch.Tensor] | None = None,
        finetuned_state_dict: dict[str, torch.Tensor] | None = None,
        exclude_param_names_regex: list[str] | None = None,
        include_param_names_prefixes: tuple[str, ...] | None = None,
        task_vector_param_dict: dict[str, torch.Tensor] | None = None,
    ) -> None:
        if task_vector_param_dict is not None:
            self.task_vector_param_dict = task_vector_param_dict
            return

        assert pretrained_state_dict is not None and finetuned_state_dict is not None
        exclude_param_names_regex = exclude_param_names_regex or []

        param_names_to_merge = get_param_names_to_merge(
            list(pretrained_state_dict.keys()),
            exclude_param_names_regex=exclude_param_names_regex,
            include_param_names_prefixes=include_param_names_prefixes,
        )

        self.task_vector_param_dict = {}
        with torch.no_grad():
            for name in param_names_to_merge:
                if name not in finetuned_state_dict:
                    logger.warning(f"  skip (missing in ft): {name}")
                    continue
                a = pretrained_state_dict[name]
                b = finetuned_state_dict[name]
                if a.shape != b.shape:
                    logger.warning(f"  skip (shape mismatch {a.shape} vs {b.shape}): {name}")
                    continue
                # 用 float32 做差，避免 bfloat16 的舍入误差累积。
                self.task_vector_param_dict[name] = b.to(torch.float32) - a.to(torch.float32)

    def __add__(self, other: "TaskVector") -> "TaskVector":
        assert isinstance(other, TaskVector)
        new_dict: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name in self.task_vector_param_dict:
                assert name in other.task_vector_param_dict, f"{name} not in other"
                new_dict[name] = self.task_vector_param_dict[name] + other.task_vector_param_dict[name]
        return TaskVector(task_vector_param_dict=new_dict)

    def __radd__(self, other) -> "TaskVector":
        return self.__add__(other)

    def combine_with_pretrained_model(
        self,
        pretrained_state_dict: dict[str, torch.Tensor],
        scaling_coefficient: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """合并: theta_star[name] = theta_pretrained[name] + alpha * tau[name]。

        未在 task_vector_param_dict 中的 key（被 exclude 的）会原样保留 pretrained 值，
        以保证输出 state_dict 与 pretrained 同构。
        """
        merged: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, base_val in pretrained_state_dict.items():
                if name in self.task_vector_param_dict:
                    tv = self.task_vector_param_dict[name]
                    out = base_val.to(torch.float32) + scaling_coefficient * tv
                    # 保持与 pretrained 同 dtype（避免 bf16↔fp32 不一致）。
                    merged[name] = out.to(base_val.dtype)
                else:
                    merged[name] = base_val
        return merged


# ---------------------------------------------------------------------------
# 顶层融合函数（对齐 model_merging.py 的 task_arithmetic）
# ---------------------------------------------------------------------------

def task_arithmetic(
    pretrained_state_dict: dict[str, torch.Tensor],
    finetuned_state_dicts: list[dict[str, torch.Tensor]],
    exclude_param_names_regex: list[str],
    scaling_coefficient: float = 1.0,
    include_param_names_prefixes: tuple[str, ...] | None = None,
) -> dict[str, torch.Tensor]:
    """Task Arithmetic 融合。

    对 N 个微调模型，先各自计算 task_vector_i，再求和，最后:
        theta_star = theta_pretrained + alpha * sum_i tau_i

    N=1 时退化为: theta_star = theta_pretrained + alpha * (theta_ft - theta_pretrained)
    """
    assert isinstance(scaling_coefficient, float)
    assert len(finetuned_state_dicts) >= 1

    task_vectors = [
        TaskVector(
            pretrained_state_dict=pretrained_state_dict,
            finetuned_state_dict=ft,
            exclude_param_names_regex=exclude_param_names_regex,
            include_param_names_prefixes=include_param_names_prefixes,
        )
        for ft in finetuned_state_dicts
    ]

    with torch.no_grad():
        merged_tv = task_vectors[0]
        for tv in task_vectors[1:]:
            merged_tv = merged_tv + tv
        merged = merged_tv.combine_with_pretrained_model(
            pretrained_state_dict=pretrained_state_dict,
            scaling_coefficient=scaling_coefficient,
        )

    return merged


# ---------------------------------------------------------------------------
# Safetensors I/O
# ---------------------------------------------------------------------------

def load_safetensors_state_dict(ckpt_dir: str | pathlib.Path) -> dict[str, torch.Tensor]:
    """读取 ckpt_dir/model.safetensors 为 state_dict (CPU, 原 dtype)。"""
    path = pathlib.Path(ckpt_dir) / "model.safetensors"
    if not path.exists():
        raise FileNotFoundError(f"model.safetensors not found at {path}")
    out: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
    logger.info(f"  loaded {len(out)} tensors from {path}")
    return out


def save_merged_checkpoint(
    merged_state_dict: dict[str, torch.Tensor],
    output_dir: str | pathlib.Path,
    base_dir: str | pathlib.Path,
    cast_dtype: torch.dtype | None = torch.bfloat16,
) -> None:
    """保存合并后的 checkpoint，并复制 base_dir 的辅助文件 (config.json, assets/, ...)。"""
    output_dir = pathlib.Path(output_dir)
    base_dir = pathlib.Path(base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 全部 cast 到指定 dtype（默认 bf16，与 base_25k config.json 一致）。
    if cast_dtype is not None:
        merged_state_dict = {k: v.to(cast_dtype).contiguous() for k, v in merged_state_dict.items()}
    else:
        merged_state_dict = {k: v.contiguous() for k, v in merged_state_dict.items()}

    save_path = output_dir / "model.safetensors"
    save_file(merged_state_dict, str(save_path))
    logger.info(f"  saved merged model.safetensors → {save_path}")

    # 复制 config.json (若 base 有则复制；没有则按 base_25k 默认结构生成)。
    src_cfg = base_dir / "config.json"
    if src_cfg.exists():
        shutil.copy2(src_cfg, output_dir / "config.json")
        logger.info(f"  copied config.json")

    # 复制 assets/ 和 physical-intelligence/ 子目录，serve_policy.py 需要 norm_stats。
    for sub in ("assets", "physical-intelligence"):
        src = base_dir / sub
        dst = output_dir / sub
        if src.exists() and not dst.exists():
            shutil.copytree(src, dst)
            logger.info(f"  copied {sub}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_merge(
    base_path: str,
    ft_paths: list[str],
    output_path: str,
    alpha: float,
    exclude_param_names_regex: list[str],
    cast_dtype: str = "bfloat16",
    include_preset: str = "pi05_ppo",
) -> None:
    logger.info(f"Loading base: {base_path}")
    base_sd = load_safetensors_state_dict(base_path)

    logger.info(f"Loading {len(ft_paths)} fine-tuned ckpt(s)…")
    ft_sds = [load_safetensors_state_dict(p) for p in ft_paths]

    # 检查 key 对齐情况
    base_keys = set(base_sd.keys())
    for i, ft in enumerate(ft_sds):
        ft_keys = set(ft.keys())
        only_base = base_keys - ft_keys
        only_ft = ft_keys - base_keys
        if only_base:
            logger.info(f"  ft[{i}] missing {len(only_base)} keys (will keep base value)")
        if only_ft:
            logger.info(f"  ft[{i}] has {len(only_ft)} extra keys (will be ignored)")

    # 显式选定融合参数白名单 (preset)。
    presets = {
        "pi05_ppo": PI05_PPO_TRAINABLE_PREFIXES,  # RLinf train_expert_only=True 下实际可训练的 208 个 key
        "all":      None,                          # 不限制（包含 backbone）
    }
    if include_preset not in presets:
        raise ValueError(f"include_preset must be one of {list(presets)}, got {include_preset!r}")
    include_prefixes = presets[include_preset]
    n_eligible = sum(
        1 for k in base_sd
        if (include_prefixes is None or any(k.startswith(p) for p in include_prefixes))
    )
    logger.info(
        f"Computing task_arithmetic alpha={alpha} preset={include_preset} "
        f"(eligible={n_eligible}/{len(base_sd)} keys) exclude={exclude_param_names_regex}"
    )
    merged = task_arithmetic(
        pretrained_state_dict=base_sd,
        finetuned_state_dicts=ft_sds,
        exclude_param_names_regex=exclude_param_names_regex,
        scaling_coefficient=alpha,
        include_param_names_prefixes=include_prefixes,
    )

    # 统计实际改变的参数数量（用于 sanity check）
    n_total = len(merged)
    n_diff = 0
    for k, v in merged.items():
        if k in base_sd and not torch.equal(v.to(torch.float32), base_sd[k].to(torch.float32)):
            n_diff += 1
    logger.info(f"  changed {n_diff} / {n_total} tensors vs base")

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32, "none": None}
    cast = dtype_map[cast_dtype]
    save_merged_checkpoint(merged, output_path, base_dir=base_path, cast_dtype=cast)
    logger.info(f"Done: {output_path}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ap = argparse.ArgumentParser(description="Task Arithmetic merging for PyTorch safetensors")
    ap.add_argument("--base", required=True, help="Pretrained/SFT checkpoint dir (contains model.safetensors)")
    ap.add_argument("--ft", required=True, nargs="+", help="Fine-tuned checkpoint dir(s)")
    ap.add_argument("--output", required=True, help="Output dir for merged ckpt")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="Scaling coefficient. theta* = theta_base + alpha * sum(tau_i). "
                         "alpha=0 → base, alpha=1 (N=1) → ft, alpha in (0,1) → linear interp.")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="Regex patterns of param names to exclude from merging "
                         "(those params keep base values). E.g. --exclude '.*value_head.*'")
    ap.add_argument("--cast-dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32", "none"],
                    help="Cast merged tensors to this dtype before saving (default bf16 to match base_25k)")
    ap.add_argument("--include-preset", default="pi05_ppo",
                    choices=["pi05_ppo", "all"],
                    help="显式融合参数白名单。 "
                         "'pi05_ppo' (默认): 只融合 RL 实际训练过的 action expert + projections（208 keys）。 "
                         "'all': 所有 key 都参与（与未指定 include 时一致；冻结参数差为 0，结果等价）。")
    args = ap.parse_args()

    run_merge(
        base_path=args.base,
        ft_paths=args.ft,
        output_path=args.output,
        alpha=args.alpha,
        exclude_param_names_regex=args.exclude,
        cast_dtype=args.cast_dtype,
        include_preset=args.include_preset,
    )


if __name__ == "__main__":
    main()
