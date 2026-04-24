#!/usr/bin/env python3
"""Filter pi0.5 checkpoint parameters into mergeable / excluded groups.

Applies the same exclusion logic as MLLMerging's exclude_param_names_regex,
mapped to openpi's parameter naming convention:

  MLLMerging exclusion          openpi equivalent
  ─────────────────────────     ─────────────────────────────────────────
  vision_model.*                PaliGemma/img/*  (SigLIP ViT)
  .*embed_tokens.*              PaliGemma/llm/embedder/*  (token embedding)
  .*lm_head.*                   action_in_proj/ action_out_proj/
                                time_mlp_in/    time_mlp_out/
  .*norm.*                      any segment containing "norm"
  .*bias.*                      leaf name == "bias"

Usage:
    # From real checkpoint (reads _METADATA only):
    python scripts/filter_merge_params.py \
        --ckpt /storage/yukaichengLab/lishiwen/jiayusun/openpi_pt/pi05_model/pi05_base

    # Dry-run (architecture-derived paths):
    python scripts/filter_merge_params.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re

# ─────────────────────────────────────────────────────────────────────────────
# Exclusion rules
# ─────────────────────────────────────────────────────────────────────────────

# Rule → human-readable reason
_RULES: list[tuple[re.Pattern, str]] = [
    # 1. Vision model (SigLIP ViT entire encoder)
    (re.compile(r"^PaliGemma/img/"),
     "vision_model (SigLIP ViT)"),

    # 2. Token embedding table
    (re.compile(r"^PaliGemma/llm/embedder/"),
     "embed_tokens (token embedding table)"),

    # 3. Flow matching I/O & time MLP  (≈ lm_head in MLLMerging)
    (re.compile(r"^(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out)/"),
     "lm_head-equivalent (flow matching I/O / time MLP)"),

    # 4. Any parameter whose path contains "norm" (covers all Norm layers)
    #    Examples: pre_attention_norm, pre_ffw_norm, final_norm, encoder_norm, LayerNorm
    (re.compile(r"(^|/)norm(/|$)|norm\d*(/|$)|(LayerNorm|RMSNorm|encoder_norm|"
                r"pre_attention_norm|pre_ffw_norm|final_norm|attention_norm|ffn_norm)"),
     "norm layer (RMSNorm / LayerNorm / adaRMSNorm)"),

    # 5. Any bias parameter (leaf segment == "bias")
    (re.compile(r"(^|/)bias$"),
     "bias term"),
]


def exclusion_reason(key: str) -> str | None:
    """Return the reason string if key is excluded, else None."""
    for pattern, reason in _RULES:
        if pattern.search(key):
            return reason
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint metadata reader
# ─────────────────────────────────────────────────────────────────────────────

def read_metadata(ckpt_root: str | pathlib.Path) -> list[tuple[str, list[int]]]:
    root = pathlib.Path(ckpt_root).resolve()
    candidates = [root / "params" / "_METADATA", root / "_METADATA"]
    meta_file = next((p for p in candidates if p.exists()), None)
    if meta_file is None:
        raise FileNotFoundError(f"Cannot find _METADATA under {root}")

    print(f"[info] Reading metadata: {meta_file}")
    data = json.loads(meta_file.read_text())

    rows: list[tuple[str, list[int]]] = []
    for _, entry in data["tree_metadata"].items():
        segs = [x["key"] for x in entry["key_metadata"]]
        if segs and segs[0] == "params":
            segs = segs[1:]
        flat_key = "/".join(str(s) for s in segs)
        shape: list[int] = entry["value_metadata"].get("write_shape", [])
        rows.append((flat_key, shape))

    rows.sort(key=lambda x: x[0])
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Architecture-derived paths (dry-run)
# ─────────────────────────────────────────────────────────────────────────────

def _all_paths_dry() -> list[tuple[str, list[int]]]:
    """Return all pi0.5 parameter paths derived from architecture knowledge."""
    b = "PaliGemma/img"
    T = f"{b}/Transformer"
    E = f"{T}/encoderblock"
    L = "PaliGemma/llm/layers"
    llm = "PaliGemma/llm"

    return [
        # ── SigLIP ────────────────────────────────────────────────────────
        (f"{b}/embedding/kernel",     [14, 14, 3, 1152]),
        (f"{b}/embedding/bias",       [1152]),
        (f"{b}/pos_embedding",        [1, 256, 1152]),
        (f"{E}/LayerNorm_0/scale",    [27, 1152]),
        (f"{E}/LayerNorm_0/bias",     [27, 1152]),
        (f"{E}/MultiHeadDotProductAttention_0/query/kernel", [27, 1152, 16, 72]),
        (f"{E}/MultiHeadDotProductAttention_0/query/bias",   [27, 16, 72]),
        (f"{E}/MultiHeadDotProductAttention_0/key/kernel",   [27, 1152, 16, 72]),
        (f"{E}/MultiHeadDotProductAttention_0/key/bias",     [27, 16, 72]),
        (f"{E}/MultiHeadDotProductAttention_0/value/kernel", [27, 1152, 16, 72]),
        (f"{E}/MultiHeadDotProductAttention_0/value/bias",   [27, 16, 72]),
        (f"{E}/MultiHeadDotProductAttention_0/out/kernel",   [27, 16, 72, 1152]),
        (f"{E}/MultiHeadDotProductAttention_0/out/bias",     [27, 1152]),
        (f"{E}/LayerNorm_1/scale",    [27, 1152]),
        (f"{E}/LayerNorm_1/bias",     [27, 1152]),
        (f"{E}/MlpBlock_0/Dense_0/kernel", [27, 1152, 4304]),
        (f"{E}/MlpBlock_0/Dense_0/bias",   [27, 4304]),
        (f"{E}/MlpBlock_0/Dense_1/kernel", [27, 4304, 1152]),
        (f"{E}/MlpBlock_0/Dense_1/bias",   [27, 1152]),
        (f"{T}/encoder_norm/scale",   [1152]),
        (f"{T}/encoder_norm/bias",    [1152]),
        (f"{b}/head/kernel",          [1152, 2048]),
        (f"{b}/head/bias",            [2048]),
        # ── PaliGemma LLM ─────────────────────────────────────────────────
        (f"{llm}/embedder/input_embedding",          [257152, 2048]),
        (f"{llm}/final_norm/scale",                  [2048]),
        (f"{L}/pre_attention_norm/scale",            [18, 2048]),
        (f"{L}/attn/q_einsum/w",                     [18, 8, 2048, 256]),
        (f"{L}/attn/kv_einsum/w",                    [18, 2, 1, 2048, 256]),
        (f"{L}/attn/attn_vec_einsum/w",              [18, 8, 256, 2048]),
        (f"{L}/pre_ffw_norm/scale",                  [18, 2048]),
        (f"{L}/mlp/gating_einsum",                   [18, 2, 2048, 16384]),
        (f"{L}/mlp/linear",                          [18, 16384, 2048]),
        # ── Action Expert ─────────────────────────────────────────────────
        (f"{llm}/final_norm_1/Dense_0/kernel",       [1024, 3072]),
        (f"{llm}/final_norm_1/Dense_0/bias",         [3072]),
        (f"{L}/pre_attention_norm_1/Dense_0/kernel", [18, 1024, 3072]),
        (f"{L}/pre_attention_norm_1/Dense_0/bias",   [18, 3072]),
        (f"{L}/attn/q_einsum_1/w",                   [18, 8, 1024, 256]),
        (f"{L}/attn/kv_einsum_1/w",                  [18, 2, 1, 1024, 256]),
        (f"{L}/attn/attn_vec_einsum_1/w",            [18, 8, 256, 1024]),
        (f"{L}/pre_ffw_norm_1/Dense_0/kernel",       [18, 1024, 3072]),
        (f"{L}/pre_ffw_norm_1/Dense_0/bias",         [18, 3072]),
        (f"{L}/mlp_1/gating_einsum",                 [18, 2, 1024, 4096]),
        (f"{L}/mlp_1/linear",                        [18, 4096, 1024]),
        # ── Flow matching I/O ─────────────────────────────────────────────
        ("action_in_proj/kernel",   [32, 1024]),
        ("action_in_proj/bias",     [1024]),
        ("action_out_proj/kernel",  [1024, 32]),
        ("action_out_proj/bias",    [32]),
        ("time_mlp_in/kernel",      [1024, 1024]),
        ("time_mlp_in/bias",        [1024]),
        ("time_mlp_out/kernel",     [1024, 1024]),
        ("time_mlp_out/bias",       [1024]),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printing
# ─────────────────────────────────────────────────────────────────────────────

_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_RESET  = "\033[0m"


def print_results(
    mergeable: list[tuple[str, list[int]]],
    excluded:  list[tuple[str, list[int], str]],
    use_color: bool,
) -> None:
    g = _GREEN  if use_color else ""
    r = _RED    if use_color else ""
    y = _YELLOW if use_color else ""
    rst = _RESET if use_color else ""

    # ── Mergeable ──────────────────────────────────────────────────────────
    print(f"\n{g}{'═'*72}")
    print(f"  [MERGEABLE]  参数将被 WUDI 融合")
    print(f"{'═'*72}{rst}")
    for key, shape in mergeable:
        print(f"  {key}  {shape}")
    print(f"\n  → 共 {len(mergeable)} 个参数张量")

    # ── Excluded by reason ─────────────────────────────────────────────────
    # Group by reason
    reason_groups: dict[str, list[tuple[str, list[int]]]] = {}
    for key, shape, reason in excluded:
        reason_groups.setdefault(reason, []).append((key, shape))

    print(f"\n{r}{'═'*72}")
    print(f"  [EXCLUDED]  参数不参与融合（保持 base 值不变）")
    print(f"{'═'*72}{rst}")
    for reason, items in reason_groups.items():
        print(f"\n  {y}── {reason}{rst}")
        for key, shape in items:
            print(f"     {key}  {shape}")
        print(f"     ({len(items)} 个)")

    total_excl = len(excluded)
    total_all  = len(mergeable) + total_excl
    print(f"\n  → 共 {total_excl} 个参数张量被排除")

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"  汇总: 总计 {total_all} 个张量")
    print(f"        可融合  {len(mergeable):2d} 个  ({100*len(mergeable)/total_all:.1f}%)")
    print(f"        排除    {total_excl:2d} 个  ({100*total_excl/total_all:.1f}%)")
    print(f"{'─'*72}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Filter pi0.5 params into mergeable / excluded",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--ckpt", type=str, default=None,
                    help="Checkpoint root (reads _METADATA only, no weights loaded)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Use architecture-derived paths (no checkpoint needed)")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()

    use_color = not args.no_color

    # Get parameter list
    if args.ckpt and not args.dry_run:
        rows = read_metadata(args.ckpt)
    else:
        rows = _all_paths_dry()
        print("[dry-run] Using architecture-derived paths")

    # Apply exclusion rules
    mergeable: list[tuple[str, list[int]]]        = []
    excluded:  list[tuple[str, list[int], str]]   = []

    for key, shape in rows:
        reason = exclusion_reason(key)
        if reason:
            excluded.append((key, shape, reason))
        else:
            mergeable.append((key, shape))

    print_results(mergeable, excluded, use_color)


if __name__ == "__main__":
    main()
