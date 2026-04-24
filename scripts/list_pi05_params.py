#!/usr/bin/env python3
"""List and categorize pi0.5 checkpoint parameter paths.

Reads ONLY the checkpoint _METADATA file (no weight loading → no OOM).

Usage
─────
    # From real checkpoint (reads _METADATA only, fast + no OOM):
    python scripts/list_pi05_params.py \
        --ckpt /storage/yukaichengLab/lishiwen/jiayusun/openpi_pt/pi05_model/pi05_base

    # Dry-run (architecture-derived paths, no checkpoint needed):
    python scripts/list_pi05_params.py --dry-run [--pi0]

Output categories
─────────────────
  [SIGLIP]    PaliGemma/img/      — SigLIP ViT vision encoder  (27 layers, scan)
  [PALI-LLM]  PaliGemma/llm/      — PaliGemma 2B language expert (18 layers)
  [ACT-EXP]   PaliGemma/llm/ _1   — Action Expert 300M  (18 layers, adaRMSNorm)
  [FLOW]      action_*/time_*      — Flow matching I/O & time MLP
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

# ─────────────────────────────────────────────────────────────────────────────
# Category classifiers  (work on flat path strings)
# ─────────────────────────────────────────────────────────────────────────────

_EXPERT1_RE = re.compile(
    r"^(attn|mlp|pre_attention_norm|pre_ffw_norm|final_norm|"
    r"q_einsum|kv_einsum|attn_vec_einsum|gating_einsum|linear|Dense)_(\d+)$"
)


def classify(key: str) -> str:
    """Assign a category tag to a flat checkpoint key (no leading 'params/')."""
    if key.startswith("PaliGemma/img/"):
        return "SIGLIP"
    if key.startswith(("action_in_proj/", "action_out_proj/",
                        "time_mlp_in/",   "time_mlp_out/",
                        "state_proj/",
                        "action_time_mlp_in/", "action_time_mlp_out/")):
        return "FLOW"
    if key.startswith("PaliGemma/llm/"):
        # Action-expert params carry a _N (N≥1) suffix on the module segment.
        for seg in key.split("/"):
            m = _EXPERT1_RE.fullmatch(seg)
            if m and int(m.group(2)) >= 1:
                return "ACT-EXP"
        return "PALI-LLM"
    return "OTHER"


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint metadata reader  (reads _METADATA JSON, no weights)
# ─────────────────────────────────────────────────────────────────────────────

def read_metadata(ckpt_root: str | pathlib.Path) -> list[tuple[str, list[int]]]:
    """Parse Orbax _METADATA and return [(flat_key, shape), ...]."""
    root = pathlib.Path(ckpt_root).resolve()
    # Accept either the root dir or the params/ subdir
    candidates = [root / "params" / "_METADATA", root / "_METADATA"]
    meta_file = next((p for p in candidates if p.exists()), None)
    if meta_file is None:
        raise FileNotFoundError(f"Cannot find _METADATA under {root}")

    print(f"[info] Reading metadata: {meta_file}")
    data = json.loads(meta_file.read_text())

    rows: list[tuple[str, list[int]]] = []
    for raw_key, entry in data["tree_metadata"].items():
        segs = [x["key"] for x in entry["key_metadata"]]
        # Strip leading "params" key added by Orbax
        if segs and segs[0] == "params":
            segs = segs[1:]
        flat_key = "/".join(str(s) for s in segs)
        shape: list[int] = entry["value_metadata"].get("write_shape", [])
        rows.append((flat_key, shape))

    rows.sort(key=lambda x: x[0])
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Architecture-derived paths for dry-run  (verified against pi05_base metadata)
# ─────────────────────────────────────────────────────────────────────────────

def _siglip_paths() -> list[tuple[str, list[int]]]:
    """SigLIP ViT So400m/14 parameter paths.

    All 27-layer arrays use nn.scan → single tensor with shape[0]=27.
    """
    b = "PaliGemma/img"
    T = "PaliGemma/img/Transformer"
    E = f"{T}/encoderblock"   # scan-merged, shape[0]=27
    return [
        # ── Patch embedding (stem conv) ───────────────────────────────────
        (f"{b}/embedding/kernel",     [14, 14, 3, 1152]),  # Conv2d(3→1152, k=14)
        (f"{b}/embedding/bias",       [1152]),
        # ── 2D SinCos positional embedding (stored in ckpt) ──────────────
        (f"{b}/pos_embedding",        [1, 256, 1152]),
        # ── 27 × Transformer encoder block (scan) ────────────────────────
        (f"{E}/LayerNorm_0/scale",    [27, 1152]),  # pre-attn LN
        (f"{E}/LayerNorm_0/bias",     [27, 1152]),
        (f"{E}/MultiHeadDotProductAttention_0/query/kernel", [27, 1152, 16, 72]),
        (f"{E}/MultiHeadDotProductAttention_0/query/bias",   [27, 16, 72]),
        (f"{E}/MultiHeadDotProductAttention_0/key/kernel",   [27, 1152, 16, 72]),
        (f"{E}/MultiHeadDotProductAttention_0/key/bias",     [27, 16, 72]),
        (f"{E}/MultiHeadDotProductAttention_0/value/kernel", [27, 1152, 16, 72]),
        (f"{E}/MultiHeadDotProductAttention_0/value/bias",   [27, 16, 72]),
        (f"{E}/MultiHeadDotProductAttention_0/out/kernel",   [27, 16, 72, 1152]),
        (f"{E}/MultiHeadDotProductAttention_0/out/bias",     [27, 1152]),
        (f"{E}/LayerNorm_1/scale",    [27, 1152]),  # pre-FFN LN
        (f"{E}/LayerNorm_1/bias",     [27, 1152]),
        (f"{E}/MlpBlock_0/Dense_0/kernel", [27, 1152, 4304]),  # fc1  1152→4304
        (f"{E}/MlpBlock_0/Dense_0/bias",   [27, 4304]),
        (f"{E}/MlpBlock_0/Dense_1/kernel", [27, 4304, 1152]),  # fc2  4304→1152
        (f"{E}/MlpBlock_0/Dense_1/bias",   [27, 1152]),
        # ── Final LayerNorm (after all 27 blocks) ─────────────────────────
        (f"{T}/encoder_norm/scale",   [1152]),
        (f"{T}/encoder_norm/bias",    [1152]),
        # ── Head projection (1152 → 2048, align to PaliGemma hidden dim) ──
        (f"{b}/head/kernel",          [1152, 2048]),
        (f"{b}/head/bias",            [2048]),
    ]


def _paligemma_llm_paths() -> list[tuple[str, list[int]]]:
    """PaliGemma 2B language expert paths (Gemma 2B, 18 layers, RMSNorm)."""
    base = "PaliGemma/llm"
    L = f"{base}/layers"
    return [
        # ── Token embedder ────────────────────────────────────────────────
        (f"{base}/embedder/input_embedding",   [257152, 2048]),
        # ── 18 × Block attention params (scan, shape[0]=18) ──────────────
        (f"{L}/pre_attention_norm/scale",      [18, 2048]),          # RMSNorm
        (f"{L}/attn/q_einsum/w",               [18, 8, 2048, 256]),  # Q: 8 heads, dim→256
        (f"{L}/attn/kv_einsum/w",              [18, 2, 1, 2048, 256]),# KV: GQA 1 head
        (f"{L}/attn/attn_vec_einsum/w",        [18, 8, 256, 2048]),  # output proj
        # ── 18 × Block FFN params (GeGLU) ────────────────────────────────
        (f"{L}/pre_ffw_norm/scale",            [18, 2048]),          # RMSNorm
        (f"{L}/mlp/gating_einsum",             [18, 2, 2048, 16384]),# gate+up packed
        (f"{L}/mlp/linear",                    [18, 16384, 2048]),   # down proj
        # ── Final RMSNorm ─────────────────────────────────────────────────
        (f"{base}/final_norm/scale",           [2048]),
    ]


def _action_expert_paths() -> list[tuple[str, list[int]]]:
    """Action Expert 300M paths (_1 suffix, adaRMSNorm for π0.5, 18 layers)."""
    base = "PaliGemma/llm"
    L = f"{base}/layers"
    return [
        # ── 18 × Block attention params (scan, shape[0]=18) ──────────────
        # adaRMSNorm: Dense_0 maps time_emb (1024) → 3×dim (scale/shift/gate)
        (f"{L}/pre_attention_norm_1/Dense_0/kernel", [18, 1024, 3072]),
        (f"{L}/pre_attention_norm_1/Dense_0/bias",   [18, 3072]),
        (f"{L}/attn/q_einsum_1/w",                   [18, 8, 1024, 256]),   # Q expert
        (f"{L}/attn/kv_einsum_1/w",                  [18, 2, 1, 1024, 256]),# KV expert
        (f"{L}/attn/attn_vec_einsum_1/w",            [18, 8, 256, 1024]),   # O expert
        # ── 18 × Block FFN params ─────────────────────────────────────────
        (f"{L}/pre_ffw_norm_1/Dense_0/kernel",       [18, 1024, 3072]),
        (f"{L}/pre_ffw_norm_1/Dense_0/bias",         [18, 3072]),
        (f"{L}/mlp_1/gating_einsum",                 [18, 2, 1024, 4096]),  # gate+up
        (f"{L}/mlp_1/linear",                        [18, 4096, 1024]),     # down
        # ── Final adaRMSNorm ──────────────────────────────────────────────
        (f"{base}/final_norm_1/Dense_0/kernel",      [1024, 3072]),
        (f"{base}/final_norm_1/Dense_0/bias",        [3072]),
    ]


def _flow_paths(pi05: bool = True) -> list[tuple[str, list[int]]]:
    """Flow-matching and action I/O projection paths.

    pi05=True  → π0.5: time_mlp_in / time_mlp_out, no state_proj.
    pi05=False → π0:   action_time_mlp_in/_out, state_proj.
    """
    rows: list[tuple[str, list[int]]] = [
        # ── Action token projection (both π0 and π0.5) ────────────────────
        ("action_in_proj/kernel",  [32, 1024]),   # Linear(32→1024)
        ("action_in_proj/bias",    [1024]),
        ("action_out_proj/kernel", [1024, 32]),   # Linear(1024→32)
        ("action_out_proj/bias",   [32]),
    ]
    if pi05:
        # π0.5: time MLP → adaRMSNorm conditioning
        rows += [
            ("time_mlp_in/kernel",  [1024, 1024]),  # Linear(1024→1024)
            ("time_mlp_in/bias",    [1024]),
            ("time_mlp_out/kernel", [1024, 1024]),  # Linear(1024→1024)
            ("time_mlp_out/bias",   [1024]),
        ]
    else:
        # π0: time MLP fuses timestep into action tokens before Transformer
        rows += [
            ("action_time_mlp_in/kernel",  [2048, 1024]),  # Linear(2048→1024)
            ("action_time_mlp_in/bias",    [1024]),
            ("action_time_mlp_out/kernel", [1024, 1024]),  # Linear(1024→1024)
            ("action_time_mlp_out/bias",   [1024]),
            ("state_proj/kernel",          [32, 1024]),    # Linear(32→1024) π0 only
            ("state_proj/bias",            [1024]),
        ]
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printing
# ─────────────────────────────────────────────────────────────────────────────

_COLORS = {
    "SIGLIP":   "\033[94m",   # blue
    "PALI-LLM": "\033[92m",   # green
    "ACT-EXP":  "\033[93m",   # yellow
    "FLOW":     "\033[95m",   # magenta
    "OTHER":    "\033[91m",   # red
}
_RESET = "\033[0m"

_LABELS = {
    "SIGLIP":   "SigLIP ViT  (Vision Encoder, 27 layers, PaliGemma/img/)",
    "PALI-LLM": "PaliGemma Language Expert  (Gemma 2B, 18 layers, RMSNorm)",
    "ACT-EXP":  "Action Expert  (Gemma 300M, 18 layers, adaRMSNorm, _1 suffix)",
    "FLOW":     "Flow Matching I/O  (action_in/out_proj, time_mlp_in/out)",
    "OTHER":    "Other / Unknown",
}


def print_section(
    category: str,
    entries: list[tuple[str, list[int]]],
    use_color: bool,
) -> None:
    color = _COLORS.get(category, "") if use_color else ""
    reset = _RESET if use_color else ""
    label = _LABELS.get(category, category)
    print(f"\n{color}{'═'*72}")
    print(f"  [{category}]  {label}")
    print(f"{'═'*72}{reset}")
    for key, shape in entries:
        shape_str = f"  {shape}" if shape else ""
        print(f"  {key}{shape_str}")
    print(f"  → {len(entries)} parameter tensors")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="List and categorize pi0/pi0.5 checkpoint parameter paths",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--ckpt", type=str, default=None,
        help="Checkpoint root (dir containing params/_METADATA). "
             "Reads metadata only — no weight loading."
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Print architecture-derived paths without reading a checkpoint"
    )
    ap.add_argument(
        "--pi0", action="store_true",
        help="Use π0 flow paths instead of π0.5 (only affects --dry-run output)"
    )
    ap.add_argument("--no-color", action="store_true", help="Disable ANSI color")
    args = ap.parse_args()

    use_color = not args.no_color
    pi05 = not args.pi0

    # ── Real checkpoint: parse _METADATA ─────────────────────────────────────
    if args.ckpt and not args.dry_run:
        rows = read_metadata(args.ckpt)
        buckets: dict[str, list[tuple[str, list[int]]]] = {
            "SIGLIP": [], "PALI-LLM": [], "ACT-EXP": [], "FLOW": [], "OTHER": [],
        }
        for key, shape in rows:
            buckets[classify(key)].append((key, shape))

        print(f"\n[info] Total parameter tensors: {len(rows)}")
        for cat in ("SIGLIP", "PALI-LLM", "ACT-EXP", "FLOW", "OTHER"):
            if buckets[cat]:
                print_section(cat, buckets[cat], use_color)
        return

    # ── Dry-run: architecture-derived paths ───────────────────────────────────
    model_name = "π0.5" if pi05 else "π0"
    buckets_dry: dict[str, list[tuple[str, list[int]]]] = {
        "SIGLIP":   _siglip_paths(),
        "PALI-LLM": _paligemma_llm_paths(),
        "ACT-EXP":  _action_expert_paths(),
        "FLOW":     _flow_paths(pi05=pi05),
    }
    total = sum(len(v) for v in buckets_dry.values())
    print(f"\n[dry-run] Architecture-derived paths for {model_name}  ({total} tensors)")
    for cat in ("SIGLIP", "PALI-LLM", "ACT-EXP", "FLOW"):
        print_section(cat, buckets_dry[cat], use_color)


if __name__ == "__main__":
    main()
