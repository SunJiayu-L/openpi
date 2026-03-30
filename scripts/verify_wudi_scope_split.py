#!/usr/bin/env python3
"""Verify WUDI key-splitting logic and print action/language parameter groups.

This script reads Orbax checkpoint `_METADATA` (no tensor loading), converts keys to the
same flat path style used by `scripts/wudi_merge.py`, and prints:
1) action expert keys (expert1)
2) language expert keys (expert0)

It focuses on attention/FFN keys, and can optionally print all in-scope keys.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.wudi_merge import _in_scope, _is_attn_ffn, _is_expert1, _is_vision


def load_flat_keys_from_metadata(metadata_path: Path) -> list[str]:
    with metadata_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    keys: list[str] = []
    for raw in data["tree_metadata"].keys():
        tup = ast.literal_eval(raw)
        if not tup or tup[0] != "params":
            continue
        parts = list(tup[1:])
        if parts and parts[-1] == "value":
            parts = parts[:-1]
        keys.append("/".join(parts))
    return sorted(set(keys))


def print_group(title: str, keys: list[str], limit: int) -> None:
    print(f"\n[{title}] count={len(keys)}")
    for k in keys[:limit]:
        print(f"  {k}")
    if len(keys) > limit:
        print(f"  ... ({len(keys) - limit} more)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify action/language split for WUDI merge logic")
    ap.add_argument(
        "--params",
        type=str,
        required=True,
        help="Path to checkpoint params dir, e.g. checkpoints/pi05_libero/my_experiment/29999/params",
    )
    ap.add_argument("--scope", type=str, default="expert1_only", choices=["expert1_only", "both_experts", "llm_only"])
    ap.add_argument("--limit", type=int, default=80, help="Max keys to print per group")
    ap.add_argument("--show-all-in-scope", action="store_true", help="Also print all in-scope keys (not only attn/ffn)")
    args = ap.parse_args()

    params_dir = Path(args.params)
    metadata_path = params_dir / "_METADATA"
    if not metadata_path.exists():
        raise FileNotFoundError(f"_METADATA not found: {metadata_path}")

    keys = load_flat_keys_from_metadata(metadata_path)

    in_scope = [k for k in keys if _in_scope(k, args.scope)]
    attn_ffn = [k for k in keys if _is_attn_ffn(k)]
    attn_ffn_in_scope = [k for k in attn_ffn if _in_scope(k, args.scope)]

    action_attn_ffn = [k for k in attn_ffn_in_scope if _is_expert1(k)]
    language_attn_ffn = [k for k in attn_ffn_in_scope if not _is_expert1(k)]

    vision_in_scope = [k for k in in_scope if _is_vision(k)]

    print("=== WUDI scope split verification ===")
    print(f"params: {params_dir}")
    print(f"scope: {args.scope}")
    print(f"total keys: {len(keys)}")
    print(f"in-scope keys: {len(in_scope)}")
    print(f"attn/ffn keys total: {len(attn_ffn)}")
    print(f"attn/ffn keys in-scope: {len(attn_ffn_in_scope)}")
    print(f"action attn/ffn (expert1): {len(action_attn_ffn)}")
    print(f"language attn/ffn (expert0): {len(language_attn_ffn)}")
    print(f"vision keys in-scope: {len(vision_in_scope)}")

    print_group("Action attn/ffn keys", action_attn_ffn, args.limit)
    print_group("Language attn/ffn keys", language_attn_ffn, args.limit)

    if args.show_all_in_scope:
        print_group("All in-scope keys", in_scope, args.limit)


if __name__ == "__main__":
    main()
