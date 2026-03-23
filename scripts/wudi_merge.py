#!/usr/bin/env python3
"""WUDI merging for pi0.5 with per-layer 2D decomposition and SVD de-centering.

Algorithm: get_redundant_task_vector from wudi_merging2 (SVD de-centering variant).
  For each 2D sub-block (per layer, L looped):
    1. Compute average task vector across models
    2. Per task: full SVD of original → masked low_rank basis
                 compact SVD of (vector - avg) → de-centered task reference
    3. Adam optimize merging vector to minimize projection onto low_rank subspace

Only Gemma attention and FFN parameters are WUDI-merged;
all other parameters keep base values unchanged.

Parameter 2D decomposition (per layer, L dimension looped over):
  q_einsum/w        (L,N,D,H)   → per layer (N,D,H)   → reshape(N*D, H)
  kv_einsum/w       (L,2,K,D,H) → per layer (2,K,D,H) → split K/V → each reshape(D, K*H)
  attn_vec_einsum/w (L,N,H,D)   → per layer (N,H,D)   → reshape(N*H, D)
  gating_einsum     (L,2,D,Hff) → per layer (2,D,Hff) → split gate/value → each (D,Hff)
  linear            (L,Hff,D)   → per layer (Hff,D)   → already 2D

Usage:
    # Smoke test (CPU, tiny matrices):
    python scripts/wudi_merge.py --test

    # Merge (action expert only):
    python scripts/wudi_merge.py \\
        --base   /path/to/pi05_base/params \\
        --ft     /path/to/ft1/params /path/to/ft2/params \\
        --output checkpoints/merged/wudi_e1_libero_goal \\
        --scope  expert1_only \\
        --iter   300 \\
        --scaling 1.0 \\
        --device cuda
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import re
import shutil
import sys

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parameter identification
# ---------------------------------------------------------------------------

_ATTN_FFN_RE = re.compile(
    r".*/(q_einsum|kv_einsum|attn_vec_einsum)(_\d+)?/w$"
    r"|.*/gating_einsum(_\d+)?$"
    r"|.*/linear(_\d+)?$"
)


def _param_type(key: str) -> str | None:
    """Return type tag: 'q', 'kv', 'av', 'gate', 'lin', or None."""
    segs = key.split("/")
    last = segs[-1]
    parent = segs[-2] if len(segs) >= 2 else ""
    if last == "w":
        if re.search(r"q_einsum", parent) and not re.search(r"kv_einsum", parent):
            return "q"
        if re.search(r"kv_einsum", parent):
            return "kv"
        if re.search(r"attn_vec_einsum", parent):
            return "av"
        return None
    if re.match(r"gating_einsum(_\d+)?$", last):
        return "gate"
    if re.match(r"linear(_\d+)?$", last):
        return "lin"
    return None


def _is_expert1(key: str) -> bool:
    """True if key belongs to action expert (expert index >= 1)."""
    segs = key.split("/")
    # parent of /w has _1 suffix  e.g. q_einsum_1/w
    if len(segs) >= 2 and re.search(r"_1$", segs[-2]):
        return True
    # any path segment is mlp_N with N >= 1
    for seg in segs:
        m = re.fullmatch(r"mlp_(\d+)", seg)
        if m and int(m.group(1)) >= 1:
            return True
    return False


def _is_attn_ffn(key: str) -> bool:
    return bool(_ATTN_FFN_RE.match(key))


def _in_scope(key: str, scope: str) -> bool:
    if scope == "expert1_only":
        return _is_expert1(key)
    if scope == "both_experts":
        return True
    raise ValueError(f"Unknown scope: {scope!r}")


# ---------------------------------------------------------------------------
# 2D decomposition / composition (operates on a single layer, no L dim)
# ---------------------------------------------------------------------------

def decompose_layer(ptype: str, layer: np.ndarray) -> list[np.ndarray]:
    """Split one layer's tensor into 2D sub-blocks.

    Args:
        ptype: one of 'q', 'kv', 'av', 'gate', 'lin'
        layer: numpy array without the leading L dimension

    Returns:
        list of 2D numpy arrays
    """
    if ptype == "q":
        # (N, D, H) → (N*D, H)
        N, D, H = layer.shape
        return [layer.reshape(N * D, H)]

    if ptype == "kv":
        # (2, K, D, H) → split K / V → each (K, D, H) → (D, K*H)
        _, K, D, H = layer.shape
        k_block = layer[0].transpose(1, 0, 2).reshape(D, K * H)  # (D, K*H)
        v_block = layer[1].transpose(1, 0, 2).reshape(D, K * H)
        return [k_block, v_block]

    if ptype == "av":
        # (N, H, D) → (N*H, D)
        N, H, D = layer.shape
        return [layer.reshape(N * H, D)]

    if ptype == "gate":
        # (2, D, Hff) → gate (D, Hff), value (D, Hff)
        assert layer.shape[0] == 2
        return [layer[0], layer[1]]

    if ptype == "lin":
        # (Hff, D) — already 2D
        assert layer.ndim == 2
        return [layer]

    raise ValueError(f"Unknown ptype: {ptype!r}")


def compose_layer(ptype: str, sub_blocks: list[np.ndarray], original_shape: tuple) -> np.ndarray:
    """Reconstruct a single-layer tensor from merged 2D sub-blocks."""
    if ptype == "q":
        N, D, H = original_shape
        return sub_blocks[0].reshape(N, D, H)

    if ptype == "kv":
        _, K, D, H = original_shape
        k = sub_blocks[0].reshape(D, K, H).transpose(1, 0, 2)  # (K, D, H)
        v = sub_blocks[1].reshape(D, K, H).transpose(1, 0, 2)
        return np.stack([k, v], axis=0)  # (2, K, D, H)

    if ptype == "av":
        N, H, D = original_shape
        return sub_blocks[0].reshape(N, H, D)

    if ptype == "gate":
        return np.stack([sub_blocks[0], sub_blocks[1]], axis=0)  # (2, D, Hff)

    if ptype == "lin":
        return sub_blocks[0]

    raise ValueError(f"Unknown ptype: {ptype!r}")


# ---------------------------------------------------------------------------
# WUDI optimization (SVD de-centering variant, ported from wudi_merging2)
# ---------------------------------------------------------------------------

# Use compact SVD when matrix element count exceeds this to avoid OOM
_OOM_THRESHOLD = 8_000_000


def wudi_optimize(
    label: str,
    vectors: "torch.Tensor",  # (T, m, n) float32 on any device
    iter_num: int = 300,
    device: str = "cuda",
) -> "torch.Tensor":
    """Minimize inter-task interference for a single 2D sub-block.

    Implements SVD de-centering variant of get_redundant_task_vector.

    Args:
        label:    human-readable name for logging
        vectors:  stacked task vectors, shape (T, m, n)
        iter_num: Adam optimization steps
        device:   'cuda' or 'cpu'

    Returns:
        Optimized merged task vector, shape (m, n), same dtype as input.
    """
    import torch

    original_dtype = vectors.dtype
    vectors = vectors.float().to(device)
    T, m, n = vectors.shape
    use_compact = (m * n > _OOM_THRESHOLD)

    average_vector = vectors.mean(dim=0)  # (m, n)
    low_rank_list: list[torch.Tensor] = []
    taskvector_list: list[torch.Tensor] = []

    for i in range(T):
        vector = vectors[i]  # (m, n)

        # --- SVD of original task vector → low_rank basis ---
        if use_compact:
            # compact SVD: avoid large V for wide matrices
            _, s, v = torch.linalg.svd(vector, full_matrices=False)
            # s: (min_dim,), v: (min_dim, n)
            min_dim = s.shape[0]
            reduced_r = max(1, min_dim // T)
            s_masked = s.clone()
            s_masked[reduced_r:] = 0.0
            low_rank_i = s_masked.unsqueeze(1) * v  # (min_dim, n)
        else:
            _, s, v = torch.linalg.svd(vector, full_matrices=True)
            # s: (min(m,n),), v: (m, m) if m<=n else (n, n)
            min_dim = min(m, n)
            reduced_r = max(1, s.shape[0] // T)

            s_mask = torch.zeros_like(s)
            s_mask[:reduced_r] = 1.0
            s_masked = s * s_mask

            v_mask = torch.zeros_like(v)
            v_mask[:reduced_r, :] = 1.0
            v_masked = v * v_mask

            S_mat = torch.zeros(m, n, device=device)
            S_mat[:min_dim, :min_dim] = torch.diag(s_masked[:min_dim])
            low_rank_i = S_mat @ v_masked  # (m, n)

        # --- compact SVD of de-meaned vector → task-specific reference ---
        u2, s2, v2 = torch.linalg.svd(vector - average_vector, full_matrices=False)
        u2 = u2[:, :reduced_r]
        s2 = s2[:reduced_r]
        v2 = v2[:reduced_r, :]

        low_rank_list.append(low_rank_i)
        taskvector_list.append(u2 @ torch.diag(s2) @ v2 + average_vector)

    low_rank   = torch.stack(low_rank_list)    # (T, ?, n)
    taskvector = torch.stack(taskvector_list)  # (T, m, n)

    merging = torch.nn.Parameter(vectors.sum(dim=0).clone())
    opt = torch.optim.Adam([merging], lr=1e-5)
    norms = vectors.reshape(T, -1).norm(p=2, dim=-1).square()  # (T,)

    for step in range(iter_num):
        diff = merging.unsqueeze(0) - taskvector               # (T, m, n)
        ip   = torch.matmul(diff, low_rank.transpose(-2, -1))  # (T, m, ?)
        loss = (ip.square() / norms[:, None, None]).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            logger.debug(f"  [{label}] step {step:3d}  loss={loss.item():.4e}")

    return merging.detach().to(original_dtype)


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def _load_flat(params_path: str | pathlib.Path) -> dict[str, np.ndarray]:
    """Load checkpoint → flat dict keyed by '/'-joined path."""
    import flax.traverse_util as traverse_util
    from openpi.models.model import restore_params

    path = pathlib.Path(params_path).resolve()
    logger.info(f"  Loading: {path}")
    params = restore_params(path, restore_type=np.ndarray)
    flat = traverse_util.flatten_dict(params)
    return {"/".join(k): v for k, v in flat.items()}


def _save(flat_params: dict[str, np.ndarray], output_dir: str | pathlib.Path) -> None:
    """Save flat param dict as Orbax PyTree checkpoint."""
    import flax.traverse_util as traverse_util
    import orbax.checkpoint as ocp

    output = pathlib.Path(output_dir).resolve()
    if output.exists():
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    tuple_flat = {tuple(k.split("/")): v for k, v in flat_params.items()}
    nested = traverse_util.unflatten_dict(tuple_flat)

    ckptr = ocp.PyTreeCheckpointer()
    ckptr.save(output, {"params": nested})
    logger.info(f"  Saved: {output}")


# ---------------------------------------------------------------------------
# Main merge logic
# ---------------------------------------------------------------------------

def run_merge(
    base_path: str,
    ft_paths: list[str],
    output_path: str,
    scope: str = "expert1_only",
    iter_num: int = 300,
    scaling: float = 1.0,
    device: str = "cuda",
) -> None:
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA unavailable, falling back to CPU")
        device = "cpu"

    logger.info("Loading base checkpoint…")
    base = _load_flat(base_path)

    logger.info(f"Loading {len(ft_paths)} fine-tuned checkpoint(s)…")
    fts = [_load_flat(p) for p in ft_paths]

    eligible = [k for k in base if _is_attn_ffn(k) and _in_scope(k, scope)]
    logger.info(f"WUDI-eligible: {len(eligible)} / {len(base)} params  (scope={scope})")

    merged: dict[str, np.ndarray] = {}

    for key, base_val in base.items():
        if key not in eligible:
            merged[key] = base_val
            continue

        ptype = _param_type(key)
        if ptype is None:
            merged[key] = base_val
            continue

        # Compute task vectors: list of (L, ...) arrays
        tvecs = [ft[key].astype(np.float32) - base_val.astype(np.float32) for ft in fts]
        L = tvecs[0].shape[0]

        merged_layers = []
        for l in range(L):
            layer_tvecs = [tv[l] for tv in tvecs]  # list of (...) one per model

            # decompose each model's layer into 2D sub-blocks
            blocks_per_model = [decompose_layer(ptype, lt) for lt in layer_tvecs]
            num_blocks = len(blocks_per_model[0])

            merged_subs = []
            for b in range(num_blocks):
                block_stack = np.stack([blocks_per_model[m][b] for m in range(len(fts))], axis=0)
                label = f"{key}[L{l}][b{b}]"
                result = wudi_optimize(
                    label,
                    torch.from_numpy(block_stack),
                    iter_num=iter_num,
                    device=device,
                )
                merged_subs.append(result.numpy())

            merged_layers.append(compose_layer(ptype, merged_subs, layer_tvecs[0].shape))

        merged_tv = np.stack(merged_layers, axis=0)  # (L, ...)
        merged[key] = base_val + scaling * merged_tv
        logger.info(f"  merged {key}  {base_val.shape}")

    logger.info("Saving…")
    _save(merged, output_path)
    logger.info(f"Done. Output: {output_path}")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def run_test() -> None:
    import torch

    print("=== Round-trip test ===")
    cases = {
        "q":    (4, 8, 2),
        "kv":   (2, 1, 6, 2),
        "av":   (4, 2, 8),
        "gate": (2, 6, 12),
        "lin":  (12, 6),
    }
    for ptype, shape in cases.items():
        orig   = np.random.randn(*shape).astype(np.float32)
        blocks = decompose_layer(ptype, orig)
        assert all(b.ndim == 2 for b in blocks), f"{ptype}: not all 2D"
        recon  = compose_layer(ptype, blocks, shape)
        assert recon.shape == shape, f"{ptype}: shape mismatch {recon.shape} != {shape}"
        assert np.allclose(recon, orig, atol=1e-6), f"{ptype}: values differ"
        block_shapes = [b.shape for b in blocks]
        print(f"  {ptype:4s}  {shape} → {block_shapes} → {recon.shape}  ✓")

    print("\n=== Linearity test  decompose(ft - base) == decompose(ft) - decompose(base) ===")
    for ptype, shape in cases.items():
        base_ = np.random.randn(*shape).astype(np.float32)
        ft_   = np.random.randn(*shape).astype(np.float32)
        lhs   = decompose_layer(ptype, ft_ - base_)
        rhs   = [f - b for f, b in zip(decompose_layer(ptype, ft_), decompose_layer(ptype, base_))]
        for i, (l, r) in enumerate(zip(lhs, rhs)):
            assert np.allclose(l, r, atol=1e-5), f"{ptype}[{i}]: linearity failed"
        print(f"  {ptype:4s}  ✓")

    print("\n=== WUDI optimize smoke test (CPU, 2 models, 5 steps) ===")
    vectors = torch.randn(2, 8, 16)
    result  = wudi_optimize("smoke", vectors, iter_num=5, device="cpu")
    assert result.shape == (8, 16)
    assert torch.isfinite(result).all()
    print(f"  shape={tuple(result.shape)}  finite={torch.isfinite(result).all().item()}  ✓")

    print("\nAll tests passed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ap = argparse.ArgumentParser(description="WUDI merging for pi0.5")
    ap.add_argument("--test",    action="store_true", help="Run smoke test and exit")
    ap.add_argument("--base",    type=str)
    ap.add_argument("--ft",      type=str, nargs="+")
    ap.add_argument("--output",  type=str)
    ap.add_argument("--scope",   type=str, default="expert1_only",
                    choices=["expert1_only", "both_experts"])
    ap.add_argument("--iter",    type=int, default=300)
    ap.add_argument("--scaling", type=float, default=1.0)
    ap.add_argument("--device",  type=str, default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    if args.test:
        run_test()
        return

    if not all([args.base, args.ft, args.output]):
        ap.error("--base, --ft, and --output are required (or use --test)")

    run_merge(
        base_path=args.base,
        ft_paths=args.ft,
        output_path=args.output,
        scope=args.scope,
        iter_num=args.iter,
        scaling=args.scaling,
        device=args.device,
    )


if __name__ == "__main__":
    main()
