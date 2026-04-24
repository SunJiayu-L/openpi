#!/usr/bin/env python3
"""WUDI merging for pi0.5 with per-layer 2D decomposition and SVD de-centering.

Algorithm: get_redundant_task_vector from wudi_merging2 (SVD de-centering variant).
  For each 2D sub-block (per layer, L looped):
    1. Compute average task vector across models
    2. Per task: full SVD of original → masked low_rank basis
                 compact SVD of (vector - avg) → de-centered task reference
    3. Adam optimize merging vector to minimize projection onto low_rank subspace

Gemma attention and FFN parameters are WUDI-merged;
other in-scope parameters use simple-averaged task vectors (aligned with wudi_merging2);
out-of-scope parameters (e.g. vision model) keep base values unchanged.

Parameter 2D decomposition (per layer, L dimension looped over).
Each 2D block corresponds to exactly ONE semantic linear projection, matching
the split in examples/convert_jax_model_to_pytorch.py (7 projections per layer):

  q_einsum/w   (L,N,D,H)   → transpose(0,2,1).reshape(N*H,D)  → (N*H, D)  [1块 q_proj]
  kv_einsum/w  (L,2,K,D,H) → each: transpose(0,2,1).reshape(K*H,D) → (K*H, D) [2块 k/v_proj]
  attn_vec/w   (L,N,H,D)   → transpose(2,0,1).reshape(D,N*H)  → (D, N*H) [1块 o_proj]
  gating       (L,2,D,Hff) → each: [x].transpose()            → (Hff, D)  [2块 gate/up_proj]
  linear       (L,Hff,D)   → transpose()                      → (D, Hff)  [1块 down_proj]

Total: 7 independent WUDI calls per layer, each exactly matching one PyTorch nn.Linear.weight.

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
# 说明：
# 这个脚本对 pi0.5 的注意力/FFN参数做 WUDI 合并。
# 核心做法是：先把每层参数拆成 2D 子块，再对每个子块做去中心化 SVD + Adam 优化。

from __future__ import annotations

import argparse
import logging
import pathlib
import re
import shutil
import sys

import numpy as np

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# 将仓库根目录加入 sys.path，保证可导入 src/openpi 下的模块。
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

# 模块级 logger。
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
    # 路径按 '/' 分段，例如：xxx/q_einsum_1/w
    segs = key.split("/")
    # 最后一个段，常见为 'w' 或 'linear_1' 这类名字。
    last = segs[-1]
    # 倒数第二段，用于判断 xxx/w 的父节点类型。
    parent = segs[-2] if len(segs) >= 2 else ""
    # 如果当前参数是权重叶子 'w'，从父节点推断其类型。
    if last == "w":
        # q_einsum 但不是 kv_einsum（避免误判）。
        if re.search(r"q_einsum", parent) and not re.search(r"kv_einsum", parent):
            return "q"
        # key/value 投影。
        if re.search(r"kv_einsum", parent):
            return "kv"
        # attention 输出投影。
        if re.search(r"attn_vec_einsum", parent):
            return "av"
        # 其他 /w 参数不参与此脚本的 2D 规则。
        return None
    # gating_einsum 没有 /w 叶子，类型直接由最后段判断。
    if re.match(r"gating_einsum(_\d+)?$", last):
        return "gate"
    # FFN linear 同理。
    if re.match(r"linear(_\d+)?$", last):
        return "lin"
    return None


def _is_expert1(key: str) -> bool:
    """True if key belongs to action expert (expert index >= 1)."""
    # 只在 LLM 参数树内判断 expert，避免把视觉分支中的 LayerNorm_1 / Dense_1 误判为 expert1。
    if not key.startswith("PaliGemma/llm/"):
        return False

    segs = key.split("/")

    # 仅这些模块名后缀 _N 表示 expert index；N>=1 即 action expert。
    expert_named = re.compile(
        r"^(attn|mlp|pre_attention_norm|pre_ffw_norm|final_norm|"
        r"q_einsum|kv_einsum|attn_vec_einsum|gating_einsum|linear)_(\d+)$"
    )
    for seg in segs:
        m = expert_named.fullmatch(seg)
        if m and int(m.group(2)) >= 1:
            return True
    return False


def _is_attn_ffn(key: str) -> bool:
    # 用正则统一过滤目标参数：注意力 Q/KV/AV + FFN gating/linear。
    return bool(_ATTN_FFN_RE.match(key))


def _is_vision(key: str) -> bool:
    """True if key belongs to the vision encoder (SigLIP / PaliGemma img)."""
    return key.startswith("PaliGemma/img/")


# 始终保持 base 值不变的参数前缀（无论 scope 如何）。
# 包括：词表 embedding、动作 I/O 投影、时步 MLP。
_FROZEN_PREFIXES = (
    "PaliGemma/llm/embedder/",
    "action_in_proj/",
    "action_out_proj/",
    "time_mlp_in/",
    "time_mlp_out/",
)


def _is_frozen(key: str) -> bool:
    return any(key.startswith(p) for p in _FROZEN_PREFIXES)


def _in_scope(key: str, scope: str) -> bool:
    # 冻结参数不参与任何融合，始终保持 base。
    if _is_frozen(key):
        return False
    # scope=expert1_only：只合并 action expert 参数。
    if scope == "expert1_only":
        return _is_expert1(key)
    # scope=both_experts：两个 expert 都参与（包括 vision）。
    if scope == "both_experts":
        return True
    # scope=llm_only：两个 expert 都参与，但排除 vision model。
    if scope == "llm_only":
        return not _is_vision(key)
    # 其他 scope 为非法输入。
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
    # q_proj: (N,D,H) → transpose(0,2,1) → (N,H,D) → reshape(N*H,D)
    # 官方: llm_attention_q_einsum[i].transpose(0,2,1).reshape(N*H, hidden_size)
    if ptype == "q":
        N, D, H = layer.shape
        return [layer.transpose(0, 2, 1).reshape(N * H, D)]

    # k_proj + v_proj: layer[x]: (K,D,H) → transpose(0,2,1) → (K,H,D) → reshape(K*H,D)
    # 官方: kv_einsum[i,0,0].transpose() = (D,H).T = (H,D) = (K*H,D) for K=1
    if ptype == "kv":
        _, K, D, H = layer.shape
        k_block = layer[0].transpose(0, 2, 1).reshape(K * H, D)
        v_block = layer[1].transpose(0, 2, 1).reshape(K * H, D)
        return [k_block, v_block]

    # o_proj: (N,H,D) → transpose(2,0,1) → (D,N,H) → reshape(D,N*H)
    # 官方 action: reshape(N*H,D).transpose(1,0) = (D,N*H)
    # 官方 language: transpose(2,0,1).reshape(N*H,D) — 当D=N*H时等价，通用形式用此路
    if ptype == "av":
        N, H, D = layer.shape
        return [layer.transpose(2, 0, 1).reshape(D, N * H)]

    # gate_proj + up_proj: layer[x]: (D,Hff) → transpose() → (Hff,D)
    # 官方: gating_einsum[i,0].transpose() = (D,Hff).T = (Hff,D)
    if ptype == "gate":
        assert layer.shape[0] == 2
        return [layer[0].transpose(), layer[1].transpose()]

    # down_proj: (Hff,D) → transpose() → (D,Hff)
    # 官方: mlp_linear[i].transpose() = (Hff,D).T = (D,Hff)
    if ptype == "lin":
        assert layer.ndim == 2
        return [layer.transpose()]

    raise ValueError(f"Unknown ptype: {ptype!r}")


def compose_layer(ptype: str, sub_blocks: list[np.ndarray], original_shape: tuple) -> np.ndarray:
    """Reconstruct a single-layer tensor from merged 2D sub-blocks."""
    # q 逆变换：(N*H,D) → reshape(N,H,D) → transpose(0,2,1) → (N,D,H)
    if ptype == "q":
        N, D, H = original_shape
        return sub_blocks[0].reshape(N, H, D).transpose(0, 2, 1)

    # kv 逆变换：(K*H,D) → reshape(K,H,D) → transpose(0,2,1) → (K,D,H) → stack → (2,K,D,H)
    if ptype == "kv":
        _, K, D, H = original_shape
        k = sub_blocks[0].reshape(K, H, D).transpose(0, 2, 1)
        v = sub_blocks[1].reshape(K, H, D).transpose(0, 2, 1)
        return np.stack([k, v], axis=0)

    # av 逆变换：(D,N*H) → reshape(D,N,H) → transpose(1,2,0) → (N,H,D)
    if ptype == "av":
        N, H, D = original_shape
        return sub_blocks[0].reshape(D, N, H).transpose(1, 2, 0)

    # gate 逆变换：(Hff,D) → transpose() → (D,Hff) → stack → (2,D,Hff)
    if ptype == "gate":
        return np.stack([sub_blocks[0].transpose(), sub_blocks[1].transpose()], axis=0)

    # lin 逆变换：(D,Hff) → transpose() → (Hff,D)
    if ptype == "lin":
        return sub_blocks[0].transpose()

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
    log_every: int = 50,
    curve_every: int = 1,
    progress_cb=None,
) -> tuple["torch.Tensor", dict[str, float]]:
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
    # 延迟导入 torch，避免仅做静态分析时强依赖。
    import torch

    # 记录输入 dtype，优化时统一转 float32，最后再转回。
    original_dtype = vectors.dtype
    vectors = vectors.float().to(device)
    # T=任务数（微调模型数），m/n=2D 子块形状。
    T, m, n = vectors.shape
    # 大矩阵走紧凑 SVD，减少显存占用。
    use_compact = (m * n > _OOM_THRESHOLD)

    # 任务向量平均值，用于 de-centering。
    average_vector = vectors.mean(dim=0)  # (m, n)
    # 保存每个任务构建出的低秩参考。
    low_rank_list: list[torch.Tensor] = []
    # 保存每个任务构建出的"去中心化重建向量"。
    taskvector_list: list[torch.Tensor] = []

    for i in range(T):
        # 第 i 个任务向量。
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
            # 这里的 low_rank_i 是截断奇异值后形成的近似子空间表示。
            low_rank_i = s_masked.unsqueeze(1) * v  # (min_dim, n)
        else:
            #! 计算sv。
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
            # 恢复到 (m,n) 形状的低秩表示。
            low_rank_i = S_mat @ v_masked  # (m, n)

        # --- compact SVD of de-meaned vector → task-specific reference ---
        # 对 (vector - average) 做紧凑 SVD，得到去中心后的主方向。
        u2, s2, v2 = torch.linalg.svd(vector - average_vector, full_matrices=False)
        u2 = u2[:, :reduced_r]
        s2 = s2[:reduced_r]
        v2 = v2[:reduced_r, :]

        low_rank_list.append(low_rank_i)
        # 再加回平均向量，作为每个任务的对齐参考。
        taskvector_list.append(u2 @ torch.diag(s2) @ v2 + average_vector)

    # 堆叠后便于批量算损失。
    low_rank = torch.stack(low_rank_list)    # (T, ?, n)
    taskvector = torch.stack(taskvector_list)  # (T, m, n)

    # 把可学习合并向量初始化为任务向量和。
    merging = torch.nn.Parameter(vectors.mean(dim=0).clone())
    # Adam 优化器。
    opt = torch.optim.Adam([merging], lr=1e-5)
    # 每个任务向量范数平方，用于归一化损失。
    norms = vectors.reshape(T, -1).norm(p=2, dim=-1).square()  # (T,)

    first_loss: float | None = None
    best_loss = float("inf")
    best_step = 0
    final_loss = float("inf")

    for step in range(iter_num):
        # 当前合并向量与各任务参考的差。
        diff = merging.unsqueeze(0) - taskvector               # (T, m, n)
        # 投影到各任务低秩子空间，作为"干扰量"。
        ip = torch.matmul(diff, low_rank.transpose(-2, -1))  # (T, m, ?)
        # 最小化投影能量（归一化后求和）。
        loss = (ip.square() / norms[:, None, None]).sum()
        loss_item = float(loss.item())
        if first_loss is None:
            first_loss = loss_item
        if loss_item < best_loss:
            best_loss = loss_item
            best_step = step
        final_loss = loss_item

        opt.zero_grad()
        loss.backward()
        opt.step()
        lr = opt.param_groups[0]["lr"]
        if progress_cb is not None and (step % max(1, curve_every) == 0 or step == iter_num - 1):
            progress_cb(step, loss_item, lr)

        if step % max(1, log_every) == 0 or step == iter_num - 1:
            logger.info(f"  [{label}] step {step:3d}  loss={loss_item:.4e}  lr={lr:.2e}")
            if _WANDB_AVAILABLE and _wandb.run is not None:
                _wandb.log({
                    f"{label}/loss": loss.item(),
                    f"{label}/lr": lr,
                    f"{label}/step": step,
                })

    summary = {
        "first_loss": float(first_loss if first_loss is not None else final_loss),
        "best_loss": float(best_loss),
        "best_step": float(best_step),
        "final_loss": float(final_loss),
    }
    logger.info(
        f"  [{label}] summary first={summary['first_loss']:.4e} "
        f"best={summary['best_loss']:.4e}@{int(summary['best_step'])} "
        f"final={summary['final_loss']:.4e}"
    )
    if _WANDB_AVAILABLE and _wandb.run is not None:
        _wandb.log({
            f"{label}/first_loss": summary["first_loss"],
            f"{label}/best_loss": summary["best_loss"],
            f"{label}/best_step": summary["best_step"],
            f"{label}/final_loss": summary["final_loss"],
        })

    # 返回优化后的张量，并恢复原 dtype。
    return merging.detach().to(original_dtype), summary


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def _load_flat(params_path: str | pathlib.Path) -> dict[str, np.ndarray]:
    """Load checkpoint → flat dict keyed by '/'-joined path."""
    # 仅在需要时导入，减少启动依赖。
    import flax.traverse_util as traverse_util
    from openpi.models.model import restore_params

    # 解析 checkpoint 路径：若给的是 checkpoint root（含 _CHECKPOINT_METADATA），
    # 自动切换到 params/ 子目录（orbax 需要 _METADATA 在该目录下）。
    path = pathlib.Path(params_path).resolve()
    if (path / "params" / "_METADATA").exists() and not (path / "_METADATA").exists():
        path = path / "params"
    logger.info(f"  Loading: {path}")
    params = restore_params(path, restore_type=np.ndarray)
    # flatten_dict 后 key 是 tuple，把它拼成 '/' 路径字符串。
    flat = traverse_util.flatten_dict(params)
    return {"/".join(k): v for k, v in flat.items()}


def _save(flat_params: dict[str, np.ndarray], output_dir: str | pathlib.Path) -> None:
    """Save flat param dict as Orbax PyTree checkpoint."""
    # 仅在保存时导入。
    import flax.traverse_util as traverse_util
    import orbax.checkpoint as ocp

    # serve_policy.py 会自动拼接 /params，因此保存到 {output_dir}/params/ 子目录。
    output = pathlib.Path(output_dir).resolve() / "params"
    if output.exists():
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # 将 'a/b/c' 字符串 key 还原成 tuple key，再 unflatten 成嵌套树。
    tuple_flat = {tuple(k.split("/")): v for k, v in flat_params.items()}
    nested = traverse_util.unflatten_dict(tuple_flat)

    # 以 {"params": nested} 的格式保存，兼容训练/加载管线。
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
    scaling: float = 0.1,
    scaling2: float = 1.0,
    device: str = "cuda",
    curve_every: int = 1,
) -> None:
    # scaling2: base_val 的系数，默认 1.0（等同原公式 base + scaling * tv）
    # 公式：merged = scaling2 * base_val + scaling * merged_tv
    # 运行合并主流程所需的 torch。
    import torch

    # 若用户指定 cuda 但不可用，自动回退 cpu。
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA unavailable, falling back to CPU")
        device = "cpu"

    # 初始化 wandb（如果可用）
    run_name = pathlib.Path(output_path).name
    if _WANDB_AVAILABLE:
        _wandb.init(
            project="wudi-merge-pi05",
            name=run_name,
            config=dict(
                scope=scope, iter_num=iter_num,
                scaling=scaling, scaling2=scaling2,
                device=device, output=output_path,
            ),
            reinit=True,
        )
        logger.info("wandb initialized.")
    else:
        logger.info("wandb not available, skipping.")

    # 读取 base 参数。
    logger.info("Loading base checkpoint…")
    base = _load_flat(base_path)

    # 读取全部微调参数。
    logger.info(f"Loading {len(ft_paths)} fine-tuned checkpoint(s)…")
    fts = [_load_flat(p) for p in ft_paths]

    # 挑出可参与 WUDI 的目标参数（attn/FFN 且在 scope 内）。
    eligible = [k for k in base if _is_attn_ffn(k) and _in_scope(k, scope)]
    # 在 scope 内但非 attn/FFN 的参数，做简单平均（对齐 MLLMerging wudi_merging2）。
    in_scope_keys = set(k for k in base if _in_scope(k, scope))
    logger.info(f"WUDI-eligible: {len(eligible)} / {len(base)} params  (scope={scope})")
    logger.info(f"In-scope (avg for non-attn/FFN): {len(in_scope_keys)} params")

    # merged 将保存最终完整参数。
    merged: dict[str, np.ndarray] = {}

    all_first_losses: list[float] = []
    all_best_losses: list[float] = []
    all_final_losses: list[float] = []
    global_curve_step = 0
    block_index = 0

    for key, base_val in base.items():
        # 非 scope 参数（如 vision model）保持 base，不做任何合并。
        if key not in in_scope_keys:
            merged[key] = base_val
            continue

        # 在 scope 内但非 attn/FFN → 简单平均 task vector（对齐 wudi_merging2 第二阶段）。
        if key not in eligible:
            if all(key in ft for ft in fts):
                avg_tv = np.mean(
                    [ft[key].astype(np.float32) - base_val.astype(np.float32) for ft in fts],
                    axis=0,
                )
                merged[key] = scaling2 * base_val + scaling * avg_tv
                logger.info(f"  avg-tv {key}  {base_val.shape}")
            else:
                merged[key] = base_val
            continue

        # 推断参数类型，用于分解/重组。
        ptype = _param_type(key)
        if ptype is None:
            merged[key] = base_val
            continue

        # 每个微调模型相对 base 的差值作为任务向量。
        tvecs = [ft[key].astype(np.float32) - base_val.astype(np.float32) for ft in fts]
        # 按层处理，L 是第一维。
        L = tvecs[0].shape[0]

        # 收集每层合并后的参数。
        merged_layers = []
        for l in range(L):
            # 取出第 l 层在所有任务上的张量。
            layer_tvecs = [tv[l] for tv in tvecs]  # list of (...) one per model

            # 每个任务的该层都分解成若干 2D 子块。
            blocks_per_model = [decompose_layer(ptype, lt) for lt in layer_tvecs]
            num_blocks = len(blocks_per_model[0])

            # 存储该层各子块的 WUDI 结果。
            merged_subs = []
            for b in range(num_blocks):
                # 按任务维堆叠，形成 (T, m, n)。
                block_stack = np.stack([blocks_per_model[m][b] for m in range(len(fts))], axis=0)
                # 便于日志定位：参数名 + 层号 + 子块号。
                label = f"{key}[L{l}][b{b}]"
                # 对该子块执行 WUDI 优化。
                this_block = block_index

                def _curve_logger(local_step: int, loss_val: float, lr_val: float) -> None:
                    nonlocal global_curve_step
                    if _WANDB_AVAILABLE and _wandb.run is not None:
                        _wandb.log(
                            {
                                "loss_curve/global_loss": loss_val,
                                "loss_curve/lr": lr_val,
                                "loss_curve/local_step": local_step,
                                "loss_curve/block_index": this_block,
                            },
                            step=global_curve_step,
                        )
                    global_curve_step += 1

                result, loss_summary = wudi_optimize(
                    label,
                    torch.from_numpy(block_stack),
                    iter_num=iter_num,
                    device=device,
                    log_every=max(1, iter_num // 6),
                    curve_every=max(1, curve_every),
                    progress_cb=_curve_logger,
                )
                all_first_losses.append(loss_summary["first_loss"])
                all_best_losses.append(loss_summary["best_loss"])
                all_final_losses.append(loss_summary["final_loss"])
                block_index += 1
                # 转回 numpy 以便后续 compose。
                merged_subs.append(result.cpu().numpy())

            # 子块重组为该层完整形状。
            merged_layers.append(compose_layer(ptype, merged_subs, layer_tvecs[0].shape))

        # 所有层堆叠后得到该参数的合并任务向量。
        merged_tv = np.stack(merged_layers, axis=0)  # (L, ...)
        # 最终参数 = scaling2 * base + scaling * merged_task_vector。
        merged[key] = scaling2 * base_val + scaling * merged_tv
        logger.info(f"  merged {key}  {base_val.shape}")

    if all_final_losses:
        global_first = float(np.mean(all_first_losses))
        global_best = float(np.mean(all_best_losses))
        global_final = float(np.mean(all_final_losses))
        logger.info(
            "WUDI loss summary (mean over blocks): "
            f"first={global_first:.4e} best={global_best:.4e} final={global_final:.4e} "
            f"(blocks={len(all_final_losses)})"
        )
        if _WANDB_AVAILABLE and _wandb.run is not None:
            _wandb.log({
                "loss/first_mean": global_first,
                "loss/best_mean": global_best,
                "loss/final_mean": global_final,
                "loss/num_blocks": len(all_final_losses),
            })

    # 写回 checkpoint。
    logger.info("Saving…")
    _save(merged, output_path)
    logger.info(f"Done. Output: {output_path}")

    if _WANDB_AVAILABLE and _wandb.run is not None:
        _wandb.finish()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def run_test() -> None:
    # 冒烟测试依赖 torch。
    import torch

    # 1) 分解/重组往返一致性测试。
    print("=== Round-trip test ===")
    # 各参数类型的样例 shape。
    cases = {
        "q":    (4, 8, 2),
        "kv":   (2, 1, 6, 2),
        "av":   (4, 2, 8),
        "gate": (2, 6, 12),
        "lin":  (12, 6),
    }
    for ptype, shape in cases.items():
        # 随机原始矩阵。
        orig = np.random.randn(*shape).astype(np.float32)
        # 分解成子块。
        blocks = decompose_layer(ptype, orig)
        # 子块必须全是 2D。
        assert all(b.ndim == 2 for b in blocks), f"{ptype}: not all 2D"
        # 重组并检查形状与数值。
        recon = compose_layer(ptype, blocks, shape)
        assert recon.shape == shape, f"{ptype}: shape mismatch {recon.shape} != {shape}"
        assert np.allclose(recon, orig, atol=1e-6), f"{ptype}: values differ"
        block_shapes = [b.shape for b in blocks]
        print(f"  {ptype:4s}  {shape} → {block_shapes} → {recon.shape}  ✓")

    # 2) 线性性测试：decompose(ft-base) == decompose(ft)-decompose(base)
    print("\n=== Linearity test  decompose(ft - base) == decompose(ft) - decompose(base) ===")
    for ptype, shape in cases.items():
        base_ = np.random.randn(*shape).astype(np.float32)
        ft_ = np.random.randn(*shape).astype(np.float32)
        lhs = decompose_layer(ptype, ft_ - base_)
        rhs = [f - b for f, b in zip(decompose_layer(ptype, ft_), decompose_layer(ptype, base_))]
        for i, (l, r) in enumerate(zip(lhs, rhs)):
            assert np.allclose(l, r, atol=1e-5), f"{ptype}[{i}]: linearity failed"
        print(f"  {ptype:4s}  ✓")

    # 3) WUDI 优化器小规模运行测试。
    print("\n=== WUDI optimize smoke test (CPU, 2 models, 5 steps) ===")
    vectors = torch.randn(2, 8, 16)
    result, _ = wudi_optimize("smoke", vectors, iter_num=5, device="cpu")
    assert result.shape == (8, 16)
    assert torch.isfinite(result).all()
    print(f"  shape={tuple(result.shape)}  finite={torch.isfinite(result).all().item()}  ✓")

    print("\nAll tests passed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    # 初始化日志格式。
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # 定义命令行参数。
    ap = argparse.ArgumentParser(description="WUDI merging for pi0.5")
    ap.add_argument("--test",    action="store_true", help="Run smoke test and exit")
    ap.add_argument("--base",    type=str)
    ap.add_argument("--ft",      type=str, nargs="+")
    ap.add_argument("--output",  type=str)
    ap.add_argument("--scope",   type=str, default="llm_only",
                    choices=["expert1_only", "both_experts", "llm_only"])
    ap.add_argument("--iter",    type=int, default=300)
    ap.add_argument("--scaling",  type=float, default=1.0)
    ap.add_argument("--scaling2", type=float, default=1.0,
                    help="base_val 系数，默认 1.0。公式: scaling2*base + scaling*merged_tv")
    ap.add_argument("--curve-every", type=int, default=1,
                    help="log unified wandb loss curve every N optimization steps")
    ap.add_argument("--device",   type=str, default="cuda", choices=["cuda", "cpu"])
    # 解析参数。
    args = ap.parse_args()

    # 若指定 --test，只跑测试后退出。
    if args.test:
        run_test()
        return

    # 合并模式要求 base/ft/output 必填。
    if not all([args.base, args.ft, args.output]):
        ap.error("--base, --ft, and --output are required (or use --test)")

    # 执行主流程。
    run_merge(
        base_path=args.base,
        ft_paths=args.ft,
        output_path=args.output,
        scope=args.scope,
        iter_num=args.iter,
        scaling=args.scaling,
        scaling2=args.scaling2,
        device=args.device,
        curve_every=args.curve_every,
    )


if __name__ == "__main__":
    # CLI 入口。
    main()
