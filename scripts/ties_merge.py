#!/usr/bin/env python3
"""TIES merging for pi0.5.

从 MLLMerging/InternVL/internvl_chat/model_merging.py :: ties_merging 移植而来。

算法（Yadav et al., NeurIPS 2023）：
  1. Trim（修剪）  ：按任务分别处理，把 |value| 最小的 mask_rate 比例置零，
                    仅保留幅值较大的元素。
  2. Elect（选签） ：把各任务相同位置的值相加，取符号作为"共同意志"；
                    若和为 0，则用整体多数符号兜底。
  3. Merge（合并） ：每个参数位置只平均"符号与选签一致"的任务值
                    （disjoint merge，不一致的任务被忽略）。

scope / 冻结参数 的处理与 wudi_merge.py 保持一致：
  - 冻结前缀（embedder、action_in/out、time_mlp）  -> 直接使用 base 值
  - norm / bias 参数                                -> 直接使用 base 值
  - 不在 scope 内的参数（如 vision model）          -> 直接使用 base 值
  - scope 内其余参数                                -> 参与 TIES 融合

与 WUDI 不同：TIES 不做按层 2D 分解，而是把所有可融合参数拼成一个大向量一次处理。

Usage:
    # 冒烟测试（CPU，随机张量）：
    python scripts/ties_merge.py --test

    # 执行融合（仅 LLM）：
    python scripts/ties_merge.py \\
        --base   /path/to/pi05_base/params \\
        --ft     /path/to/ft1/params /path/to/ft2/params \\
        --output checkpoints/merged/ties_llm \\
        --scope  llm_only \\
        --mask-rate 0.8 \\
        --scaling 1.0 \\
        --device cuda
"""
# 使用 __future__ 的 annotations，允许类型注解中引用尚未导入/定义的名字。
from __future__ import annotations

# 标准库：命令行解析、日志、路径、正则、文件拷贝、sys 路径操作。
import argparse
import logging
import pathlib
import re
import shutil
import sys

# numpy 用于 checkpoint 读写与 task vector 的 CPU 侧构造。
import numpy as np

# wandb 可选：可用则初始化，不可用则跳过（不影响融合正确性）。
try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# 把仓库根目录（openpi/）加进 sys.path，确保能导入 src/openpi/* 模块。
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

# 模块级 logger，后续日志都通过它输出。
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 参数识别（与 wudi_merge.py 完全一致）
# ---------------------------------------------------------------------------

def _is_expert1(key: str) -> bool:
    """判断参数是否属于 action expert（expert index >= 1）。"""
    # 只在 PaliGemma/llm/ 子树下判断 expert，避免视觉分支里的 LayerNorm_1 被误判。
    if not key.startswith("PaliGemma/llm/"):
        return False
    # 按 '/' 切段，对每一段做名字匹配。
    segs = key.split("/")
    # 允许出现 expert 下标的模块名白名单：attn_N / mlp_N / *_norm_N / einsum_N / linear_N。
    expert_named = re.compile(
        r"^(attn|mlp|pre_attention_norm|pre_ffw_norm|final_norm|"
        r"q_einsum|kv_einsum|attn_vec_einsum|gating_einsum|linear)_(\d+)$"
    )
    # 任一段带有 _N 且 N>=1，就判为 action expert。
    for seg in segs:
        m = expert_named.fullmatch(seg)
        if m and int(m.group(2)) >= 1:
            return True
    return False


def _is_vision(key: str) -> bool:
    # 视觉编码器的参数都放在 PaliGemma/img/ 下。
    return key.startswith("PaliGemma/img/")


# 始终保持 base 值不变的"冻结前缀"：词表 embedding、动作输入/输出投影、时步 MLP。
# 这些模块不参与任何融合，无论 scope 怎么设置。
_FROZEN_PREFIXES = (
    "PaliGemma/llm/embedder/",
    "action_in_proj/",
    "action_out_proj/",
    "time_mlp_in/",
    "time_mlp_out/",
)


def _is_frozen(key: str) -> bool:
    # 只要命中任意一个冻结前缀，就是冻结参数。
    return any(key.startswith(p) for p in _FROZEN_PREFIXES)


# 匹配所有 norm 层参数与 bias 参数的正则。
# 与 MLLMerging 的 exclude_param_names_regex 对齐：norm 和 bias 不参与融合。
_NORM_BIAS_RE = re.compile(
    r"(^|/)norm(/|$)|norm\d*(/|$)"
    r"|(LayerNorm|RMSNorm|encoder_norm|pre_attention_norm|pre_ffw_norm"
    r"|final_norm|attention_norm|ffn_norm)"
    r"|(^|/)bias$"
)


def _is_norm_or_bias(key: str) -> bool:
    # 命中正则 => 该参数是 norm 或 bias => 保持 base 值。
    return bool(_NORM_BIAS_RE.search(key))


def _in_scope(key: str, scope: str) -> bool:
    # 冻结参数永远不在 scope 内。
    if _is_frozen(key):
        return False
    # expert1_only: 只融合 action expert 的参数。
    if scope == "expert1_only":
        return _is_expert1(key)
    # both_experts: 两个 expert + 视觉分支，全部融合。
    if scope == "both_experts":
        return True
    # llm_only: 融合两个 expert，但不动视觉分支。
    if scope == "llm_only":
        return not _is_vision(key)
    # lang_and_vision: 融合 language expert(expert0) + 视觉，不动 action expert。
    if scope == "lang_and_vision":
        return not _is_expert1(key)
    # 不认识的 scope 直接报错。
    raise ValueError(f"Unknown scope: {scope!r}")


# ---------------------------------------------------------------------------
# TIES 核心算法（移植自 MLLMerging 的 ties_merging）
# ---------------------------------------------------------------------------

def _mask_smallest_magnitude(flat: "torch.Tensor", mask_rate: float) -> "torch.Tensor":
    """第 1 步 Trim：按任务（行）把 |value| 最小的 mask_rate 比例置零。

    flat: (T, N) 的 float 张量，T 为任务数，N 为参数总数；返回形状不变。
    """
    # 这里 import 只是为了类型注解提示；函数体内并不实际使用 torch 名字。
    import torch  # noqa: F401
    # mask_rate 为 0 直接返回原张量，避免无谓计算。
    if mask_rate <= 0.0:
        return flat
    # 要置零的元素个数（对每个任务行都一样）。
    num_mask = int(flat.shape[1] * mask_rate)
    # 保护：比例太小导致 0 个，直接原样返回。
    if num_mask <= 0:
        return flat
    # kthvalue 取出每行"第 num_mask 小"的 |value|，得到阈值。
    kth_vals, _ = flat.abs().kthvalue(k=num_mask, dim=1, keepdim=True)
    # 大于等于阈值的位置保留，其它置零。
    keep = flat.abs() >= kth_vals
    return flat * keep


def _elect_signs(flat: "torch.Tensor") -> "torch.Tensor":
    """第 2 步 Elect：跨任务汇总符号；若某位置为 0，则用整体多数符号。"""
    import torch
    # 对同一参数位置把所有任务的值相加，再取符号 => 该位置的"共识方向"。
    signs = torch.sign(flat.sum(dim=0))
    # 计算整体多数符号（正多还是负多）。
    majority = torch.sign(signs.sum())
    # 0 符号用多数符号兜底，避免 disjoint_merge 丢整列。
    signs = torch.where(signs == 0, majority, signs)
    return signs


def _disjoint_merge(flat: "torch.Tensor", signs: "torch.Tensor") -> "torch.Tensor":
    """第 3 步 Merge：每列只平均"符号与选签一致"的任务值。"""
    # 构造 bool 掩码：签为正 & 任务值为正；或 签为负 & 任务值为负。
    keep = ((signs.unsqueeze(0) > 0) & (flat > 0)) | ((signs.unsqueeze(0) < 0) & (flat < 0))
    # 被保留的值，其余位置乘 False 置 0。
    kept = flat * keep
    # 每列被保留的任务个数；clamp 防止除 0。
    denom = keep.sum(dim=0).clamp(min=1).float()
    # 求和再除以保留个数 = "按符号一致任务"求平均。
    return kept.sum(dim=0) / denom


def ties_core(task_matrix: "torch.Tensor", mask_rate: float) -> "torch.Tensor":
    """把 TIES 三步组合起来：输入 (T, N)，输出 (N,)。"""
    import torch
    # 整个流程不需要梯度，关掉自动求导以省内存。
    with torch.no_grad():
        # float() 保证 kthvalue/sign 在 float32 下数值稳定。
        masked = _mask_smallest_magnitude(task_matrix.float(), mask_rate)
        signs = _elect_signs(masked)
        merged = _disjoint_merge(masked, signs)
    return merged


# ---------------------------------------------------------------------------
# Checkpoint 读写（与 wudi_merge.py 相同）
# ---------------------------------------------------------------------------

def _load_flat(params_path: str | pathlib.Path) -> dict[str, np.ndarray]:
    # 延迟导入：避免仅做静态扫描时强依赖 flax / openpi。
    import flax.traverse_util as traverse_util
    from openpi.models.model import restore_params

    # 规范化路径；若给的是上层目录就自动拼一个 params/。
    path = pathlib.Path(params_path).resolve()
    params_dir = path / "params" if (path / "params" / "_METADATA").exists() else path
    logger.info(f"  Loading: {params_dir}")
    # 用 openpi 提供的 restore_params，返回嵌套 dict，叶子为 np.ndarray。
    params = restore_params(params_dir, restore_type=np.ndarray)
    # 将嵌套结构压平：key 是 tuple。
    flat = traverse_util.flatten_dict(params)
    # tuple -> 以 '/' 连接的字符串，便于打印/匹配。
    return {"/".join(k): v for k, v in flat.items()}


def _save(flat_params: dict[str, np.ndarray], output_dir: str | pathlib.Path) -> None:
    # 延迟导入：orbax 只在保存时需要。
    import flax.traverse_util as traverse_util
    import orbax.checkpoint as ocp

    # serve_policy 等下游会自动拼 /params，因此我们显式保存到 {output_dir}/params/。
    output = pathlib.Path(output_dir).resolve() / "params"
    if output.exists():
        # 覆盖同名目录，避免旧文件残留。
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # 字符串 key 还原回 tuple key。
    tuple_flat = {tuple(k.split("/")): v for k, v in flat_params.items()}
    # 再 unflatten 成嵌套树，符合 orbax 期望的形式。
    nested = traverse_util.unflatten_dict(tuple_flat)

    # PyTreeCheckpointer：用 {"params": ...} 包一层以兼容训练/推理管线。
    ckptr = ocp.PyTreeCheckpointer()
    ckptr.save(output, {"params": nested})
    logger.info(f"  Saved: {output}")


# ---------------------------------------------------------------------------
# 融合主流程
# ---------------------------------------------------------------------------

def run_merge(
    base_path: str,
    ft_paths: list[str],
    output_path: str,
    scope: str = "llm_only",
    mask_rate: float = 0.8,
    scaling: float = 1.0,
    scaling2: float = 1.0,
    device: str = "cuda",
) -> None:
    """最终公式：merged = scaling2 * base + scaling * ties(task_vectors)。"""
    # 只在真正需要时导入 torch。
    import torch

    # 若用户指定 cuda 但机器无 GPU，优雅回退到 CPU。
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA unavailable, falling back to CPU")
        device = "cpu"

    # wandb run 名取输出目录名；若 wandb 不可用则只记日志。
    run_name = pathlib.Path(output_path).name
    if _WANDB_AVAILABLE:
        _wandb.init(
            project="ties-merge-pi05",
            name=run_name,
            config=dict(
                scope=scope, mask_rate=mask_rate,
                scaling=scaling, scaling2=scaling2,
                device=device, output=output_path,
            ),
            reinit=True,
        )
        logger.info("wandb initialized.")
    else:
        logger.info("wandb not available, skipping.")

    # 载入 base checkpoint，得到 {key: np.ndarray}。
    logger.info("Loading base checkpoint...")
    base = _load_flat(base_path)

    # 载入所有微调 checkpoint。
    logger.info(f"Loading {len(ft_paths)} fine-tuned checkpoint(s)...")
    fts = [_load_flat(p) for p in ft_paths]

    # 挑选可参与 TIES 融合的 key：
    #   · 在 scope 内
    #   · 不是 norm/bias
    #   · 所有微调 ckpt 都含该 key（否则 delta 没法算）
    eligible_keys = [
        k for k in base
        if _in_scope(k, scope)
        and not _is_norm_or_bias(k)
        and all(k in ft for ft in fts)
    ]
    # 所有 norm/bias 参数的集合：这些参数最终保持 base 值。
    frozen_norm_bias = {k for k in base if _is_norm_or_bias(k)}
    logger.info(f"TIES-eligible: {len(eligible_keys)} / {len(base)}  (scope={scope})")
    logger.info(f"Norm/bias frozen at base: {len(frozen_norm_bias)} params")

    # 排序以保证 flatten / unflatten 具有确定性顺序，可复现。
    eligible_keys_sorted = sorted(eligible_keys)
    # 每个 key 的原始形状（用于还原）。
    shapes = {k: base[k].shape for k in eligible_keys_sorted}
    # 每个 key 的元素个数（用于 offset 计算）。
    sizes = {k: int(np.prod(shapes[k])) for k in eligible_keys_sorted}
    # 所有可融合参数的总数。
    total = sum(sizes.values())
    logger.info(f"Total eligible params: {total:,}")

    # 在 CPU 上先构造出 (T, N) 的 task-vector 矩阵，然后一次性搬到 device。
    per_task_rows: list[np.ndarray] = []
    for ft in fts:
        # 对每个 key：task vector = ft - base（用 float32 做差，避免精度损失）。
        chunks = [
            (ft[k].astype(np.float32) - base[k].astype(np.float32)).reshape(-1)
            for k in eligible_keys_sorted
        ]
        # 把该任务的所有 key 拼成一个长向量。
        per_task_rows.append(np.concatenate(chunks, axis=0))
    # 再在任务维上堆叠，得到 (T, N)。
    flat_np = np.stack(per_task_rows, axis=0)
    logger.info(f"Flat task-vector matrix: {flat_np.shape}  dtype={flat_np.dtype}")

    # numpy -> torch 并搬到目标 device。
    flat = torch.from_numpy(flat_np).to(device)
    # 执行 TIES 三步，结果再搬回 numpy 便于 scatter。
    merged_flat = ties_core(flat, mask_rate=mask_rate).cpu().numpy()

    # 记录一些整体统计到 wandb（L2 便于对比不同 mask_rate 的幅度）。
    if _WANDB_AVAILABLE and _wandb.run is not None:
        _wandb.log({
            "ties/num_params": total,
            "ties/num_eligible_keys": len(eligible_keys_sorted),
            "ties/mask_rate": mask_rate,
            "ties/merged_l2": float(np.linalg.norm(merged_flat)),
        })

    # 把合并后的扁平向量按 size 切回每个 key 的原始形状。
    merged_tvs: dict[str, np.ndarray] = {}
    offset = 0
    for k in eligible_keys_sorted:
        n = sizes[k]
        merged_tvs[k] = merged_flat[offset : offset + n].reshape(shapes[k])
        offset += n
    # 确认每个字节都被用掉，没有遗留。
    assert offset == total

    # 构造最终参数字典：
    #   · frozen / norm / bias / 非 eligible -> 直接用 base
    #   · eligible -> scaling2 * base + scaling * merged_tv
    merged: dict[str, np.ndarray] = {}
    for key, base_val in base.items():
        if key in frozen_norm_bias or key not in merged_tvs:
            merged[key] = base_val
            continue
        # 在 float32 空间相加，再回退到原 dtype（通常是 bf16/fp16）。
        combined = scaling2 * base_val.astype(np.float32) + scaling * merged_tvs[key]
        merged[key] = combined.astype(base_val.dtype)

    # 持久化。
    logger.info("Saving...")
    _save(merged, output_path)
    logger.info(f"Done. Output: {output_path}")

    # 关闭 wandb run。
    if _WANDB_AVAILABLE and _wandb.run is not None:
        _wandb.finish()


# ---------------------------------------------------------------------------
# 冒烟测试
# ---------------------------------------------------------------------------

def run_test() -> None:
    # 测试用到 torch。
    import torch

    # 用例 1：核心端到端是否能跑通，结果有限且非 NaN。
    print("=== TIES core smoke test ===")
    torch.manual_seed(0)
    T, N = 3, 1000
    tv = torch.randn(T, N)
    merged = ties_core(tv, mask_rate=0.8)
    assert merged.shape == (N,)
    assert torch.isfinite(merged).all()
    print(f"  shape={tuple(merged.shape)}  l2={merged.norm().item():.3f}  ok")

    # 用例 2：mask_rate=0 不应改变输入。
    print("\n=== mask_rate=0 preserves input ===")
    tv2 = torch.randn(2, 50)
    masked = _mask_smallest_magnitude(tv2, 0.0)
    assert torch.equal(masked, tv2)
    print("  ok")

    # 用例 3：符号选举——前两行正号多数，第三行虽负但投票被压倒。
    print("\n=== sign election ===")
    mat = torch.tensor([[1.0, -1.0], [2.0, -2.0], [-0.5, 0.1]])
    signs = _elect_signs(mat)
    # 第 0 列和 = 1+2-0.5 > 0 => +1； 第 1 列和 = -1-2+0.1 < 0 => -1。
    assert signs[0].item() == 1.0
    assert signs[1].item() == -1.0
    print("  ok")

    # 用例 4：disjoint_merge 只平均符号一致的任务。
    print("\n=== disjoint merge ===")
    mat = torch.tensor([[1.0, 2.0], [3.0, -4.0]])
    signs = torch.tensor([1.0, 1.0])
    out = _disjoint_merge(mat, signs)
    # 第 0 列两行都为正 => mean(1,3)=2；第 1 列只有第一行为正 => 2。
    assert abs(out[0].item() - 2.0) < 1e-6
    assert abs(out[1].item() - 2.0) < 1e-6
    print("  ok")

    print("\nAll tests passed.")


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main() -> None:
    # 统一的日志格式。
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # 参数定义。
    ap = argparse.ArgumentParser(description="TIES merging for pi0.5")
    ap.add_argument("--test",    action="store_true", help="Run smoke test and exit")
    ap.add_argument("--base",    type=str)
    ap.add_argument("--ft",      type=str, nargs="+")
    ap.add_argument("--output",  type=str)
    ap.add_argument("--scope",   type=str, default="llm_only",
                    choices=["expert1_only", "both_experts", "llm_only", "lang_and_vision"])
    # mask_rate: 每任务要置零的 |value| 最小元素占比；TIES 原论文默认 0.8。
    ap.add_argument("--mask-rate", type=float, default=0.8,
                    help="fraction of smallest-magnitude entries to zero per task (default 0.8)")
    # scaling: 合并后 task vector 的缩放系数。
    ap.add_argument("--scaling",  type=float, default=1.0,
                    help="scaling coefficient on the merged task vector")
    # scaling2: base_val 的缩放系数；一般保持 1.0。
    ap.add_argument("--scaling2", type=float, default=1.0,
                    help="base_val coefficient; merged = scaling2*base + scaling*merged_tv")
    ap.add_argument("--device",   type=str, default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    # --test 优先，仅跑冒烟测试后退出。
    if args.test:
        run_test()
        return

    # 非测试模式下 base/ft/output 必填。
    if not all([args.base, args.ft, args.output]):
        ap.error("--base, --ft, and --output are required (or use --test)")

    # 执行融合主流程。
    run_merge(
        base_path=args.base,
        ft_paths=args.ft,
        output_path=args.output,
        scope=args.scope,
        mask_rate=args.mask_rate,
        scaling=args.scaling,
        scaling2=args.scaling2,
        device=args.device,
    )


if __name__ == "__main__":
    # 脚本入口。
    main()
