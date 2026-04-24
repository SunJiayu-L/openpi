# TIES Merging for pi0.5 — 设计与使用文档

脚本：`scripts/ties_merge.py`
作者：jiayusun
日期：2026-04-18

---

## 1. 目的

把若干个在不同 LIBERO 任务上微调过的 pi0.5 checkpoint 融合成一个单一 checkpoint，
在不重新训练的前提下尽可能保留各自任务能力，用于后续多任务评测。

本脚本实现 **TIES Merging**（Yadav et al., *Resolving Interference When Merging
Models*, NeurIPS 2023），作为 `wudi_merge.py` 的对照 baseline。

---

## 2. 代码由来

### 2.1 算法参考

从 `MLLMerging/InternVL/internvl_chat/model_merging.py` 的 `ties_merging` 函数
（line 163–276）移植核心算法。原实现针对 PyTorch `nn.Module`，用 `state_dict()` /
`named_parameters()` 访问权重。

### 2.2 基础脚手架

Checkpoint I/O、scope 过滤、冻结前缀、norm/bias 屏蔽、wandb 日志、CLI 骨架
全部复用自本仓库已有的 `scripts/wudi_merge.py`（已在 `pi05_libero_4task_from_wudi_*`
等多个 checkpoint 上验证过）。**具体做法是先 `cp wudi_merge.py ties_merge.py`，
再替换核心算法部分**，以保证：
- JAX checkpoint 读写路径一致
- scope 选项（`expert1_only` / `both_experts` / `llm_only` / `lang_and_vision`）
  与 WUDI 版本口径完全相同，便于公平对比
- 冻结前缀 / norm-bias 过滤规则与 WUDI 版本对齐

### 2.3 从 torch 到 JAX 的桥接

原 MLLMerging 实现依赖 PyTorch `nn.Module`；而 pi0.5 是 JAX Flax + Orbax
checkpoint。处理方式：
- 用 `openpi.models.model.restore_params` 把 Orbax PyTree 读成 `{str: np.ndarray}`
- 计算 task vector 在 numpy 里完成
- `torch.from_numpy(...)` 仅用于调用 `kthvalue` / `sign` 等数值算子
  （torch 在这里相当于"带 GPU 加速的 numpy"，不涉及 autograd / nn.Module）
- 结果 `.cpu().numpy()` 转回，再用 `orbax.checkpoint.PyTreeCheckpointer` 写回

### 2.4 和 MergeVLA 的区别

`MergeVLA/model_merging/mergy.py` 走的是完全 PyTorch 的路线
（prismatic + HuggingFace + `fusion_bench.method.TiesMergingAlgorithm`，
操作 `state_dict()`）。不适用于 openpi 的 JAX checkpoint，所以本脚本没有
复用它的代码，只参考了"TIES 要 exclude 掉 norm/bias/embedding"这一设计经验。

---

## 3. 算法思路

TIES 的核心观察：多模型融合时，**不同任务在同一参数位置的更新方向常相互抵消**，
直接平均会让 task vector 幅值减弱到丢失任务能力。TIES 分三步解决：

### Step 1: Trim（修剪）

按任务分别处理，把该任务 task vector 中 `|value|` 最小的 `mask_rate` 比例元素置零，
只保留幅值较大的元素。直觉：小幅值更新大多是噪声或任务无关漂移，丢弃后不影响能力。

实现：`_mask_smallest_magnitude`（ties_merge.py:145-162）
```
kth_vals = flat.abs().kthvalue(num_mask, dim=1)     # 每任务阈值
keep     = flat.abs() >= kth_vals                   # 保留 top (1-rate)
return flat * keep
```

### Step 2: Elect（选签）

对每个参数位置，跨任务把所有值相加，取符号作为"共识方向"；
若和为 0，则用整体多数符号兜底。

实现：`_elect_signs`（ties_merge.py:165-172）
```
signs    = sign(flat.sum(dim=0))                    # 列和的符号
majority = sign(signs.sum())                        # 整体多数
signs    = where(signs == 0, majority, signs)       # 0 位兜底
```

### Step 3: Disjoint Merge（分组平均）

每个参数位置只平均"符号与选签一致"的任务值，忽略方向冲突的任务。

实现：`_disjoint_merge`（ties_merge.py:175-182）
```
keep  = (sign(flat) == signs)                       # 广播匹配
return (flat * keep).sum(0) / keep.sum(0).clamp(1)
```

### 与 WUDI 的对比

|  | WUDI | TIES |
|---|---|---|
| 解决冲突方式 | Adam 优化合并向量，最小化与各任务低秩子空间的投影能量 | 基于统计的三步骤（trim + sign vote + disjoint avg） |
| 是否需要梯度 | 是（每个 2D 子块 300 步） | 否 |
| 是否按层分解 | 是（per-layer 2D 子块） | 否（全局 flatten） |
| 速度 | 慢（数小时） | 快（几分钟） |
| 显存 | 大（SVD + Adam） | 小（只需装下 flat 矩阵） |

---

## 4. 参数识别规则

### 4.1 参与融合（scope 内 & 非 norm/bias）

以 `llm_only`（默认）为例：
- `PaliGemma/llm/.../attn/q_einsum/w`、`kv_einsum/w`、`attn_vec_einsum/w`
- `PaliGemma/llm/.../mlp/gating_einsum`、`linear`
- 及其 action expert 版本（带 `_1` 后缀）

### 4.2 保持 base 值（不融合）

| 类别 | 匹配 | 原因 |
|---|---|---|
| 冻结前缀 | `PaliGemma/llm/embedder/`、`action_in_proj/`、`action_out_proj/`、`time_mlp_in/`、`time_mlp_out/` | 输入/输出接口需严格对齐 base，融合会破坏 IO |
| norm 层 | `*_norm*`、`LayerNorm`、`RMSNorm` 等 | 数值敏感，且微调通常只做轻微偏移，融合收益低风险高 |
| bias | 所有 `bias` 参数 | 与 MLLMerging 默认排除规则对齐 |
| scope 外 | 如 `PaliGemma/img/*`（在 `llm_only` 下） | 视觉编码器通常不做任务特化，base 即可 |

### 4.3 scope 选项对比

| scope | 融合 | 不融合 |
|---|---|---|
| `expert1_only` | 仅 action expert | 视觉 + language expert |
| `both_experts` | 视觉 + 两 expert | — |
| `llm_only`（默认） | 两 expert | 视觉 |
| `lang_and_vision` | 视觉 + language expert | action expert |

---

## 5. 最终参数合成公式

```
merged_param = scaling2 * base_param + scaling * ties(task_vectors)
```

其中 `task_vector_i = ft_i - base`，ties 三步在全部 eligible 参数拼成的
单一 flat 向量上执行。

---

## 6. How to Use

### 6.1 冒烟测试

```bash
cd /storage/yukaichengLab/lishiwen/jiayusun/openpi
python scripts/ties_merge.py --test
```

验证 TIES 三步在随机张量上的正确性（约 1 秒完成）。

### 6.2 正式融合（示例：合并 4 个 LIBERO 任务）

```bash
python scripts/ties_merge.py \
    --base   checkpoints/pi05_libero_RETRAIN_base/params \
    --ft     checkpoints/pi05_libero_spatial/params \
             checkpoints/pi05_libero_object/params \
             checkpoints/pi05_libero_goal/params \
             checkpoints/pi05_libero_10/params \
    --output checkpoints/merged_ties_4task \
    --scope  llm_only \
    --mask-rate 0.8 \
    --scaling 1.0 \
    --device cuda
```

输出目录结构：
```
checkpoints/merged_ties_4task/
└── params/
    ├── _METADATA
    └── ...  (Orbax 分片)
```

可直接被 `scripts/serve_policy.py` 加载评测。

### 6.3 超参速查

| 参数 | 默认 | 说明 |
|---|---|---|
| `--mask-rate` | 0.8 | Trim 比例；越大越激进。设 0 则跳过 trim |
| `--scaling` | 1.0 | task vector 缩放 α |
| `--scaling2` | 1.0 | base 缩放；一般保持 1.0 |
| `--scope` | `llm_only` | 融合范围 |
| `--device` | `cuda` | cuda 不可用会自动回退 cpu |

### 6.4 调参建议

- **首次尝试**：`--mask-rate 0.8 --scaling 1.0`（论文默认）
- **性能下降明显**：减小 `--scaling` 到 0.3~0.5
- **任务冲突严重**（如动作方向相反）：增大 `--mask-rate` 到 0.9
- **任务近似同分布**：可减小 `--mask-rate` 到 0.5

### 6.5 运行耗时参考

- 4 个 pi0.5 checkpoint 合并（scope=`llm_only`，GPU）：约 1–2 分钟
- 对比 WUDI 同配置：约 2–4 小时

---

## 7. 代码走查要点（for review）

请 reviewer 重点关注：

1. **scope / 冻结过滤**（ties_merge.py:64-120、246-254）：是否与 `wudi_merge.py`
   一致。`_is_expert1`、`_FROZEN_PREFIXES`、`_NORM_BIAS_RE` 直接 copy 自 WUDI 版，
   未做修改。

2. **TIES 三步正确性**（ties_merge.py:145-192）：核对是否与 MLLMerging 原版
   `ties_merging` 内联函数（`mask_smallest_magnitude_param_values` /
   `get_param_signs` / `disjoint_merge`）行为一致。
   冒烟测试 `run_test` 覆盖了每一步的单元检查。

3. **JAX 桥接**（ties_merge.py:210-231、260-274）：`_load_flat` / `_save`
   逻辑与 WUDI 版本完全相同；torch 仅作数值库使用，没有 `nn.Module` 依赖。

4. **公式对齐**（ties_merge.py:296-299）：`scaling2 * base + scaling * tv`
   与 WUDI 版保持一致，便于横向对比。

5. **确定性**：eligible keys 用 `sorted()` 保证 flatten / unflatten 顺序
   可复现（ties_merge.py:256）。

### 已知限制

- Trim 阈值是**跨所有 eligible 参数全局**计算的（因为先 flatten 再 kthvalue）。
  这和 MLLMerging 原版行为一致，但与另一种"per-layer trim"变种不同。
  如需 per-layer，可在 eligible_keys 循环里分别调用 `ties_core`。
- 所有 task vector 需一次性装入显存：4 个 pi0.5 约 `4 × 1.6B × 4B = 26 GB`，
  A100 80G 够用；显存不足时改 `--device cpu`。

---

## 8. 相关文件

- `scripts/ties_merge.py` — 本脚本
- `scripts/wudi_merge.py` — WUDI 版本，共享 scope / I/O 逻辑
- `MLLMerging/InternVL/internvl_chat/model_merging.py` — 算法参考（PyTorch）
- `MergeVLA/model_merging/mergy.py` — 另一套 PyTorch VLA 融合方案（不共享代码）
- `docs/wudi_merge_summary.md` — WUDI 实验总结
