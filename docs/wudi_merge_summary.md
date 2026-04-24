# WUDI 权重融合实现说明

**分支**：`optmerge`
**关键 commit**：`61f4ae5`（2026-03-23）

---

## 一、新建文件

| 文件 | 行数 | 说明 |
|---|---|---|
| `scripts/wudi_merge.py` | 469 行 | 主融合脚本，含全部算法逻辑 |
| `wudi_merge.sbatch` | 62 行 | Slurm GPU 提交脚本 |

其余已有文件**均未修改**（仅新增，无改动）。

---

## 二、与上一版本的核心区别

上一版本（已通过 `git reset --hard 1bb61e7` 撤销）将 18 层打包折叠进矩阵，并使用了自研的 SVD 压缩路径。本版本按以下原则重写：

| 维度 | 旧版本 | 本版本 |
|---|---|---|
| L 维度处理 | 折叠进矩阵（L×N×H 等） | **逐层循环**，每层独立 WUDI |
| 算法 | 自研 compact SVD 路径 | **移植 `wudi_merging2`**（SVD de-centering） |
| 非 attn/FFN 参数 | 简单平均 | **保持 base 值，完全不融合** |

---

## 三、采用的融合权重

### 3.1 融合哪些参数（allowlist）

只处理 Gemma Transformer 的 **Attention + FFN** 共 5 类参数，其余一律保持 base：

| 参数名模式 | 形状（单层，去 L） | 2D 化方式 | 子块数 |
|---|---|---|---|
| `q_einsum(_N)?/w` | `(N, D, H)` | `reshape(N×D, H)` | 1 |
| `kv_einsum(_N)?/w` | `(2, K, D, H)` | split K/V → 各 `reshape(D, K×H)` | 2 |
| `attn_vec_einsum(_N)?/w` | `(N, H, D)` | `reshape(N×H, D)` | 1 |
| `gating_einsum(_N)?` | `(2, D, Hff)` | split gate/value → 各 `(D, Hff)` | 2 |
| `linear(_N)?` | `(Hff, D)` | 直接使用（已 2D）| 1 |

checkpoint 实际存储有 leading **L=18**（来自 `nn.scan`），脚本在 axis=0 上循环后再做上述拆分。

### 3.2 Expert 范围控制（`--scope`）

| 模式 | 处理对象 | 典型参数路径 |
|---|---|---|
| `expert1_only`（默认） | 仅 action expert | `…/q_einsum_1/w`, `…/mlp_1/gating_einsum` |
| `both_experts` | action expert + PaliGemma LLM | 上述 + `…/q_einsum/w`, `…/mlp/gating_einsum` |

### 3.3 不融合的参数（保持 base）

以下参数**不参与任何融合**，直接从 base checkpoint 复制：

- 视觉编码器（SigLIP）：所有 `img_*` 路径
- 词嵌入：`embedder/input_embedding`
- LayerNorm / RMSNorm：所有 `*norm*`, `*scale*`
- 时间调制：`time_mlp*`
- 输入输出投影：`action_in_proj`, `action_out_proj`, `state_proj`
- LoRA 参数：`*lora*`
- expert1_only 模式下的 expert0（PaliGemma）attn/FFN

---

## 四、融合算法（`wudi_merging2` 移植）

对每个 2D 子块（task vector stacked 为 `(T, m, n)`）执行：

```
average = mean(task_vectors, dim=0)          # 共享均值基线

for each task i:
    # 原始向量 SVD → 低秩干扰基
    _, s, v = svd(task_vectors[i])
    reduced_r = max(1, rank // T)
    low_rank[i] = diag(s[:r]) @ v[:r, :]    # (r, n)

    # 去均值向量 compact SVD → 任务特有参考
    u2, s2, v2 = svd(task_vectors[i] - average, full=False)
    taskvec[i] = u2[:,:r] @ diag(s2[:r]) @ v2[:r] + average

# Adam 优化：最小化融合向量在干扰子空间的投影
merging = sum(task_vectors)                  # 初始化
for step in range(iter_num):
    diff = merging - taskvec                 # (T, m, n)
    ip   = diff @ low_rank.T                 # (T, m, r)
    loss = sum(ip² / ||task_vectors||²)
    Adam step on merging

merged_param = base + scaling × merging
```

**内存保护**：当子块元素数 > 800 万时自动切换 compact SVD（`full_matrices=False`），避免 PaliGemma 大矩阵 OOM。

---

## 五、使用方式

### 直接运行

```bash
uv run python scripts/wudi_merge.py \
    --base    /path/to/pi05_base/params \
    --ft      /path/to/ft1/params /path/to/ft2/params \
    --output  checkpoints/merged/wudi_e1_libero_goal \
    --scope   expert1_only \
    --iter    300 \
    --scaling 1.0 \
    --device  cuda
```

### Slurm 提交

```bash
# expert1_only（64GB 足够）
sbatch wudi_merge.sbatch expert1_only \
    checkpoints/pi05_libero/my_experiment/29999/params \
    checkpoints/pi05_libero_goal/my_experiment/29999/params \
    wudi_e1_libero_goal

# both_experts（建议改 --mem=128G，PaliGemma 矩阵更大）
sbatch wudi_merge.sbatch both_experts \
    checkpoints/pi05_libero/my_experiment/29999/params \
    checkpoints/pi05_libero_goal/my_experiment/29999/params \
    wudi_e01_libero_goal
```

### Smoke test

```bash
uv run python scripts/wudi_merge.py --test
# 验证：round-trip、线性性、WUDI optimizer 输出有限
```

---

## 六、当前实验状态

| Job | 实验 | Scope | 状态 |
|---|---|---|---|
| 32109 | Exp A | `expert1_only` | 运行中 |

完成后可用 `scripts/analyze_merge.py` 验证：action_expert 权重有变化，llm_backbone / vision_encoder 应为 0（expert1_only 模式）。
