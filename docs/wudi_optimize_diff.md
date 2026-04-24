# `wudi_optimize` vs `get_redundant_task_vector`：代码差异分析

本文对比 `scripts/wudi_merge.py` 中的 `wudi_optimize` 与 MLLMerging 的 `wudi_merging2` 中 `get_redundant_task_vector` 的实现差异。

---

## 整体结构对比

两者实现相同的核心算法：
1. 计算任务向量平均值
2. 对每个任务向量做 SVD，构造低秩干扰基（`low_rank`）和去中心参考（`taskvector`）
3. Adam 优化融合向量，最小化其在 `low_rank` 子空间的投影

---

## 差异点逐一说明

### 差异 1：原始向量 SVD 的 `full_matrices` 参数

**MLLMerging（始终 full SVD）：**
```python
u, s, v = torch.linalg.svd(vector, full_matrices=True)
# v shape: (n, n) — 完整方阵
```

**我们的代码（自动选择）：**
```python
use_compact = (m * n > 8_000_000)   # 元素数超过 800 万走 compact

if use_compact:
    _, s, v = torch.linalg.svd(vector, full_matrices=False)
    # v shape: (min(m,n), n) — 紧凑形式
else:
    _, s, v = torch.linalg.svd(vector, full_matrices=True)
    # v shape: (n, n) — 同原版
```

**原因**：pi0.5 的 PaliGemma `gating_einsum` 每层子块为 `(2048, 16384)`，`full_matrices=True` 时 V 矩阵为 `(16384, 16384)` ≈ 1GB，直接 OOM。compact SVD 将 V 压缩至 `(2048, 16384)`，内存降低 8 倍。

---

### 差异 2：`low_rank` 的构造方式与形状

**MLLMerging：**
```python
# 始终构造 (m, n) 形状的 low_rank
S_matrix = torch.zeros(m, n)
S_matrix[:min_dim, :min_dim] = torch.diag_embed(s)   # s 已被掩码
low_rank_i = S_matrix @ v                             # (m, n) @ (n, n) → (m, n)
```

**我们的代码（full SVD 路径，等价于原版）：**
```python
S_mat = torch.zeros(m, n)
S_mat[:min_dim, :min_dim] = torch.diag(s_masked[:min_dim])
low_rank_i = S_mat @ v_masked                         # (m, n) @ (n, n) → (m, n)
```

**我们的代码（compact SVD 路径，形状不同）：**
```python
low_rank_i = s_masked.unsqueeze(1) * v    # (min_dim, n)，不再是 (m, n)
```

| | MLLMerging | 我们（full） | 我们（compact） |
|---|---|---|---|
| `low_rank_i` 形状 | `(m, n)` | `(m, n)` | `(min_dim, n)` |
| `inner_product` 形状 | `(T, m, m)` | `(T, m, m)` | `(T, m, min_dim)` |

compact 路径的 inner product 维度缩小，loss 约束空间从 `m×m` 降为 `m×min_dim`，节省显存同时保留最重要的方向约束。

---

### 差异 3：`reduced_r` 的零值保护

**MLLMerging：**
```python
reduced_index_s = int(s.shape[0] / vectors.shape[0])
# 当 min(m,n) < T 时结果为 0，后续截断会取空张量
```

**我们的代码：**
```python
reduced_r = max(1, min_dim // T)
# 最少保留 1 个奇异分量，避免空张量
```

对 pi0.5 的参数矩阵（通常 `min(m,n)` >> T=2）不影响结果，但防御性更强。

---

### 差异 4：`torch.diag_embed` vs `torch.diag`

**MLLMerging：**
```python
S_matrix[:min_dim, :min_dim] = torch.diag_embed(s)
# ...
taskvector_list.append(u2 @ torch.diag_embed(s2) @ v2 + average_vector)
```

**我们的代码：**
```python
S_mat[:min_dim, :min_dim] = torch.diag(s_masked[:min_dim])
# ...
taskvector_list.append(u2 @ torch.diag(s2) @ v2 + average_vector)
```

对一维输入，`torch.diag_embed(x)` 和 `torch.diag(x)` 结果完全相同，无功能差异。

---

### 差异 5：`v` 的处理方式

**MLLMerging（显式掩码后矩阵乘）：**
```python
v_mask = torch.zeros_like(v)
v_mask[:reduced_index_s, :] = 1
v = v * v_mask           # 零化 reduced_index_s 之后的行
low_rank = S_matrix @ v  # S 对角线也已被掩码，等效于只取前 r 个方向
```

**我们的代码（full 路径，等价）：**
```python
v_mask = torch.zeros_like(v)
v_mask[:reduced_r, :] = 1.0
v_masked = v * v_mask
low_rank_i = S_mat @ v_masked   # 同上
```

**我们的代码（compact 路径，直接利用截断 v）：**
```python
s_masked[reduced_r:] = 0.0
low_rank_i = s_masked.unsqueeze(1) * v   # v 本身就只有 min_dim 行
```

compact 路径不需要显式掩码，因为 compact SVD 的 V 本来就只有 `min_dim` 行，截断 `s` 后直接广播乘即可。

---

## 完整对照表

| 项目 | MLLMerging | 我们的代码 |
|---|---|---|
| 原始 SVD 模式 | 始终 `full_matrices=True` | 自动：`full`（小矩阵）/ `compact`（大矩阵）|
| `low_rank` 形状 | `(T, m, n)` | `(T, m, n)`（full）或 `(T, min_dim, n)`（compact）|
| inner product 形状 | `(T, m, m)` | `(T, m, m)`（full）或 `(T, m, min_dim)`（compact）|
| `reduced_r = 0` 时 | 不保护，取空张量 | `max(1, ...)` 保证至少 1 |
| `diag` 函数 | `torch.diag_embed` | `torch.diag`（功能等价）|
| 适用矩阵规模 | 小矩阵（ViT/RoBERTa 级别）| 小矩阵 + 大矩阵（LLM 级别）|

---

## 算法等价性

当 `use_compact=False`（矩阵元素数 ≤ 800 万）时，两者数学上完全等价：
- `reduced_r` 计算结果相同（`min(m,n) // T`）
- `low_rank`、`taskvector` 构造方式相同
- loss 函数形式相同

当 `use_compact=True` 时，`low_rank` 的行数从 `m` 缩减为 `min(m,n)`，相当于在更紧凑的子空间内施加干扰约束，是内存与精度的权衡。对 pi0.5 实际触发 compact 的参数：

| 参数 | 子块形状 | 元素数 | 触发 compact |
|---|---|---|---|
| action expert `gating_einsum` | `(1024, 4096)` | 4.2M | 否 |
| PaliGemma `gating_einsum` | `(2048, 16384)` | 33.6M | **是** |
| PaliGemma `linear` | `(16384, 2048)` | 33.6M | **是** |
| PaliGemma `q_einsum` | `(8192, 256)` | 2.1M | 否 |
