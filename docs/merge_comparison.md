# OpenPI 模型融合方法对比

## 一、项目概览

| 项目 | 路径 | 定位 |
|------|------|------|
| **wudi_merge** | `openpi/scripts/wudi_merge.py` | 数据无关，基于 task vector 的 WUDI 优化融合 |
| **model_arithmetic** | `kai0/model_arithmetic/` | 数据驱动，基于验证集 loss 的加权平均融合 |

---

## 二、核心设计哲学差异

```
wudi_merge:
  fine-tuned_1 ──┐
                 ├── 计算 task vector (ft - base) ──→ WUDI 优化 ──→ base + scaling * merged_tv
  fine-tuned_2 ──┘   （最小化 task vector 之间的干扰）

model_arithmetic:
  fine-tuned_1 ──┐
                 ├── 直接对完整权重做加权平均 ──→ Σ w_i * params_i
  fine-tuned_2 ──┘   （权重 w_i 由验证集 loss 优化）
```

**根本区别**：
- `wudi_merge` 在 **task vector 空间**操作（delta = ft - base），解决多任务向量之间的干扰问题
- `model_arithmetic` 在 **权重空间**操作（直接平均原始参数），解决"哪个 checkpoint 贡献更多"的问题

---

## 三、融合方法详细对比

### 3.1 融合公式

**wudi_merge**：
```
# 对 attn/FFN 参数（WUDI 优化）：
task_vector_i = ft_i - base
merged_tv     = WUDI_optimize([tv_1, tv_2, ...])  # Adam 最小化投影干扰
merged        = scaling2 × base + scaling × merged_tv

# 对 norm 参数（简单平均 task vector）：
avg_tv = mean([ft_i - base for i in models])
merged = scaling2 × base + scaling × avg_tv

# 对冻结参数（vision/embedder/action proj）：
merged = base  （不变）
```

**model_arithmetic**：
```
# 对所有参数统一处理（加权平均原始权重）：
weights = optimize(validation_loss)   # 多种方式求 w_i
merged  = Σ w_i × params_i           # 直接在权重空间加权
```

### 3.2 参数分类处理

**wudi_merge** 对不同参数使用不同策略：

| 参数类型 | 具体参数 | 处理方式 |
|---------|---------|---------|
| attn 投影 | q/kv/attn_vec einsum | **WUDI 优化**（per-layer 2D SVD）|
| FFN | gating_einsum, linear | **WUDI 优化** |
| Norm | pre_attention_norm, pre_ffw_norm, final_norm | 简单平均 task vector |
| 冻结 | embedder, action_in/out_proj, time_mlp | **保持 base 不变** |
| Vision | PaliGemma/img/ | 保持 base 不变（llm_only 时）|

**model_arithmetic** 对所有参数一视同仁：

| 参数类型 | 处理方式 |
|---------|---------|
| 所有参数（包括 vision、norm、attn、FFN）| 统一加权平均 |

### 3.3 WUDI 核心算法（wudi_merge 独有）

```python
# 对每一层的每个 2D 子块：
for layer l in range(18):
    # 1. 分解：将参数拆成 2D 子块（q_einsum: N*D×H, kv: D×K*H, ...）
    blocks = decompose_layer(ptype, layer_tvec)

    # 2. WUDI 优化（最小化跨任务干扰）：
    #    - 计算平均 task vector
    #    - SVD 提取每个任务的低秩主方向
    #    - de-centering：对 (tv_i - avg) 做 compact SVD
    #    - Adam 优化 merging vector，最小化其在低秩子空间上的投影
    merged_block = wudi_optimize(blocks)

    # 3. 重组
    merged_layer = compose_layer(ptype, merged_block)
```

---

## 四、权重确定方式对比

**wudi_merge**（数据无关）：
- `--scaling`：task vector 的缩放系数，手动指定（无需验证集）
- `--scaling2`：base 权重系数，默认 1.0
- 所有 checkpoint 的 task vector 在 WUDI 框架内自动优化融合权重

**model_arithmetic**（数据驱动，5种方法）：

| 方法 | 原理 | 是否需要数据 |
|------|------|------------|
| `average` | 等权平均 1/N | ❌ |
| `inverse_loss` | w_i ∝ 1/loss_i² | ✅ |
| `gradient_descent` | Adam 在 simplex 上优化混合 loss | ✅ |
| `adaptive_gradient_descent` | 同上，loss 越大步长越大 | ✅ |
| `greedy` | 贪心前向选择最优 checkpoint 子集 | ✅ |

gradient_descent 实现原理（对比 JAX vs PyTorch 版）：
```
# 优化变量: log_weights（softmax 后满足 simplex 约束）
# 每步：
weights = softmax(log_weights)
mixed_params = Σ w_i × params_i
loss = model.forward(mixed_params, validation_batch)
∂loss/∂w_k = Σ_p (∂loss/∂p · p_k)   # 链式法则投影
∂loss/∂log_w_k = w_k × (g_k - Σ w_j g_j)  # softmax Jacobian
Adam update(log_weights)
```

---

## 五、工程实现对比

### 5.1 Checkpoint 格式支持

| | wudi_merge | model_arithmetic |
|--|-----------|-----------------|
| **JAX** (Orbax/OCDBT) | ✅ | ✅ (`arithmetic.py`) |
| **PyTorch** (safetensors) | ❌ | ✅ (`arithmetic_torch.py`) |

### 5.2 输入输出

| | wudi_merge | model_arithmetic |
|--|-----------|-----------------|
| 输入 | base checkpoint + N 个 fine-tuned | N 个 fine-tuned checkpoint |
| 需要 base | ✅ 必须 | ❌ 不需要（直接平均 ft） |
| 需要验证数据 | ❌ | ✅（除 average 方法外）|
| 输出格式 | Orbax PyTreeCheckpointer | JAX: CheckpointManager / PyTorch: safetensors |

### 5.3 Scope 控制

**wudi_merge** 有三种 scope：
```
expert1_only  → 只融合 action expert（_1 后缀）
llm_only      → 融合两个 expert，跳过 vision（默认）
both_experts  → 全部参数（含 vision）
```

**model_arithmetic** 无 scope 概念，对所有参数统一处理。

---

## 六、适用场景总结

| 场景 | 推荐方法 |
|------|---------|
| 无验证数据，多任务微调模型融合 | **wudi_merge**（WUDI 最小化干扰）|
| 有验证数据，希望最优化融合 loss | **model_arithmetic**（gradient_descent）|
| 快速合并，不需精细调权 | **model_arithmetic**（average 或 inverse_loss）|
| 只想融合 action expert 权重 | **wudi_merge**（`--scope expert1_only`）|
| PyTorch checkpoint | **model_arithmetic**（arithmetic_torch.py）|
| JAX checkpoint + 干扰最小化 | **wudi_merge** |

---

## 七、两者的本质区别一句话总结

- **wudi_merge**：*"如何让多个任务的 delta 融合时互不干扰"*——在 task vector 空间用 SVD+Adam 优化
- **model_arithmetic**：*"各个 checkpoint 应该贡献多少权重"*——在权重空间用验证集 loss 优化
