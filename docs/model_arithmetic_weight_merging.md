# 模型权重融合方案（Model Arithmetic）

## 概述

本方案移植自 `kai0/model_arithmetic/`，实现对多个独立训练的 checkpoint 进行权重融合，
生成一个在多个任务上均有能力的混合模型，无需重新训练。

---

## 核心原理

### 1. 参数混合

对 N 个 checkpoint 的参数按权重加权平均：

```
θ_mixed = Σᵢ wᵢ · θᵢ,   Σwᵢ = 1, wᵢ ≥ 0
```

前提：所有 checkpoint 来自同一个 base model（处于同一 loss basin），
否则加权平均会落在 loss landscape 的鞍点，效果变差。

实现：`scripts/model_arithmetic_common.py` → `mix_params()`

```python
def mix_params(params_list, weights):
    weights /= weights.sum()
    for key in params_list[0].keys():
        stacked = np.stack([p[key] for p in params_list], axis=0)
        mixed[key] = np.average(stacked, axis=0, weights=weights).astype(np.float32)
```

### 2. norm_stats 混合

每个 checkpoint 有自己的归一化统计量（mean/std），融合时需要同步混合：

```
norm_mixed = Σᵢ wᵢ · norm_statsᵢ
```

使用与参数相同的权重，保证模型对输入的归一化预期一致。

实现：`mix_norm_stats()` —— 对 JSON 中 mean/std/quantile 数组分别做加权平均。

norm_stats 保存位置：
- 原始 checkpoint：`{ckpt_dir}/assets/physical-intelligence/libero/norm_stats.json`
- 融合输出：`{output_dir}/norm_stats.json`（同时复制到 `{output_dir}/0/assets/physical-intelligence/libero/`，
  供 `serve_policy.py` 自动加载）

### 3. 权重参数化（Log-space Softmax）

为保证权重始终满足概率单纯形约束（wᵢ ≥ 0，Σwᵢ = 1），使用 log-space 参数化：

```
wᵢ = softmax(uᵢ) = exp(uᵢ) / Σⱼ exp(uⱼ)
```

优化变量为无约束的 `u`，通过 softmax 映射到权重空间。

### 4. 梯度投影

梯度下降时，对 loss 关于混合参数的梯度，通过链式法则投影回权重空间：

```
gₖ = <∂L/∂θ_mixed, θₖ>        # 每个 checkpoint 参数与 loss 梯度的内积
∂L/∂uₖ = wₖ · (gₖ - Σwᵢgᵢ)   # softmax Jacobian
```

---

## 权重优化方法

共 5 种策略，通过 `--optimize_method` 指定：

| 方法 | 描述 | 适用场景 |
|------|------|---------|
| `average` | 等权重 `wᵢ = 1/N` | 快速验证，baseline |
| `inverse_loss` | 按各 checkpoint val loss 的平方反比 `wᵢ ∝ 1/lossᵢ²` | 无梯度，快速 |
| `gradient_descent` | Adam + cosine LR + 梯度投影，50 iter | 推荐，效果最好 |
| `adaptive_gradient_descent` | 在 GD 基础上加 loss 自适应缩放 | 实验性 |
| `greedy` | 逐步贪心选择最优 checkpoint 组合 | N 较小时 |

---

## 工作流程

### Step 1：Dump 验证数据

```bash
sbatch dump_libero_val_balanced.sbatch
```

关键：**必须从各 suite 分别 dump，而非混合 config**，否则数据分布偏斜导致 GD 权重偏向某个 suite。

详见：`docs/libero_val_balanced_dump.md`

输出：`pkl/libero_val_balanced.pkl`（52 batch，每 suite 各 13 batch = 25%）

### Step 2：权重融合

```bash
sbatch merge_4task_gd.sbatch        # gradient_descent 方法
# 或
sbatch merge_4task_average.sbatch   # average 方法
```

核心调用：
```bash
uv run scripts/arithmetic.py \
    --config pi05_libero \
    --data-path pkl/libero_val_balanced.pkl \
    --checkpoints \
        checkpoints/pi05_libero_spatial/my_experiment/29999 \
        checkpoints/pi05_libero_object/my_experiment/29999 \
        checkpoints/pi05_libero_goal/my_experiment/29999 \
        checkpoints/pi05_libero_10/my_experiment/29999 \
    --optimize_method gradient_descent \
    --num_iterations 50 \
    --learning_rate 0.05 \
    --output checkpoints/pi05_libero_4task_merge_gd_balanced \
    --gpu_ids 0,1
```

### Step 3：评测

```bash
sbatch eval_4task_merge_gd_balanced_all_suites.sbatch
```

---

## 关键文件

| 文件 | 说明 |
|------|------|
| `scripts/arithmetic.py` | JAX checkpoint 融合主脚本 |
| `scripts/arithmetic_torch.py` | PyTorch safetensors 融合（相同逻辑） |
| `scripts/model_arithmetic_common.py` | 共享工具：`mix_params`, `mix_norm_stats`, `compute_optimal_weights` |
| `scripts/dump_data.py` | 从 config 导出验证数据为 pkl |
| `scripts/inspect_val_distribution.py` | 查看 pkl 的 suite 分布（基于 episodes.jsonl 精确分类）|
| `dump_libero_val_balanced.sbatch` | 均衡 dump 4 个 suite 的 val 数据 |
| `merge_4task_gd.sbatch` | 4 checkpoint GD 融合 |
| `merge_4task_average.sbatch` | 4 checkpoint 平均融合 |
| `eval_4task_merge_avg_all_suites.sbatch` | 平均融合评测（4 suite） |
| `eval_4task_merge_gd_balanced_all_suites.sbatch` | GD 均衡融合评测（4 suite） |
| `pkl/libero_val_balanced.pkl` | 均衡验证数据（52 batch，各 suite 25%）|
| `pkl/libero_val_balanced_meta.json` | batch 级 suite 标签（供 inspect 用）|

---

## 注意事项

### norm_stats 路径
`serve_policy.py` 在 `{checkpoint_dir}/assets/{asset_id}/norm_stats.json` 查找 norm_stats。
融合脚本输出的 norm_stats 在 `{output}/norm_stats.json`（顶层），评测 sbatch 中需要手动复制：

```bash
mkdir -p ${CKPT_DIR}/assets/physical-intelligence/libero
cp ${OUTPUT}/norm_stats.json ${CKPT_DIR}/assets/physical-intelligence/libero/norm_stats.json
```

### GPU 分配（评测）
MuJoCo EGL 与 JAX 在同一 GPU 上运行会导致 core dump（见 `docs/libero_eval_core_dump_incident_2026-03-30.md`）：

```bash
CUDA_VISIBLE_DEVICES=1  serve_policy.py   # GPU1：策略推理
CUDA_VISIBLE_DEVICES=0  examples/libero/main.py  # GPU0：仿真渲染
MUJOCO_EGL_DEVICE_ID=0
```

### 内存需求
GD 融合需同时持有 4 个模型参数 + 梯度，实测内存需求 ~155 GB，sbatch 需申请 `--mem=256G`。

### wandb 离线日志
计算节点无网络，融合脚本支持 wandb offline：
```bash
# 融合完成后在登录节点同步
source proxy_on.sh
.venv/bin/wandb sync wandb_offline/wandb/offline-run-*
```

---

## 实验结果

### 融合的 4 个 checkpoint

| Checkpoint | 任务 suite | 训练步数 |
|------------|-----------|---------|
| `pi05_libero_spatial/my_experiment/29999` | libero_spatial | 29999 |
| `pi05_libero_object/my_experiment/29999` | libero_object | 29999 |
| `pi05_libero_goal/my_experiment/29999` | libero_goal | 29999 |
| `pi05_libero_10/my_experiment/29999` | libero_10 | 29999 |

### 方法对比

#### Average（等权重）

Val data: `libero_val.pkl`（有偏，libero_10 占 36.7%）

最终权重：`[0.25, 0.25, 0.25, 0.25]`

| Suite | 成功率 |
|-------|-------|
| libero_spatial | 41.0% |
| libero_goal | 10.5% |
| libero_object | 10.5% |
| libero_10 | 1.5% |
| **Overall** | **15.9%** |

#### Gradient Descent（有偏 val data）

Val data: `libero_val.pkl`（libero_10 占 36.7%）

最终权重：`[spatial=0.100, object=0.111, goal=0.231, libero_10=0.558]`

权重严重偏向 libero_10，原因是 val data 分布不均导致 optimizer 偏向 libero_10。

#### Gradient Descent（均衡 val data）

Val data: `libero_val_balanced.pkl`（各 suite 各 25%）

最终权重：`[spatial=0.187, goal=0.253, object=0.261, libero_10=0.299]`（更均衡）

Best val loss: 0.061254

评测结果（进行中）。
