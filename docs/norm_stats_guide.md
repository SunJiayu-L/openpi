# Norm Stats（归一化统计）详解

---

## 1. 什么是 Norm Stats

Norm Stats 是从训练数据集中预先计算的**统计量**，用于在训练和推理时对输入/输出数据做归一化（normalization）。

对于 LIBERO 数据集，norm_stats 包含两个 key：**state**（机器人状态）和 **actions**（动作），每个 key 存储 4 个统计量：

| 统计量 | 含义 | 用途 |
|--------|------|------|
| `mean` | 均值 | z-score 归一化 |
| `std` | 标准差 | z-score 归一化 |
| `q01` | 第 1 百分位 | 分位数归一化 |
| `q99` | 第 99 百分位 | 分位数归一化 |

**示例**（pi05_libero 的 state，8 维 = 7 关节角度 + 1 gripper）：

```json
"state": {
    "mean": [-0.047, 0.034, 0.764, 2.972, -0.220, -0.126, 0.027, -0.027],
    "std":  [ 0.105, 0.152, 0.379, 0.344,  0.907,  0.325, 0.014,  0.014],
    "q01":  [-0.399, -0.269, 0.038, 1.508, -2.721, -1.081, 0.002, -0.040],
    "q99":  [ 0.135,  0.336, 1.270, 3.277,  2.405,  0.597, 0.040, -0.002]
}
```

---

## 2. 为什么需要归一化

### 2.1 数值尺度差异

机器人 state 和 actions 各维度的数值范围差异很大：

```
关节1: [-0.40, 0.14]     范围 ~0.5
关节4: [ 1.51, 3.28]     范围 ~1.8
gripper: [0.002, 0.040]  范围 ~0.04
```

如果不归一化，loss 会被大数值维度主导，小数值维度（如 gripper）的学习信号被淹没。

### 2.2 归一化公式

**z-score 归一化**（openpi 默认）：

```
归一化:   x_norm = (x - mean) / (std + 1e-6)
反归一化: x = x_norm * (std + 1e-6) + mean
```

**分位数归一化**（可选，`use_quantile_norm=True`）：

```
归一化:   x_norm = (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0    → 映射到 [-1, 1]
反归一化: x = (x_norm + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
```

---

## 3. Norm Stats 和什么有关

### 只和数据集有关，和模型权重无关

Norm Stats 是从**训练数据**中统计出来的，反映的是数据集中 state/actions 的分布。它：

- **和数据集有关**：不同数据集、不同 episode 子集，统计量不同
- **和任务有关**：不同 LIBERO suite 的任务空间不同，数据分布不同
- **和模型无关**：pi0 和 pi0.5 使用同一个数据集时，norm_stats 完全相同

**各 LIBERO suite 的 norm_stats 对比**（state mean 前 3 维）：

| Config | 数据 | state mean |
|--------|------|-----------|
| `pi05_libero`（全量） | 全部 1693 episodes | [-0.047, 0.034, 0.764] |
| `pi05_libero_goal` | eps 379-806 (428 eps) | [-0.099, 0.014, 1.070] |
| `pi05_libero_object` | eps 807-1260 (454 eps) | [-0.030, -0.008, 0.203] |
| `pi05_libero_spatial` | eps 1261-1692 (432 eps) | [-0.024, 0.107, 1.058] |

可以看到，不同 suite 的数据分布差异显著。全量统计是所有 suite 的综合。

---

## 4. 为什么每次训练前都要执行

### 4.1 训练流程

```
Step 1: compute_norm_stats  →  生成 norm_stats.json
Step 2: train.py            →  读取 norm_stats.json，训练时使用
```

训练时，数据经过 transforms pipeline：

```
原始数据 → [数据变换] → [Normalize(norm_stats)] → [模型变换] → 模型输入
模型输出 → [Unnormalize(norm_stats)] → [数据反变换] → 动作输出
```

### 4.2 什么时候需要重新计算

| 场景 | 是否需要 |
|------|---------|
| 第一次训练某个 config | **必须** — 还没有 norm_stats |
| 更换数据集或 episode 范围 | **必须** — 数据分布变了 |
| 换模型（pi0 → pi0.5），数据不变 | **不需要** — 和模型无关 |
| 断点续训（resume） | **不需要** — 数据没变 |
| 同一 config 重新训练 | **不需要** — 已有缓存 |

### 4.3 计算结果缓存

计算结果保存在 `assets/{config_name}/{repo_id}/norm_stats.json`，例如：

```
assets/pi05_libero/physical-intelligence/libero/norm_stats.json
assets/pi05_libero_goal/physical-intelligence/libero/norm_stats.json
```

如果文件已存在，重新运行 `compute_norm_stats.py` 会**覆盖**它。训练脚本 `train_suite.sh` 每次都会重新计算以确保一致性，但如果确认数据没变，可以跳过（如 `train_suite_resume.sh` 就跳过了）。

---

## 5. 执行逻辑详解

### 5.1 命令

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_libero
```

### 5.2 `--config-name` 对应什么

`--config-name` 对应 `src/openpi/training/config.py` 中注册的 `TrainConfig` 的 `name` 字段。

例如 `pi05_libero` 对应：

```python
# src/openpi/training/config.py 第 748-768 行
TrainConfig(
    name="pi05_libero",                    # ← --config-name 匹配这个
    model=pi0_config.Pi0Config(pi05=True, action_horizon=10, ...),
    data=LeRobotLiberoDataConfig(
        repo_id="physical-intelligence/libero",   # ← 数据集
        base_config=DataConfig(prompt_from_task=True),
        extra_delta_transform=False,
    ),
    batch_size=256,
    ...
)
```

所有可用的 config name 都在 `_CONFIGS` 列表中，通过 `get_config(name)` 查找。

相关的 config name：

| --config-name | 数据范围 | 定义位置 |
|---------------|---------|---------|
| `pi0_libero` | 全部 40 task | config.py |
| `pi05_libero` | 全部 40 task | config.py |
| `pi05_libero_10` | libero_10 (eps 0-378) | config.py |
| `pi05_libero_goal` | libero_goal (eps 379-806) | config.py |
| `pi05_libero_object` | libero_object (eps 807-1260) | config.py |
| `pi05_libero_spatial` | libero_spatial (eps 1261-1692) | config.py |

### 5.3 执行流程

```python
# scripts/compute_norm_stats.py

def main(config_name, max_frames=None):
    # 1. 加载 TrainConfig
    config = get_config(config_name)   # e.g. "pi05_libero"

    # 2. 创建 DataConfig（包含 episode 过滤、数据变换等）
    data_config = config.data.create(config.assets_dirs, config.model)

    # 3. 创建数据加载器（加载 LeRobot 数据集，应用 episode 过滤）
    data_loader = create_torch_dataloader(data_config, ...)

    # 4. 遍历所有数据，累积统计量
    stats = {"state": RunningStats(), "actions": RunningStats()}
    for batch in data_loader:
        stats["state"].update(batch["state"])      # 增量更新 mean/std/q01/q99
        stats["actions"].update(batch["actions"])

    # 5. 计算最终统计量
    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    # 6. 保存到 assets/{config_name}/{repo_id}/norm_stats.json
    output_path = config.assets_dirs / data_config.repo_id
    normalize.save(output_path, norm_stats)
```

`RunningStats` 使用在线算法（Welford），逐 batch 累积，不需要将全部数据加载到内存。

### 5.4 数据流

```
LeRobot 数据集 (Parquet)
    │
    ├─ episode 过滤（根据 config 中的 episodes 参数）
    │    e.g. pi05_libero_goal → eps 379-806
    │
    ├─ repack_transforms（数据格式转换）
    │    e.g. LiberoInputs: 提取 state、images、language_instruction
    │
    ├─ data_transforms（数据预处理）
    │    e.g. ResizeImages, 构造 action chunk
    │
    └─ RunningStats.update()
         对每个 batch 的 state [B, 8] 和 actions [B, H, 7] 计算：
         - 均值、方差（在线 Welford 算法）
         - 分位数 q01/q99（直方图近似法）
```

---

## 6. Norm Stats 在训练和推理中的使用

### 6.1 训练时

```
[原始 state/actions] → Normalize → [归一化后数据] → 模型 loss 计算
```

训练代码从 checkpoint 或 assets 目录加载 norm_stats，构建 transforms pipeline。模型看到的数据已经归一化为接近 N(0,1) 的分布。

### 6.2 推理时

```
[环境观测 state] → Normalize → 模型 → [归一化动作] → Unnormalize → [实际动作]
```

**推理必须使用与训练时完全相同的 norm_stats**，否则：
- 输入的归一化不匹配 → 模型收到的输入分布与训练时不同
- 输出的反归一化不匹配 → 动作值被映射到错误范围

### 6.3 norm_stats 保存位置

| 阶段 | 路径 | 说明 |
|------|------|------|
| 计算后 | `assets/{config_name}/{repo_id}/norm_stats.json` | 初始位置 |
| 训练 checkpoint 中 | `checkpoints/{name}/my_experiment/{step}/assets/{repo_id}/norm_stats.json` | 每个 checkpoint 随训练保存一份 |
| 融合 checkpoint 中 | `checkpoints/merged/{name}/assets/{repo_id}/norm_stats.json` | 融合时保存（使用全量 norm_stats） |

---

## 7. 常见问题

### Q: 不同模型（pi0 vs pi0.5）需要不同的 norm_stats 吗？

**不需要**，只要训练数据相同。`pi0_libero` 和 `pi05_libero` 使用同一个数据集和相同的 episode 范围，理论上 norm_stats 完全一致。但实际中因为数据变换（`extra_delta_transform`）不同，actions 的 norm_stats 会略有差异。

### Q: 融合模型应该用哪个 norm_stats？

使用**全量配置**的 norm_stats（如 `pi05_libero`），而非任何单个 suite 的。详见 [model_merging_guide.md](model_merging_guide.md#4-norm-stats-处理)。

### Q: 计算 norm_stats 需要 GPU 吗？

不需要。它只做数据加载和统计计算，纯 CPU 即可。但在 SLURM 脚本中通常和训练一起在 GPU 节点上执行。

### Q: 如果 norm_stats 计算错了会怎样？

训练可能不收敛或收敛很慢（相当于给模型喂了分布偏移的数据）。推理时表现为动作输出异常（幅度过大/过小、方向错误），机器人行为随机。
