# OpenPI 权重融合指南

> 本文档说明如何在 openpi 中使用权重融合功能，将多个微调 checkpoint 合并为一个策略模型。
> 该功能移植自 [RETAIN_code](https://arxiv.org/abs/2512.08333)。

---

## 目录

1. [概述](#1-概述)
2. [新增/修改文件说明](#2-新增修改文件说明)
3. [融合算法](#3-融合算法)
4. [Norm Stats 处理](#4-norm-stats-处理)
5. [使用方法](#5-使用方法)
6. [LIBERO 融合实验示例](#6-libero-融合实验示例)
7. [SLURM 脚本模板](#7-slurm-脚本模板)
8. [已知限制与注意事项](#8-已知限制与注意事项)

---

## 1. 概述

权重融合将多个独立微调的 checkpoint 合并为一个模型，无需重新训练。典型场景：

```
pi05_base (预训练) ──┬── 微调 ──> pi05_libero_goal/29999
                     ├── 微调 ──> pi05_libero_object/29999
                     └── 微调 ──> pi05_libero_spatial/29999
                                        │
                              权重融合 (linear interpolation)
                                        │
                                        v
                              merged_policy (推理)
```

### 执行流程

```
merge_and_eval.sh
    │
    ├─ 1. 启动融合 server（后台）
    │      scripts/merging_experiments.py
    │          │
    │          ├─ 解析参数：config=pi05_libero, merging_fn=linear_interpolation
    │          ├─ 调用 policy_config.create_merged_policy()
    │          │      │
    │          │      ├─ 加载 N 个 checkpoint 参数到内存
    │          │      │    ckpt_object/29999  (权重 0.4)
    │          │      │    ckpt_goal/29999    (权重 0.4)
    │          │      │    pi05_base          (权重 0.2)
    │          │      │
    │          │      ├─ model_merging.linear_interpolation()
    │          │      │    merged = 0.4*obj + 0.4*goal + 0.2*base
    │          │      │
    │          │      ├─ 从 config assets_dirs 加载全量 norm_stats
    │          │      │    (assets/pi05_libero/ — 覆盖全部 40 task)
    │          │      │
    │          │      └─ 返回 Policy 对象（仅在内存中，不保存到磁盘）
    │          │
    │          └─ 启动 WebSocket server (port 8100)
    │
    ├─ 2. 等待 server 就绪（轮询 /healthz，最多 600s）
    │
    ├─ 3. 依次评测 4 个 suite（每个 10 tasks × 20 trials）
    │      libero_10 → libero_goal → libero_object → libero_spatial
    │
    └─ 4. 评测完成，kill server
```

**核心约定**: checkpoint_dirs 列表中，**最后一个**为预训练 base 模型，前面的为微调模型。

---

## 2. 新增/修改文件说明

### 2.1 `src/openpi/policies/model_merging.py`（新增）

融合算法实现，包含 6 种融合策略和统一注册表。从 RETAIN_code 移植，算法逻辑完全一致。

**注册表**:

```python
merging_functions = {
    "linear_interpolation": linear_interpolation,
    "slerp_interpolation": spherical_linear_interpolation,
    "multimodal_linear_interpolation": multimodal_linear_interpolation,
    "dare_interpolation": dare_interpolation,
    "dare_slerp_interpolation": dare_slerp_interpolation,
    "task_vector_interpolation": task_vector_interpolation,
}
```

**统一函数签名**:

```python
def merging_fn(
    train_config: TrainConfig,
    checkpoint_dirs: list[Path | str],
    merging_fn_kwargs: dict[str, Any] | None = None,
) -> at.Params  # 返回融合后的 model
```

**工具函数**:

| 函数 | 作用 |
|------|------|
| `validate_merging_coefficients` | 校验系数在 [0,1] 且总和为 1 |
| `validate_params_list` | 校验所有 checkpoint 参数树结构一致 |
| `linearly_merge_params` | 对参数树做加权平均（核心原语） |

### 2.2 `src/openpi/policies/policy_config.py`（修改）

新增 `create_merged_policy()` 函数：

```python
from openpi.policies.model_merging import merging_functions

def create_merged_policy(
    train_config, checkpoint_dirs, merging_fn, merging_fn_kwargs,
    *, default_prompt=None, norm_stats=None,
) -> Policy:
```

**流程**:
1. 下载/定位所有 checkpoint
2. 调用注册表中的融合函数 → 返回融合后的 model
3. **从 `train_config.assets_dirs` 加载全量 norm_stats**（不是从单个 checkpoint）
4. 构建 Policy（transforms pipeline 与 `create_trained_policy` 一致）

### 2.3 `scripts/merging_experiments.py`（新增）

CLI 入口，融合 + 启动 WebSocket policy server：

```python
@dataclasses.dataclass
class Args:
    port: int                    # server 端口
    config: str                  # 训练配置名 (如 "pi05_libero")
    merging_fn: str              # 融合方法名
    merging_fn_kwargs: str       # JSON 格式的融合参数
    checkpoint_dirs: list[str]   # checkpoint 路径列表（最后一个为 base）
    default_prompt: str | None   # 可选默认 prompt
```

### 2.4 `merge_and_eval.sh`（新增）

一体化 SLURM 脚本：启动融合 server → 等待就绪 → 评测 4 个 suite → 清理。

---

## 3. 融合算法

### 3.1 Linear Interpolation（线性插值）

最常用，RETAIN 论文中所有实验均使用此方法。

```
merged_params = Σ(wᵢ × paramsᵢ)
```

**参数**: `model_mixing_coefficients: list[float]` — 每个 checkpoint 的权重，总和为 1。

**示例**: 3 个 checkpoint，系数 [0.4, 0.4, 0.2]
- checkpoint_dirs[0] (微调模型 A) × 0.4
- checkpoint_dirs[1] (微调模型 B) × 0.4
- checkpoint_dirs[2] (base 模型) × 0.2

### 3.2 Spherical Linear Interpolation (SLERP)

在参数空间的超球面上插值，保持方向信息。

```
merged = sin(θ-θt)/sin(θ) × base + sin(θt)/sin(θ) × finetuned
```

**限制**: 仅支持 2 个 checkpoint。
**参数**: `t: float` — 插值系数 [0,1]，越大越偏向微调模型。

### 3.3 Multimodal Linear Interpolation（多模态线性插值）

对 pi0 模型的三个模块使用**独立**融合系数：

| 模块 | 参数键模式 | 系数参数 |
|------|-----------|---------|
| Vision Encoder | `img` | `vision_mixing_coefficients` |
| LLM Backbone | `llm` (排除 `_1`) | `llm_mixing_coefficients` |
| Action Expert | `llm` + `_1` | `action_expert_mixing_coefficients` |

辅助投影层 (action_in/out_proj, time_mlp, state_proj) 使用 action expert 系数。

### 3.4 Task Vector Interpolation

```
task_vector_i = finetuned_i - base
merged = base + Σ(λᵢ × task_vectorᵢ)
```

**限制**: ≥3 个 checkpoint（至少 2 个微调 + 1 个 base）。
**参数**: `lambda_list: list[float]` — 每个 task vector 的缩放系数。

### 3.5 DARE (Drop And REscale)

对 task vector 随机 dropout 后 rescale，保持期望不变。

**限制**: 仅支持 2 个 checkpoint。
**参数**:
- `dropout_prob: float` — dropout 概率
- `seed: int` — 随机种子
- `task_vector_scaling_factor: float` — task vector 注入强度

### 3.6 DARE + SLERP

先做 DARE dropout/rescale，然后用 SLERP 融合 base 和处理后的微调参数。

---

## 4. Norm Stats 处理

### 4.1 问题背景

不同 LIBERO suite 的数据分布不同，各 suite 微调 checkpoint 携带的 norm_stats 也不同：

| Suite | state mean (前3维) |
|-------|-------------------|
| libero_goal | [-0.099, 0.014, 1.070] |
| libero_object | [-0.030, -0.008, 0.203] |
| libero_spatial | [-0.024, 0.107, 1.058] |

如果融合模型使用某个单 suite 的 norm_stats，在其他 suite 上推理时归一化会不正确。

### 4.2 解决方案

`create_merged_policy()` **从 `train_config.assets_dirs` 加载 norm_stats**，即使用 `--config` 指定的配置的全量 assets 目录（如 `assets/pi05_libero/`），而非从任何单个 checkpoint 加载。

```python
# policy_config.py — create_merged_policy()
# 从 config assets dir 加载全量 norm_stats（覆盖所有 suite）
norm_stats = _checkpoints.load_norm_stats(train_config.assets_dirs, data_config.asset_id)
```

**关键**：`--config pi05_libero` 的 assets 目录包含基于全部 40 task（1693 episodes）计算的归一化统计，能正确覆盖所有 suite 的数据分布。

### 4.3 与 RETAIN 的差异

RETAIN 论文假设所有微调模型共享相同的训练数据分布，因此直接使用任一 checkpoint 的 norm_stats。但在 LIBERO 跨 suite 融合场景下，各 suite 数据分布不同，必须使用全量 norm_stats。这是我们相比 RETAIN 的重要修改。

### 4.4 如何确保 norm_stats 正确

1. **融合前**：确认 `assets/{config_name}/{repo_id}/norm_stats.json` 存在
2. **`--config` 选择**：使用全量配置（如 `pi05_libero`），不要用单 suite 配置（如 `pi05_libero_goal`）
3. **如果 norm_stats 不存在**：先运行 `uv run scripts/compute_norm_stats.py --config-name pi05_libero`

---

## 5. 使用方法

### 5.1 CLI 命令

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/merging_experiments.py \
    --port <port> \
    --config <config_name> \
    --merging_fn <method_name> \
    --merging_fn_kwargs '<json_params>' \
    --checkpoint_dirs <ckpt1> <ckpt2> ... <base>
```

### 5.2 参数说明

| 参数 | 说明 |
|------|------|
| `--port` | WebSocket server 端口 |
| `--config` | 训练配置名（决定模型架构、transforms 和 **norm_stats 来源**） |
| `--merging_fn` | 融合方法名（见注册表） |
| `--merging_fn_kwargs` | JSON 字符串，传给融合函数的参数 |
| `--checkpoint_dirs` | checkpoint 路径列表，**最后一个为 base** |

### 5.3 Python API

```python
from openpi.training import config as _config
from openpi.policies import policy_config

config = _config.get_config("pi05_libero")  # 全量配置 → 全量 norm_stats
policy = policy_config.create_merged_policy(
    config,
    checkpoint_dirs=[
        "checkpoints/pi05_libero_object/my_experiment/29999",
        "checkpoints/pi05_libero_goal/my_experiment/29999",
        "/path/to/pi05_base",
    ],
    merging_fn="linear_interpolation",
    merging_fn_kwargs={"model_mixing_coefficients": [0.4, 0.4, 0.2]},
)
# policy.infer(observation) 即可推理
```

---

## 6. LIBERO 融合实验示例

### 6.1 融合 + 启动 server

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/merging_experiments.py \
    --port 8100 \
    --config pi05_libero \
    --merging_fn linear_interpolation \
    --merging_fn_kwargs '{"model_mixing_coefficients": [0.4, 0.4, 0.2]}' \
    --checkpoint_dirs \
        checkpoints/pi05_libero_object/my_experiment/29999 \
        checkpoints/pi05_libero_goal/my_experiment/29999 \
        /storage/yukaichengLab/lishiwen/.cache/openpi/openpi-assets/checkpoints/pi05_base
```

### 6.2 评测（另一个终端/脚本）

```bash
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PWD/third_party/libero:$PYTHONPATH
export MUJOCO_GL=egl

python examples/libero/main.py \
    --args.task-suite-name libero_goal \
    --args.num-trials-per-task 20 \
    --args.host 0.0.0.0 \
    --args.port 8100
```

### 6.3 不同融合策略示例

**SLERP (2 个 checkpoint)**:
```bash
--merging_fn slerp_interpolation \
--merging_fn_kwargs '{"t": 0.7}' \
--checkpoint_dirs ckpt_finetuned ckpt_base
```

**Task Vector (多个微调 + base)**:
```bash
--merging_fn task_vector_interpolation \
--merging_fn_kwargs '{"lambda_list": [0.5, 0.3]}' \
--checkpoint_dirs ckpt_goal ckpt_object ckpt_base
```

**多模态独立系数**:
```bash
--merging_fn multimodal_linear_interpolation \
--merging_fn_kwargs '{
    "vision_mixing_coefficients": [0.5, 0.5],
    "llm_mixing_coefficients": [0.3, 0.7],
    "action_expert_mixing_coefficients": [0.6, 0.4]
}' \
--checkpoint_dirs ckpt_finetuned ckpt_base
```

---

## 7. SLURM 脚本模板

```bash
#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gnho034
#SBATCH -J merge_eval
#SBATCH -o trainlogs/merge_eval.%j.log
#SBATCH -e trainlogs/merge_eval.%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

set -eo pipefail

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi

export HF_HOME=~/.cache/huggingface
export HF_HUB_OFFLINE=1
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export PYOPENGL_PLATFORM=egl

PORT=8100

CKPT_OBJ="checkpoints/pi05_libero_object/my_experiment/29999"
CKPT_GOAL="checkpoints/pi05_libero_goal/my_experiment/29999"
CKPT_BASE="/storage/yukaichengLab/lishiwen/.cache/openpi/openpi-assets/checkpoints/pi05_base"

# 启动融合 server（后台）
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/merging_experiments.py \
    --port $PORT \
    --config pi05_libero \
    --merging_fn linear_interpolation \
    --merging_fn_kwargs '{"model_mixing_coefficients": [0.4, 0.4, 0.2]}' \
    --checkpoint_dirs "$CKPT_OBJ" "$CKPT_GOAL" "$CKPT_BASE" &
SERVER_PID=$!

# 等待 server 就绪
for i in $(seq 1 600); do
    if curl -s http://localhost:${PORT}/healthz > /dev/null 2>&1; then
        echo "Server ready! (waited ${i}s)"; break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: Server died"; exit 1
    fi
    sleep 1
done

# 评测所有 4 个 suite
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PWD/third_party/libero:$PYTHONPATH

for SUITE in libero_10 libero_goal libero_object libero_spatial; do
    echo "Evaluating ${SUITE}..."
    python examples/libero/main.py \
        --args.task-suite-name "$SUITE" \
        --args.num-trials-per-task 20 \
        --args.host 0.0.0.0 \
        --args.port $PORT
done

deactivate
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
```

---

## 8. 已知限制与注意事项

### 8.1 Norm Stats — 必须使用全量配置

融合跨 suite checkpoint 时，**必须使用全量配置**（如 `pi05_libero`）的 norm_stats。使用单 suite 的 norm_stats 会导致推理时归一化错误，表现为其他 suite 上成功率接近 0%。

| 场景 | --config | norm_stats 来源 |
|------|----------|----------------|
| 融合 goal + object + base | `pi05_libero` | `assets/pi05_libero/` (全量 40 task) |
| 单个 checkpoint 推理 | `pi05_libero_goal` | checkpoint 内 `assets/` 目录 |

### 8.2 Config 选择

`--config` 参数决定模型架构、transforms 和 norm_stats。融合时应使用全量配置：

| 微调 checkpoint | 推荐 --config |
|----------------|--------------|
| pi05_libero_goal, pi05_libero_object, ... | `pi05_libero` |
| pi0_libero_goal, pi0_libero_object, ... | `pi0_libero` |

### 8.3 系数约束

所有使用 `model_mixing_coefficients` 的方法要求系数**总和为 1.0**，每个系数在 [0, 1] 范围内。

### 8.4 Checkpoint 顺序

**最后一个 checkpoint_dir 始终为 base 模型**。这是所有融合函数的约定，尤其影响 task vector 和 DARE 方法。

### 8.5 融合权重不保存到磁盘

当前实现中，融合后的权重**仅存在于 server 进程内存中**。server 被 kill 后权重消失。如需持久化，需额外添加 checkpoint 保存逻辑。

### 8.6 GPU 内存

融合过程需要同时加载所有 checkpoint 的参数到内存：
- 2 个 checkpoint: ~16 GB (bfloat16)
- 3 个 checkpoint: ~24 GB
- 4 个 checkpoint: ~32 GB
- 建议使用 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`
- 推理（评测）只需 1 张 GPU

### 8.7 FSDP Checkpoint 恢复限制

训练 checkpoint 使用 FSDP 分片保存，**恢复时必须使用相同数量的 GPU**。例如用 8 GPU 训练的 checkpoint 不能用 4 GPU 恢复。这不影响融合（融合直接读取 params，不涉及 FSDP），但影响断点续训。

### 8.8 与 RETAIN_code 的差异

本移植版本相比 RETAIN_code：
- 去除了 `debug_dir` 硬编码路径
- 去除了 `data_list` cotraining hack（openpi 不需要）
- **改为从 config assets_dirs 加载全量 norm_stats**（RETAIN 从单个 checkpoint 加载）
- 保留了 openpi 原有的 `is_pytorch` / `pytorch_device` 支持
- 融合算法 (`model_merging.py`) 完全一致，无任何修改
