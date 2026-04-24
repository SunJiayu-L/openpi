# OpenPI LIBERO 训练手册

> 本手册适用于在 yukaichenglab 集群上使用 OpenPI 框架对 LIBERO 数据集进行 VLA 模型微调。

---

## 目录

1. [集群环境](#1-集群环境)
2. [项目结构](#2-项目结构)
3. [数据集](#3-数据集)
4. [训练配置](#4-训练配置)
5. [训练流程](#5-训练流程)
6. [评测流程](#6-评测流程)
7. [常见问题与排错](#7-常见问题与排错)
8. [实验记录](#8-实验记录)

---

## 1. 集群环境

### 1.1 硬件

| 项目 | 规格 |
|------|------|
| GPU | NVIDIA H800 80GB |
| 每节点 GPU 数 | 8 |
| 内存 | 2TB / 节点 |
| 分区 | `yukaichenglab` |
| 可用节点 | gnho009, gnho031, gnho034 |

### 1.2 节点注意事项

| 节点 | 说明 |
|------|------|
| **gnho034** | 训练首选节点，EGL 渲染正常 |
| **gnho031** | 评测首选节点，EGL 渲染正常，1 GPU 即可 |
| **gnho009** | **EGL 渲染不可用**，会导致 MuJoCo core dump，不要用于评测 |

### 1.3 环境变量模板

```bash
# 离线模式（GPU 节点无外网）
export HF_HOME=~/.cache/huggingface
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled

# MuJoCo 无头渲染（评测时必需）
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export PYOPENGL_PLATFORM=egl
```

> **警告**: `MUJOCO_GL=glx` 在无头节点上不可用（无 X display）。始终使用 `egl`。

---

## 2. 项目结构

```
openpi/
├── src/openpi/
│   ├── training/
│   │   ├── config.py          # 所有训练配置定义
│   │   ├── data_loader.py     # 数据加载（含 Subset bug fix）
│   │   └── libero_split.py    # Train/test 拆分的 episode 列表
│   ├── models/                # JAX 模型实现
│   ├── policies/
│   │   └── libero_policy.py   # LIBERO 数据映射（LiberoInputs/Outputs）
│   └── serving/               # 推理服务
├── scripts/
│   ├── compute_norm_stats.py  # 归一化统计
│   ├── train.py               # JAX 训练入口
│   └── serve_policy.py        # 策略服务器
├── examples/libero/
│   ├── main.py                # 评测脚本
│   ├── .venv/                 # LIBERO Python 3.8 虚拟环境
│   └── README.md
├── checkpoints/               # 训练产出的 checkpoint
├── assets/                    # 归一化统计文件
└── third_party/libero/        # LIBERO 环境代码
```

### 关键路径

| 用途 | 路径 |
|------|------|
| 数据集 | `/storage/yukaichengLab/lishiwen/jiayusun/libero` |
| pi0 预训练权重 | `/storage/yukaichengLab/lishiwen/jiayusun/openpi_pt/pi0_base/` |
| pi0.5 预训练权重 | `/storage/yukaichengLab/lishiwen/jiayusun/openpi_pt/pi05_base/` |
| 训练日志 | `trainlogs/` |
| 评测视频 | `data/libero/videos/` |

---

## 3. 数据集

### 3.1 LIBERO 数据集概览

- 格式: LeRobot v2.0（Parquet + 图像）
- 本地路径: `/storage/yukaichengLab/lishiwen/jiayusun/libero`
- 总计: **1693 episodes, 273,465 frames, 40 tasks**
- 观测: `image` (224×224), `wrist_image` (224×224), `state` (8D)
- 动作: `actions` (7D)

### 3.2 四个 Suite

| Suite | task_index 范围 | 任务数 | Episodes | 特点 |
|-------|:---:|:---:|:---:|------|
| libero_10 | 0-9 | 10 | ~420 | 长 horizon 多样任务 |
| libero_goal | 10-19 | 10 | ~420 | 相同场景不同目标 |
| libero_object | 20-29 | 10 | ~420 | 相同布局不同物体 |
| libero_spatial | 30-39 | 10 | ~430 | 相同物体不同空间位置 |

### 3.3 Dataset task_index 与 Benchmark suite_task_id 的映射

**重要**: 这两个 ID 系统**不同**！dataset 中的 task_index 是按 suite 顺序排列的 (0-39)，但 LIBERO benchmark 中每个 suite 内部的 task 顺序是打乱的。

评测时必须使用正确的 suite_task_id：

| Dataset task_index | Suite | Suite task_id | 任务描述 |
|:---:|---------|:---:|------|
| 0 | libero_10 | 4 | put the white mug on the left plate... |
| 10 | libero_goal | 8 | pick up the black bowl next to the cookie box... |
| 20 | libero_object | 9 | pick up the orange juice and place it in the basket |
| 30 | libero_spatial | 6 | put the bowl on the plate |

> 如果要测试特定 task，必须先查清映射关系，否则会测错任务。

### 3.4 数据过滤（Subset Bug Fix）

LeRobot 的 `episodes=` 参数存在 bug：`episode_data_index` 按位置索引，但 `__getitem__` 使用原始 `episode_index`，导致 `IndexError`。

**解决方案**: 在 `data_loader.py` 中加载完整数据集后，用 `torch.utils.data.Subset` 过滤：

```python
# 正确做法：先加载全量，再用 Subset 过滤
dataset = lerobot_dataset.LeRobotDataset(repo_id, delta_timestamps=...)
if data_config.episodes is not None:
    all_ep_indices = torch.stack(dataset.hf_dataset["episode_index"]).numpy()
    mask = np.isin(all_ep_indices, list(set(data_config.episodes)))
    indices = np.where(mask)[0].tolist()
    dataset = torch.utils.data.Subset(dataset, indices)
```

---

## 4. 训练配置

### 4.1 可用配置一览

所有配置在 `src/openpi/training/config.py` 中定义：

| Config Name | 模型 | 数据 | 步数 | Batch Size |
|-------------|------|------|:---:|:---:|
| `pi0_libero` | π₀ (full) | 全部 40 tasks | 30,000 | 32 |
| `pi0_libero_low_mem_finetune` | π₀ (LoRA) | 全部 40 tasks | 30,000 | 32 |
| `pi05_libero` | π₀.₅ (full) | 全部 40 tasks | 30,000 | 256 |
| `pi05_libero_no10` | π₀.₅ (full) | 36 tasks (去除测试集) | 30,000 | 256 |

### 4.2 π₀ vs π₀.₅ 关键差异

| 参数 | π₀ (`pi0_libero`) | π₀.₅ (`pi05_libero`) |
|------|------|------|
| Action Horizon | 7 | 10 |
| Batch Size | 32 | 256 |
| LR Schedule | 默认 (peak=1e-3) | CosineDecay (peak=5e-5) |
| EMA Decay | 0.99 | 0.999 |
| Extra Delta Transform | True | False |
| 预训练权重 | `openpi_pt/pi0_base/params` | `openpi_pt/pi05_base/params` |

### 4.3 新增训练配置（单 Suite 微调）

在 `config.py` 中添加新配置，示例（以 libero_spatial 为例）：

```python
TrainConfig(
    name="pi0_libero_spatial",
    model=pi0_config.Pi0Config(),
    data=LeRobotLiberoDataConfig(
        repo_id="physical-intelligence/libero",
        base_config=DataConfig(
            prompt_from_task=True,
            episodes=LIBERO_SPATIAL_EPISODES,  # 只包含 task_index 30-39 的 episodes
        ),
        extra_delta_transform=True,
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "/storage/yukaichengLab/lishiwen/jiayusun/openpi_pt/pi0_base/params"
    ),
    num_train_steps=30_000,
),
```

### 4.4 离线权重路径

GCS 路径在 GPU 节点上不可用，必须改为本地路径：

```python
# 错误（无外网）
weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params")

# 正确
weight_loader=weight_loaders.CheckpointWeightLoader("/storage/yukaichengLab/lishiwen/jiayusun/openpi_pt/pi0_base/params")
```

---

## 5. 训练流程

### 5.1 完整训练步骤

```bash
# Step 1: 计算归一化统计（必须在训练前执行）
uv run scripts/compute_norm_stats.py --config-name <config_name>

# Step 2: 训练
uv run scripts/train.py <config_name> \
    --exp-name=<experiment_name> \
    --fsdp-devices 8
```

### 5.2 SLURM 训练脚本模板

```bash
#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:8
#SBATCH --nodelist=gnho034
#SBATCH -J <job_name>
#SBATCH -o trainlogs/<job_name>.%j.log
#SBATCH -e trainlogs/<job_name>.%j.err
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi

export HF_HOME=~/.cache/huggingface
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled

# Step 1: Norm stats
echo "Computing normalization statistics..."
uv run scripts/compute_norm_stats.py --config-name <config_name>

# Step 2: Train
echo "Starting training..."
uv run scripts/train.py <config_name> \
    --exp-name=<exp_name> \
    --fsdp-devices 8

echo "End time: $(date)"
```

### 5.3 train.py 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--exp-name` | (必填) | 实验名，决定 checkpoint 保存路径 |
| `--fsdp-devices` | 1 | FSDP 分片设备数，8 GPU 设为 8 |
| `--overwrite` | False | 覆盖已有 checkpoint 目录 |
| `--resume` | False | 从上次 checkpoint 恢复训练 |
| `--batch-size` | config 默认 | 覆盖 config 中的 batch_size |
| `--num-train-steps` | config 默认 | 覆盖训练步数 |

### 5.4 Checkpoint 保存策略

- 每 **1000 步** 保存一次 (`save_interval=1000`)
- 只保留 **5000 步整数倍** 的 checkpoint (`keep_period=5000`)
- 最终步 (如 29999) 始终保留
- 保存路径: `checkpoints/<config_name>/<exp_name>/<step>/`

### 5.5 训练速度参考

| 模型 | GPU 数 | 速度 |
|------|:---:|------|
| π₀.₅ (`pi05_libero`) | 8×H800 | ~4.2-4.4 s/step |
| π₀ (`pi0_libero`) | 8×H800 | 待确认 |

30,000 步 × 4.3 s/step ≈ **35 小时**

---

## 6. 评测流程

### 6.1 评测架构

评测采用 client-server 模式：
1. **Policy Server**: 加载 checkpoint，通过 WebSocket 提供推理服务
2. **Eval Client**: LIBERO 环境中执行 rollout，发送观测到 server 获取动作

### 6.2 启动 Policy Server

```bash
uv run scripts/serve_policy.py \
    --port 8000 \
    policy:checkpoint \
    --policy.config <config_name> \
    --policy.dir "checkpoints/<config_name>/<exp_name>/<step>"
```

### 6.3 运行评测

```bash
# 激活 LIBERO 环境（Python 3.8）
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PWD/third_party/libero:$PYTHONPATH

# 评测单个 suite 的全部 task
python examples/libero/main.py \
    --args.task-suite-name libero_spatial \
    --args.num-trials-per-task 20 \
    --args.video-out-path "data/libero/videos/output_dir" \
    --args.host 0.0.0.0 \
    --args.port 8000

# 评测单个 suite 中的指定 task（注意使用 suite_task_id，不是 dataset task_index）
python examples/libero/main.py \
    --args.task-suite-name libero_spatial \
    --args.num-trials-per-task 50 \
    --args.task-ids 6 \
    --args.host 0.0.0.0 \
    --args.port 8000
```

### 6.4 评测参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--args.task-suite-name` | libero_spatial | 评测 suite |
| `--args.num-trials-per-task` | 50 | 每个 task 的 rollout 次数 |
| `--args.task-ids` | (全部) | 指定 suite 内 task id（0-indexed） |
| `--args.video-out-path` | data/libero/videos | 视频输出路径 |
| `--args.host` | 0.0.0.0 | Policy server 地址 |
| `--args.port` | 8000 | Policy server 端口 |

### 6.5 各 Suite 最大 Episode 长度

| Suite | Max Steps | 说明 |
|-------|:---------:|------|
| libero_spatial | 220 | 最短 |
| libero_object | 280 | |
| libero_goal | 300 | |
| libero_10 | 520 | 最长，评测最慢 |

### 6.6 SLURM 评测脚本模板

```bash
#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gnho031
#SBATCH -J <eval_job_name>
#SBATCH -o trainlogs/<eval_name>.%j.log
#SBATCH -e trainlogs/<eval_name>.%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

set -eo pipefail

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi

export HF_HOME=~/.cache/huggingface
export HF_HUB_OFFLINE=1
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export PYOPENGL_PLATFORM=egl

LIBERO_VENV=examples/libero/.venv
NUM_TRIALS=20
PORT=8000
SUITES=("libero_10" "libero_goal" "libero_object" "libero_spatial")

for STEP in 5000 10000 15000 20000 25000 29999; do
    CKPT_DIR="checkpoints/<config_name>/<exp_name>/${STEP}"

    # 启动 policy server
    uv run scripts/serve_policy.py \
        --port $PORT \
        policy:checkpoint \
        --policy.config <config_name> \
        --policy.dir "$CKPT_DIR" &
    SERVER_PID=$!

    # 等待 server 就绪
    for i in $(seq 1 300); do
        if curl -s http://localhost:${PORT}/healthz > /dev/null 2>&1; then
            echo "Server ready (waited ${i}s)"
            break
        fi
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "ERROR: Server died"; exit 1
        fi
        sleep 1
    done

    # 评测
    source "$LIBERO_VENV/bin/activate"
    export PYTHONPATH=$PWD/third_party/libero:$PYTHONPATH

    for SUITE in "${SUITES[@]}"; do
        python examples/libero/main.py \
            --args.task-suite-name "$SUITE" \
            --args.num-trials-per-task "$NUM_TRIALS" \
            --args.video-out-path "data/libero/videos/<eval_name>_step${STEP}/${SUITE}" \
            --args.host 0.0.0.0 \
            --args.port $PORT
    done

    deactivate

    # 关闭 server（必须加 || true，否则 set -e 会导致脚本退出）
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
done
```

> **关键**: `kill` 和 `wait` 后必须加 `|| true`，否则 `set -eo pipefail` 下被 kill 的进程返回非零码会导致脚本退出。

### 6.7 评测时间估算

| 配置 | 预计时间 |
|------|---------|
| 1 suite × 10 tasks × 20 trials | ~10-15 分钟 |
| 4 suites × 10 tasks × 20 trials | ~45-60 分钟 |
| 1 checkpoint 全量评测 | ~45-60 分钟 |
| 6 checkpoints 全量评测 | ~5-6 小时 |

### 6.8 结果提取

```bash
# 每 50 trial（每个 task）的最终成功率
grep "successes:" trainlogs/<eval>.err | awk 'NR%50==0'

# 每 200 行 = 1 suite（10 tasks × 20 trials）
grep "successes:" trainlogs/<eval>.err | awk 'NR%200==0'

# 检查崩溃
grep -i "abort\|core dump" trainlogs/<eval>.err
```

---

## 7. 常见问题与排错

### 7.1 LeRobot episodes= IndexError

**症状**: `IndexError: index 1515 is out of bounds for dimension 0 with size 1515`
**原因**: LeRobot v2.0 的 `episodes=` 参数 bug
**解决**: 使用 `torch.utils.data.Subset` 过滤（已在 data_loader.py 中修复）

### 7.2 MuJoCo Core Dump

**症状**: `Aborted (core dumped)` 在评测时
**原因**: 某些节点 (gnho009) EGL 渲染不可用
**解决**: 使用 gnho031 或 gnho034 评测

### 7.3 Python 3.8 类型注解

**症状**: `TypeError: 'type' object is not subscriptable`
**原因**: LIBERO venv 使用 Python 3.8，不支持 `tuple[int, ...] | None` 语法
**解决**: 使用 `typing.Optional[typing.Tuple[int, ...]]` 并 `import typing`

### 7.4 eval 脚本中途退出

**症状**: 第一个 checkpoint 评完就停止
**原因**: `set -eo pipefail` 下 `kill $SERVER_PID` 返回非零码
**解决**: `kill` 和 `wait` 后加 `|| true`

### 7.5 GCS 路径不可用

**症状**: `gs://openpi-assets/...` 下载超时
**原因**: GPU 节点无外网
**解决**: config.py 中改为本地路径 `/storage/yukaichengLab/lishiwen/jiayusun/openpi_pt/...`

### 7.6 Checkpoint 目录已存在

**症状**: `FileExistsError`
**解决**: 使用 `--overwrite` 或换一个 `--exp-name`

### 7.7 评测视频目录不存在

**症状**: `FileNotFoundError: The directory '...' does not exist`
**原因**: 脚本运行时移动了视频输出目录
**解决**: 不要在评测进行中移动 `data/libero/videos/` 下的目录

---

## 8. 实验记录

### 8.1 π₀.₅ 全量训练 (40 tasks)

- Config: `pi05_libero`, exp: `my_experiment`
- 节点: gnho034 (8×H800), 训练时间 ~35h
- Checkpoints: 5000, 10000, 15000, 20000, 25000, 29999

**全 task 评测结果 (20 trials/task)**:

| Step | libero_10 | libero_goal | libero_object | libero_spatial | 平均 |
|:---:|:---------:|:-----------:|:-------------:|:--------------:|:----:|
| 5000 | 91.5% | 98.5% | 99.0% | 97.0% | 96.5% |
| 10000 | 89.0% | 97.0% | 98.0% | 98.0% | 95.5% |
| 15000 | 88.5% | 96.0% | 98.5% | 98.0% | 95.3% |
| 20000 | 93.5% | 96.0% | 99.0% | 98.0% | 96.6% |
| 25000 | 93.5% | 96.5% | 98.0% | 99.5% | 96.9% |
| **29999** | **94.0%** | **97.5%** | **99.0%** | **97.5%** | **97.0%** |

### 8.2 π₀.₅ Split 训练 (36 tasks, 4 held-out)

- Config: `pi05_libero_no10`, exp: `split_experiment`
- 训练: 36 tasks (1515 eps), 测试: 4 tasks (dataset idx 0, 10, 20, 30)

**Held-out test task 评测 (50 trials/task)**:

| Step | libero_10 | libero_goal | libero_object | libero_spatial | 平均 |
|:---:|:---------:|:-----------:|:-------------:|:--------------:|:----:|
| 5000 | 4.0% | 98.0% | 0.0% | 6.0% | 27.0% |
| 10000 | 0.0% | 100.0% | 4.0% | 30.0% | 33.5% |
| **15000** | **0.0%** | **100.0%** | **6.0%** | **30.0%** | **34.0%** |

**发现**: libero_goal 泛化性好（所有 task 共享场景和物体，只是目标不同），其余 suite 的 zero-shot 泛化较差。

### 8.3 π₀ 全量训练 (进行中)

- Config: `pi0_libero`, exp: `my_experiment`
- Job: 31859, 节点: gnho034
- 状态: 训练中

---

## 附录 A: 单 Suite 微调快速指南

针对 4 个 suite 分别微调时，需要为每个 suite 创建独立的 config 和 episode 列表。

### A.1 获取每个 suite 的 episode 列表

```python
import pandas as pd
df = pd.read_parquet("/storage/yukaichengLab/lishiwen/jiayusun/libero/data/train-00000-of-00001.parquet")
ep_task = df.groupby("episode_index")["task_index"].first()

# 每个 suite 的 episodes
libero_10_eps = sorted(ep_task[ep_task.between(0, 9)].index.tolist())     # task 0-9
libero_goal_eps = sorted(ep_task[ep_task.between(10, 19)].index.tolist()) # task 10-19
libero_object_eps = sorted(ep_task[ep_task.between(20, 29)].index.tolist()) # task 20-29
libero_spatial_eps = sorted(ep_task[ep_task.between(30, 39)].index.tolist()) # task 30-39
```

### A.2 Config 模板

```python
# 在 config.py 中添加
TrainConfig(
    name="pi0_libero_spatial",  # 改为对应 suite
    model=pi0_config.Pi0Config(),
    data=LeRobotLiberoDataConfig(
        repo_id="physical-intelligence/libero",
        base_config=DataConfig(
            prompt_from_task=True,
            episodes=LIBERO_SPATIAL_EPISODES,  # 该 suite 的 episode 列表
        ),
        extra_delta_transform=True,
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "/storage/yukaichengLab/lishiwen/jiayusun/openpi_pt/pi0_base/params"
    ),
    num_train_steps=30_000,
),
```

### A.3 训练命令

```bash
# 对每个 suite 分别执行
uv run scripts/compute_norm_stats.py --config-name pi0_libero_spatial
uv run scripts/train.py pi0_libero_spatial --exp-name=my_experiment --fsdp-devices 8
```

### A.4 评测命令

```bash
# 启动 server
uv run scripts/serve_policy.py --port 8000 \
    policy:checkpoint \
    --policy.config pi0_libero_spatial \
    --policy.dir "checkpoints/pi0_libero_spatial/my_experiment/29999"

# 评测该 suite 全部 10 个 task
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PWD/third_party/libero:$PYTHONPATH
python examples/libero/main.py \
    --args.task-suite-name libero_spatial \
    --args.num-trials-per-task 20
```
