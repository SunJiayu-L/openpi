# LIBERO Train/Test Split 实验文档

## 1. 任务目标

在 LIBERO 的 40 个任务（4 个 suite，每个 10 个 task）中，**取出每个 suite 的第 1 个 task（按数据集 task_index 顺序）作为测试集**，其余 36 个 task 作为训练集，微调 pi0.5 模型并在测试集上评估泛化性能。

## 2. 数据集分析

### 2.1 数据集基本信息

数据集路径：`/storage/yukaichengLab/lishiwen/jiayusun/libero/`（LeRobot v2.0 格式）

| 项目 | 数值 |
|------|------|
| 总 episodes | 1,693 |
| 总帧数 | 273,465 |
| 总任务数 | 40 |
| FPS | 10 |
| 特征 | image(256x256x3), wrist_image(256x256x3), state(8D), actions(7D) |

### 2.2 数据集 task_index 与 suite 的对应关系

**关键发现：数据集的 task_index 排列与 LIBERO benchmark 库的 suite 内部排列不同。**

数据集中 task_index 按以下规则分布：

| 数据集 task_index | 所属 Suite | 说明 |
|:-:|:-:|:-:|
| 0 - 9 | libero_10 | 混合长 horizon 任务 |
| 10 - 19 | **libero_goal** | 目标导向任务 |
| 20 - 29 | **libero_object** | 物体操作任务 |
| 30 - 39 | **libero_spatial** | 空间推理任务 |

> 注意：task_index 10-19 对应的是 libero_goal（不是 libero_spatial），30-39 对应 libero_spatial（不是 libero_goal）。

### 2.3 测试集（4 个 task，178 episodes）

| 数据集 task_index | Suite | Benchmark suite_task_id | 任务描述 | Episodes |
|:-:|:-:|:-:|:-:|:-:|
| 0 | libero_10 | 4 | put the white mug on the left plate and put the yellow and white mug on the right plate | 38 |
| 10 | libero_goal | 8 | put the bowl on the plate | 49 |
| 20 | libero_object | 9 | pick up the orange juice and place it in the basket | 45 |
| 30 | libero_spatial | 6 | pick up the black bowl next to the cookie box and place it on the plate | 46 |

### 2.4 训练集（36 个 task，1,515 episodes，247,029 帧）

所有 task_index 除 {0, 10, 20, 30} 之外的 36 个 task。

### 2.5 数据集 task_index 与 benchmark suite_task_id 完整映射

因为 LIBERO benchmark 库内部的 task 排列顺序与数据集不同，评估时需要使用正确的 suite_task_id。以下是完整映射：

```
# libero_10 (dataset task_index 0-9)
task_index 0 -> suite_task 4 [TEST]    task_index 5 -> suite_task 0
task_index 1 -> suite_task 6           task_index 6 -> suite_task 8
task_index 2 -> suite_task 9           task_index 7 -> suite_task 1
task_index 3 -> suite_task 2           task_index 8 -> suite_task 3
task_index 4 -> suite_task 7           task_index 9 -> suite_task 5

# libero_goal (dataset task_index 10-19)
task_index 10 -> suite_task 8 [TEST]   task_index 15 -> suite_task 5
task_index 11 -> suite_task 9          task_index 16 -> suite_task 7
task_index 12 -> suite_task 3          task_index 17 -> suite_task 1
task_index 13 -> suite_task 6          task_index 18 -> suite_task 4
task_index 14 -> suite_task 2          task_index 19 -> suite_task 0

# libero_object (dataset task_index 20-29)
task_index 20 -> suite_task 9 [TEST]   task_index 25 -> suite_task 7
task_index 21 -> suite_task 4          task_index 26 -> suite_task 2
task_index 22 -> suite_task 1          task_index 27 -> suite_task 6
task_index 23 -> suite_task 3          task_index 28 -> suite_task 5
task_index 24 -> suite_task 0          task_index 29 -> suite_task 8

# libero_spatial (dataset task_index 30-39)
task_index 30 -> suite_task 6 [TEST]   task_index 35 -> suite_task 3
task_index 31 -> suite_task 4          task_index 36 -> suite_task 8
task_index 32 -> suite_task 5          task_index 37 -> suite_task 1
task_index 33 -> suite_task 7          task_index 38 -> suite_task 2
task_index 34 -> suite_task 0          task_index 39 -> suite_task 9
```

## 3. 代码修改

### 3.1 新增文件

#### `src/openpi/training/libero_split.py`

定义训练/测试 episode 索引列表：
- `TRAIN_EPISODES`: 1,515 个 episode index（36 个训练 task）
- `TEST_EPISODES`: 178 个 episode index（4 个测试 task）

通过遍历 `libero/meta/episodes.jsonl` 和 `tasks.jsonl`，将每个 episode 按其 task 描述映射到 task_index，然后根据 task_index 是否为 {0, 10, 20, 30} 分入测试集或训练集。

### 3.2 修改文件

#### `src/openpi/training/config.py`

新增 `pi05_libero_no10` 训练配置：

```python
import openpi.training.libero_split as _libero_split

TrainConfig(
    name="pi05_libero_no10",
    model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
    data=LeRobotLiberoDataConfig(
        repo_id="physical-intelligence/libero",
        base_config=DataConfig(
            prompt_from_task=True,
            episodes=_libero_split.TRAIN_EPISODES,  # 只用训练集的 1515 个 episodes
        ),
        extra_delta_transform=False,
    ),
    ...
)
```

`DataConfig` 中新增 `episodes` 字段：

```python
@dataclasses.dataclass
class DataConfig:
    ...
    episodes: list[int] | None = None  # 如果提供，只使用这些 episode
```

#### `src/openpi/training/data_loader.py`

修改 `create_torch_dataset()` 实现 episode 过滤：

```python
def create_torch_dataset(...):
    # 加载完整数据集（全部 1693 episodes）
    dataset = LeRobotDataset(repo_id, delta_timestamps={...})

    # 再通过 Subset 过滤帧
    if data_config.episodes is not None:
        episode_set = set(data_config.episodes)
        all_ep_indices = torch.stack(dataset.hf_dataset["episode_index"]).numpy()
        mask = np.isin(all_ep_indices, list(episode_set))
        indices = np.where(mask)[0].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)
    ...
```

> **为什么不直接传 `episodes=` 给 LeRobotDataset？**
> LeRobot 存在 bug：当传入 `episodes=` 参数时，内部的 `episode_data_index` 按位置索引（0 到 N-1），但 `__getitem__` 中取到的 `episode_index` 是原始值（可能远大于 N），导致与 `delta_timestamps` 配合使用时抛出 `IndexError`。
> 解决方案：加载完整数据集（`episode_data_index` 覆盖全部 1693 个 episode），通过 `Subset` 限制 DataLoader 只采样训练集的帧。

#### `examples/libero/main.py`

新增 `task_ids` 参数用于评估指定 task：

```python
import typing

@dataclasses.dataclass
class Args:
    ...
    task_ids: typing.Optional[typing.Tuple[int, ...]] = None

def eval_libero(args):
    ...
    task_id_list = list(args.task_ids) if args.task_ids is not None else list(range(num_tasks_in_suite))
    for task_id in tqdm.tqdm(task_id_list):
        ...
```

> 使用 `typing.Optional[typing.Tuple[int, ...]]` 而非 `tuple[int, ...] | None`，因为 LIBERO 的 venv 是 Python 3.8。

### 3.3 脚本文件

#### `train_pi05_libero_no10.sh`（训练脚本）

```bash
#SBATCH --gres=gpu:8
# Step 1: 计算归一化统计量
uv run scripts/compute_norm_stats.py --config-name pi05_libero_no10
# Step 2: 训练 30,000 步
uv run scripts/train.py pi05_libero_no10 --exp-name=split_experiment --fsdp-devices 8
```

#### `eval_pi05_libero_split.sh`（评估脚本）

在单节点上依次评估 3 个 checkpoint（5k/10k/14k），每个 checkpoint 评估 4 个测试 task：

```bash
SUITES=("libero_10" "libero_goal" "libero_object" "libero_spatial")
SUITE_TASK_IDS=(4 8 9 6)  # 正确的 benchmark suite 内部编号
```

每个 checkpoint 的流程：启动 policy server → 评估 4 个 suite 的测试 task（各 50 trials）→ 关闭 server。

## 4. 运行情况

### 4.1 训练

| 项目 | 详情 |
|------|------|
| SLURM Job | 31756 |
| 节点 | gnho034 (8x GPU) |
| 配置 | pi05_libero_no10, FSDP 8 devices, batch_size=256 |
| 总步数 | 30,000 |
| 当前进度 | ~15,500/30,000 (52%), ~4.3s/step |
| 预计剩余 | ~17 小时 |
| Checkpoint | 已保存 step 5000, 10000, 14000（每 5000 步保存） |
| Norm stats | 964 batches, 约 25 分钟完成 |
| 数据验证 | `Filtering to 1515 episodes: 247029/273465 frames` ✅ |

### 4.2 评估

| 项目 | 详情 |
|------|------|
| SLURM Job | 31823 |
| 节点 | gnho009 (1x GPU) |
| 评估内容 | 3 个 checkpoint × 4 个测试 task × 50 trials = 600 episodes |
| 当前进度 | Step 5000, libero_10 suite_task 4 评估中 |

### 4.3 历史问题与修复

| 问题 | 原因 | 修复 |
|------|------|------|
| Job 31751 IndexError | LeRobot `episodes=` 与 `delta_timestamps` 不兼容 | 改用 Subset 过滤 |
| Job 31751 FileExistsError | Checkpoint 目录已存在 | 改用新 exp-name `split_experiment` |
| Job 31821 TypeError | Python 3.8 不支持 `tuple[int, ...]` 语法 | 改用 `typing.Optional[typing.Tuple[int, ...]]` |
| Job 31822 评估错误 task | `--args.task-ids 0` 对应的是 suite 内部编号，非数据集 task_index | 查明正确映射并修正为 4/8/9/6 |

## 5. 对比基线

上一次全量训练（40 task 全部参与）的评估结果（每个 suite 所有 10 个 task）：

| Suite | Success Rate |
|-------|:-----------:|
| libero_spatial | 97.8% |
| libero_object | 97.4% |
| libero_goal | 96.6% |
| libero_10 | 93.0% |

本次实验的目标是观察：排除 4 个测试 task 后，模型在这些**未见过的 task** 上的表现，以评估泛化能力。

## 6. 文件清单

| 文件 | 说明 |
|------|------|
| `src/openpi/training/libero_split.py` | 新增：train/test episode 索引 |
| `src/openpi/training/config.py` | 修改：新增 `episodes` 字段 + `pi05_libero_no10` 配置 |
| `src/openpi/training/data_loader.py` | 修改：Subset 过滤方式 |
| `examples/libero/main.py` | 修改：新增 `task_ids` 参数 |
| `train_pi05_libero_no10.sh` | 新增：训练 SLURM 脚本 |
| `eval_pi05_libero_split.sh` | 新增：评估 SLURM 脚本 |
