# Single-Suite Fine-tuning: Code Changes Summary

## 目标

对 LIBERO 四个 suite (libero_10, libero_goal, libero_object, libero_spatial) 分别独立微调，使用 pi0_base 和 pi05_base 两个预训练模型，共 2×4=8 个训练任务。

---

## 修改文件

### 1. 新增: `src/openpi/training/libero_suite_episodes.py`

定义每个 suite 的 episode 索引列表，供 config 中 `DataConfig.episodes` 使用。

```python
# libero_10: task_index 0-9, 379 episodes
LIBERO_10_EPISODES = list(range(0, 379))

# libero_goal: task_index 10-19, 428 episodes
LIBERO_GOAL_EPISODES = list(range(379, 807))

# libero_object: task_index 20-29, 454 episodes
LIBERO_OBJECT_EPISODES = list(range(807, 1261))

# libero_spatial: task_index 30-39, 432 episodes
LIBERO_SPATIAL_EPISODES = list(range(1261, 1693))
```

**数据来源**: 从 `/storage/yukaichengLab/lishiwen/jiayusun/libero` 数据集 parquet 文件中提取，按 `task_index` 范围分组：
- task_index 0-9 → libero_10
- task_index 10-19 → libero_goal
- task_index 20-29 → libero_object
- task_index 30-39 → libero_spatial

### 2. 修改: `src/openpi/training/config.py`

#### 2.1 新增 import

```python
import openpi.training.libero_suite_episodes as _suite_eps
```

#### 2.2 新增 8 个 TrainConfig（插入在 pi05_libero_no10 和 pi0_aloha_pen_uncap 之间）

**pi0 × 4 suites:**

| Config Name | 模型 | 数据 | Episodes |
|-------------|------|------|:--------:|
| `pi0_libero_10` | Pi0Config() | libero_10 | 379 |
| `pi0_libero_goal` | Pi0Config() | libero_goal | 428 |
| `pi0_libero_object` | Pi0Config() | libero_object | 454 |
| `pi0_libero_spatial` | Pi0Config() | libero_spatial | 432 |

- weight_loader: `/storage/yukaichengLab/lishiwen/jiayusun/openpi_pt/pi0_base/params` (本地路径)
- extra_delta_transform: True
- batch_size: 32 (默认)
- num_train_steps: 30,000

**pi0.5 × 4 suites:**

| Config Name | 模型 | 数据 | Episodes |
|-------------|------|------|:--------:|
| `pi05_libero_10` | Pi0Config(pi05=True, action_horizon=10) | libero_10 | 379 |
| `pi05_libero_goal` | Pi0Config(pi05=True, action_horizon=10) | libero_goal | 428 |
| `pi05_libero_object` | Pi0Config(pi05=True, action_horizon=10) | libero_object | 454 |
| `pi05_libero_spatial` | Pi0Config(pi05=True, action_horizon=10) | libero_spatial | 432 |

- weight_loader: `gs://openpi-assets/checkpoints/pi05_base/params`
- extra_delta_transform: False
- batch_size: 256
- lr_schedule: CosineDecay(warmup=10k, peak=5e-5)
- optimizer: AdamW(clip_gradient_norm=1.0)
- ema_decay: 0.999
- num_train_steps: 30,000

#### 2.3 已有修改 (之前的改动)

`pi0_libero` config 的 weight_loader 从 GCS 路径改为本地路径：
```python
# 之前
weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params")
# 现在
weight_loader=weight_loaders.CheckpointWeightLoader("/storage/yukaichengLab/lishiwen/jiayusun/openpi_pt/pi0_base/params")
```

### 3. 新增: `train_suite.sh`

通用 SLURM 训练脚本，通过 `CONFIG` 环境变量指定训练配置：

```bash
sbatch -J pi0_l10 --nodelist=gnho009 --export=CONFIG=pi0_libero_10 train_suite.sh
```

- 8 GPU FSDP
- 先 compute_norm_stats，再 train
- exp-name 统一为 `my_experiment`
- HF_HUB_OFFLINE=1, WANDB_MODE=disabled

### 4. 新增: `train_pi0_libero.sh`

pi0 全量 40-task 训练脚本（已提交运行，Job 31859）。

### 5. 新增: `docs/training_manual.md`

训练手册，包含集群环境、数据集说明、训练/评测流程、常见问题等。

---

## 数据过滤机制

利用已有的 `data_loader.py` Subset 过滤逻辑（之前为 split 实验修复的 LeRobot episodes= bug）：

1. Config 中设置 `episodes=_suite_eps.LIBERO_10_EPISODES`
2. `data_loader.py` 加载完整数据集后，用 `torch.utils.data.Subset` + `np.isin` 过滤到指定 episodes
3. 只有该 suite 的 frames 参与训练

---

## 训练调度

| Job ID | Config | 节点 | 状态 |
|:------:|--------|:----:|:----:|
| 31860 | pi0_libero_10 | gnho009 | RUNNING |
| 31861 | pi0_libero_goal | gnho031 | RUNNING |
| 31862 | pi0_libero_object | — | PENDING |
| 31863 | pi0_libero_spatial | — | PENDING |
| 31864 | pi05_libero_10 | — | PENDING |
| 31865 | pi05_libero_goal | — | PENDING |
| 31866 | pi05_libero_object | — | PENDING |
| 31867 | pi05_libero_spatial | — | PENDING |

3 个节点 (gnho009/031/034) 并行，每 job ~35h，预计 3 轮 × 35h ≈ 4-5 天完成全部 8 个训练。

---

## Review 要点

1. **episode 列表正确性**: `LIBERO_10_EPISODES = list(range(0, 379))` 等是否覆盖了正确的 task_index 范围？
2. **pi0 vs pi05 超参差异**: pi0 用默认 lr/batch，pi05 用 CosineDecay + batch=256 — 与全量训练一致
3. **pi05 weight_loader 仍为 GCS 路径**: `gs://openpi-assets/checkpoints/pi05_base/params`，需确认节点能否访问（之前 pi05 全量训练是否成功从此路径加载？如果不行需改为本地路径）
4. **data_loader.py Subset 过滤**: 依赖之前的 bug fix，确保单 suite episodes 过滤后 prompt_from_task 仍能正确获取任务描述
