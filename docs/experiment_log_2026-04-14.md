# 实验日志 — 2026-04-14

> 记录截止时间：2026-04-14 19:40 CST  
> 项目路径：`/storage/yukaichengLab/lishiwen/jiayusun/openpi`

---

## 一、研究背景与目标

**核心问题**：pi0.5 在 LIBERO 上分别微调了 4 个套件（libero_10 / goal / object / spatial），能否通过模型融合得到一个同时在 4 个套件上表现良好的统一模型？

**融合方法**：WUDI（per-layer 2D SVD de-centering + Adam 优化），实现文件为 `scripts/wudi_merge.py`。

---

## 二、Checkpoint 全景图

### 2.1 Base 模型

| Checkpoint | 说明 | 关键 step |
|---|---|---|
| `checkpoints/pi05_libero/my_experiment/` | π₀.₅ 全量 LIBERO 微调（40 tasks, 1693 eps）<br>8×GPU FSDP, batch=256, 35.5h<br>Job 31657 (2026-03-13 → 03-15) | 5k / 10k / 15k / 20k / 25k / **29999** |
| `checkpoints/pi05_libero_RETRAIN_base/pi05_libero_RETRAIN_base/` | 重训 base（细节见 TRAINING_INFO.md） | 5k / **29999** |

### 2.2 per-suite 微调（从 pi05_libero 29999 出发）

**用途**：旧版 WUDI / GD 融合的输入，来自 pi05_libero 全量微调后继续 per-suite 专精。

| Checkpoint | 说明 | 关键 step |
|---|---|---|
| `checkpoints/pi05_libero_10/my_experiment/` | libero_10 suite | 5k / **29999** |
| `checkpoints/pi05_libero_spatial/my_experiment/` | libero_spatial suite | 5k / **29999** |
| `checkpoints/pi05_libero_goal/my_experiment/` | libero_goal suite | 5k / **29999** |
| `checkpoints/pi05_libero_object/my_experiment/` | libero_object suite | 5k / **29999** |

### 2.3 from10k 系列（从 pi05_libero 10000 出发）

**用途**：当前 WUDI 融合（MLLM 路线）的主要输入。从 10k base 出发，各 suite 继续微调到 20k step。

| Checkpoint | 说明 | 关键 step |
|---|---|---|
| `checkpoints/from10k/pi05_libero_10_from_pi05libero_10k/ft_from_pi05libero_10k/` | libero_10，从 base@10k 继续 | 5k/10k/15k/**20k**/25k/29999 |
| `checkpoints/from10k/pi05_libero_goal_from_pi05libero_10k/ft_from_pi05libero_10k/` | libero_goal | 同上 |
| `checkpoints/from10k/pi05_libero_object_from_pi05libero_10k/ft_from_pi05libero_10k/` | libero_object | 同上 |
| `checkpoints/from10k/pi05_libero_spatial_from_pi05libero_10k/ft_from_pi05libero_10k/` | libero_spatial | 同上 |

---

## 三、融合实验

### 3.1 早期：GD 融合（scripts/arithmetic.py）

**方法**：梯度下降直接优化融合系数（非 WUDI）。

| 实验名 | Checkpoint | 说明 |
|---|---|---|
| 4task merge avg | `pi05_libero_4task_merge` | 简单平均 |
| 4task merge GD | `pi05_libero_4task_merge_gd` | GD 优化 |
| 4task merge GD balanced | `pi05_libero_4task_merge_gd_balanced` | balanced dataset |
| 4task merge GD frozen | `pi05_libero_4task_merge_gd_frozen` | 冻结 action expert |
| 4task merge GD frozen large | `pi05_libero_4task_merge_gd_frozen_large` | 最大规模 856 iter |

均从 `pi05_libero_{spatial,object,goal,10}/my_experiment/29999` 融合（base=pi05_libero/29999）。

从 `pi05_libero_4task_merge_gd` 重训：`pi05_libero_from_merged_4task_gd/retrain_from_merged_30k/`（仅有 wandb_id，无 checkpoint 保留）。

### 3.2 当前：WUDI 融合（scripts/wudi_merge.py）

**算法核心**（scope=llm_only）：
- SigLIP vision encoder：保持 base 不动
- Gemma attn + FFN（q/kv/av/gate/linear，expert0+1，共 10 组参数）：WUDI 优化
- 其余（norm/embedding/action_head/time_mlp）：简单平均 task vector

| 实验名 | Base | FT 输入 | iter | 输出 Checkpoint | 状态 |
|---|---|---|---|---|---|
| **2task-MLLM 5k** | pi05_libero/10000 | libero_10@20k + libero_spatial@20k | 5000 | `wudi_mllm/2task_spatial10_iter5k` | ✅ 完成 |
| 2task-MLLM (early) | pi05_libero/10000 | libero_10 + libero_spatial | — | `wudi_mllm/2task_spatial10_iter1k` | ✅ 完成（early stop） |
| **4task-MLLM 5k** | pi05_libero/10000 | libero_{10,goal,object,spatial}@20k | 5000 | `wudi_mllm/4task_from10k_iter5k` | 🔄 **运行中** (Job 33750, gnho031) |
| **4task-MLLM 10k** | pi05_libero/10000 | libero_{10,goal,object,spatial}@20k | 10000 | `wudi_mllm/4task_from10k_iter10k` | 🔄 **运行中** (Job 33751, gnho034) |

---

## 四、Fine-tuning 实验（从融合 checkpoint 出发）

| 实验名 | 初始化 | 训练数据 | Config | 输出 Checkpoint | 状态 |
|---|---|---|---|---|---|
| ft from wudi_mllm_5k (libero_10 + spatial) | `wudi_mllm/2task_spatial10_iter5k` | libero_10 + libero_spatial | `pi05_libero_10_spatial_from_wudi_mllm_5k` | `pi05_libero_10_spatial_from_wudi_mllm_5k/ft_from_wudi_mllm_5k/` | 🔄 **运行中** (Job 33754, gnho018, step ~5) |

**训练配置**（Job 33754）：
- batch_size=32，2×GPU FSDP，gnho018
- lr: CosineDecay warmup 1k → peak 5e-5 → decay 30k → 5e-6
- AdamW clip_norm=1.0，EMA=0.999
- 总步数：30,000（≈8.5h @ 1s/it）

---

## 五、评测实验

### 5.1 评测目标

**被评测 checkpoint**：`checkpoints/wudi_mllm/2task_spatial10_iter5k`  
**评测套件**：libero_spatial / libero_object / libero_goal / libero_10（各 10 task × 50 trials = 500 trials）  
**评测脚本**：`eval_wudi_mllm_5k_all_suites.sbatch`

### 5.2 已完成结果

| 运行 | 套件 | 成功率 | 备注 |
|---|---|---|---|
| Job 33734 | libero_spatial | **90.2%** (451/500) | 完成 ✅ |
| Job 33734 | libero_object | — | 在 episode 2/50 时被 kill |

### 5.3 当前运行

| Job | 套件进度 | 节点 | 状态 |
|---|---|---|---|
| 33756 | libero_spatial 进行中（~14%） | gnho018 | 🔄 **运行中** |

待完成：libero_spatial → libero_object → libero_goal → libero_10（顺序执行）

---

## 六、当前 SLURM 任务一览

| Job ID | 名称 | 节点 | GPU | 说明 |
|---|---|---|---|---|
| 33754 | train_10_spatial_wudi_5k | gnho018 | 2 | FT from wudi_mllm_5k，libero_10+spatial |
| 33755 | jiayu_hold_018 | gnho018 | 0 | 占位保节点 |
| 33756 | eval_wudi_mllm_5k | gnho018 | 2 | 评测 4 suites |
| 33750 | jiayu_wudi_4task_5k | gnho031 | 4 | WUDI 4task 融合 5k iter |
| 33751 | jiayu_wudi_4task_10k | gnho034 | 4 | WUDI 4task 融合 10k iter |

---

## 七、Hold 节点占位任务

| 节点 | Job ID | 脚本 |
|---|---|---|
| gnho009 | 33727 | `inf_task_009.sh` |
| gnho018 | 33755 | `inf_task_018.sh` |
| gnho031 | 33729 | `inf_task_031.sh` |
| gnho034 | — | 未提交（034 当前由 merge job 33751 占用） |

占位脚本均运行 `sleep_forever.py`，gpu:0，mem=1G。

---

## 八、实验路线图

```
pi05_libero (base, 30k)
│
├── @10k ──────────────────────────────────────────────────────┐
│    │                                                          │
│    ├─ FT libero_10  @20k ──┐                                 │
│    ├─ FT libero_spatial @20k┤                                 │
│    ├─ FT libero_goal @20k  ├─ WUDI 2task 5k ─► [DONE]       │
│    └─ FT libero_object @20k┤      │                          │
│                             │      └─► FT libero_10+spatial  │
│                             │            [RUNNING 33754]     │
│                             │                                 │
│                             ├─ WUDI 4task 5k  [RUNNING 33750]│
│                             └─ WUDI 4task 10k [RUNNING 33751]│
│                                                              ─┘
└── @29999
     ├─ FT libero_10 @29999 ─┐
     ├─ FT libero_spatial    ├─ GD/WUDI merge (早期实验)
     ├─ FT libero_goal       ┘
     └─ FT libero_object
```

---

## 九、后续计划

1. **等待评测完成**（Job 33756）：获得 `wudi_mllm/2task_spatial10_iter5k` 在全 4 套件的成功率
2. **等待 WUDI 4task 融合完成**（Job 33750/33751）：
   - 对 `4task_from10k_iter5k` 和 `4task_from10k_iter10k` 分别做评测
   - 预期用 `eval_wudi_mllm_5k_all_suites.sbatch` 修改 checkpoint 路径后提交
3. **等待 FT 完成**（Job 33754）：30k step 后评测 `pi05_libero_10_spatial_from_wudi_mllm_5k`
4. **对比分析**：
   - wudi_2task_5k 直接评测 vs wudi_2task_5k + FT 的对比
   - wudi_4task 5k vs 10k iter 的对比
   - 4task vs 2task 融合的泛化能力对比
