# 评测计划：pi05_libero10_task{0,6,9}_retrain

**日期**：2026-03-28
**状态**：待训练完成后执行

---

## 1. 待评测模型

三个单任务 fine-tune 模型，均从 `pi05_libero_RETRAIN_base/10000` 热启动：

| 训练 Config | 训练数据 (dataset task_index) | Checkpoint 路径（待填写） |
|---|---|---|
| `pi05_libero10_task0_retrain` | task_index = 0 | `.../pi05_libero10_task0_retrain/<exp>/29999` |
| `pi05_libero10_task6_retrain` | task_index = 6 | `.../pi05_libero10_task6_retrain/<exp>/29999` |
| `pi05_libero10_task9_retrain` | task_index = 9 | `.../pi05_libero10_task9_retrain/<exp>/29999` |

---

## 2. task_index → benchmark suite_task_id 映射（libero_10）

> 以下映射来自 `libero_split_experiment.md` §2.5

| dataset task_index | benchmark suite | benchmark suite_task_id |
|:-:|:-:|:-:|
| 0 | libero_10 | **4** |
| 6 | libero_10 | **8** |
| 9 | libero_10 | **5** |

评测脚本中的 `--task-ids` 参数均使用 **benchmark suite_task_id**（不是 dataset task_index）。

---

## 3. ID Eval（域内评测）

### 3.1 定义

每个模型在**自己训练的 task** 上评测，验证模型是否成功学会该任务。

### 3.2 评测配置

| 项目 | 设置 |
|---|---|
| Suite | `libero_10` |
| Episodes 数 | 20 episodes per task |
| Seeds | 1（固定） |
| Policy server config | `pi05_libero_10`（RETAIN_code 中） |

### 3.3 评测矩阵

| 模型 | 评测 Suite | Suite Task ID | Episodes |
|---|---|:-:|:-:|
| task0_retrain | libero_10 | 4 | 20 |
| task6_retrain | libero_10 | 8 | 20 |
| task9_retrain | libero_10 | 5 | 20 |

### 3.4 伪命令（参考 `examples/libero/main.py`）

```bash
# 启动 policy server（RETAIN_code 中）
uv run scripts/serve_policy.py \
    --env LIBERO \
    --checkpoint_dir <task0_retrain_ckpt> \
    --config pi05_libero_10

# 评测 task0 → suite_task_id=4
python examples/libero/main.py \
    --suite libero_10 \
    --task-ids 4 \
    --num-episodes 20 \
    --host <server_host> --port <port>
```

---

## 4. OOD Eval（域外评测）

### 4.1 定义

每个模型在**自己未训练的 task** 上评测，验证模型泛化能力，同时检验单任务 fine-tune 后是否对其他任务有遗忘。

### 4.2 评测配置

| 项目 | 设置 |
|---|---|
| Suite | `libero_10` |
| Episodes 数 | 5 seeds × 10 episodes/seed = **50 episodes per task** |
| Seeds | 5（每个 seed 固定不同随机初始化） |
| Policy server config | `pi05_libero_10` |

### 4.3 OOD task 选择

每个模型评测**另外两个 task**（libero_10 内跨 task 泛化）：

| 模型 | 训练 task (suite_task_id) | OOD 评测 task (suite_task_id) |
|---|:-:|:-:|
| task0_retrain | 4 | **8, 5** |
| task6_retrain | 8 | **4, 5** |
| task9_retrain | 5 | **4, 8** |

> 如需扩展至其他 suite（goal/object/spatial），可另行补充。

### 4.4 伪命令（参考 `examples/libero/main.py`）

```bash
# 评测 task0_retrain 在 OOD task (suite_task_id=8,5) 上，各 5 seeds × 10 episodes
for SEED in 0 1 2 3 4; do
    python examples/libero/main.py \
        --suite libero_10 \
        --task-ids 8 5 \
        --num-episodes 10 \
        --seed $SEED \
        --host <server_host> --port <port>
done
```

---

## 5. 结果汇总表（待填写）

### ID Eval

| 模型 | Task | Suite Task ID | Success Rate (20 eps) |
|---|---|:-:|:-:|
| task0_retrain | libero_10 task0 | 4 | — |
| task6_retrain | libero_10 task6 | 8 | — |
| task9_retrain | libero_10 task9 | 5 | — |

### OOD Eval

| 模型 | OOD Task | Suite Task ID | Success Rate (50 eps, 5 seeds) |
|---|---|:-:|:-:|
| task0_retrain | libero_10 task6 | 8 | — |
| task0_retrain | libero_10 task9 | 5 | — |
| task6_retrain | libero_10 task0 | 4 | — |
| task6_retrain | libero_10 task9 | 5 | — |
| task9_retrain | libero_10 task0 | 4 | — |
| task9_retrain | libero_10 task6 | 8 | — |

---

## 6. 待确认事项（请在 review 时核对）

1. **Checkpoint 路径**：三个训练 job 完成后，确认 exp-name 对应的实际路径
2. **Policy server config**：RETAIN_code 中用 `pi05_libero_10` 还是 `pi05_libero`（取决于 norm stats 来源）
3. **OOD 范围扩展**：是否需要在 goal/object/spatial 的任务上也做跨 suite OOD 评测
4. **SLURM GPU 数量**：评测 server 是否用 1 GPU（参考之前 `32605` 的配置）
