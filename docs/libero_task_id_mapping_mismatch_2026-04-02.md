# LIBERO `task_id` 与 `meta/tasks.jsonl task_index` 不一致说明（2026-04-02）

## 1. 问题结论

在当前环境中，以下两个编号体系**不是同一个东西**：

- `meta/tasks.jsonl` 里的 `task_index`
- `examples/libero/main.py --args.task-ids` 使用的评测 `task_id`（suite 内部编号）

这会导致“按 `task_index` 直接填到 `--args.task-ids`”时评测错任务。

## 2. 本次实际影响

本次作业里出现了这个偏差：

- 用户期望评测：
  - `put both the cream cheese box and the butter in the basket`
  - `put both the alphabet soup and the tomato sauce in the basket`
- 但旧评测脚本传了 `task_ids=7 5`，实际跑成：
  - `put both the alphabet soup and the cream cheese box in the basket`
  - `pick up the book and place it in the back compartment of the caddy`

相关作业：`33026`

## 3. 证据

### 3.1 `tasks.jsonl`（meta 索引）

文件：`/storage/yukaichengLab/lishiwen/jiayusun/libero/meta/tasks.jsonl`

- `task_index=5`: `put both the alphabet soup and the tomato sauce in the basket`
- `task_index=7`: `put both the cream cheese box and the butter in the basket`

### 3.2 `libero_10` 评测内部顺序（suite 内部 `task_id`）

通过以下方式读取：

```bash
PYTHONPATH=/storage/yukaichengLab/lishiwen/jiayusun/openpi/third_party/libero \
examples/libero/.venv/bin/python - <<'PY'
from libero.libero import benchmark
suite=benchmark.get_benchmark_dict()["libero_10"]()
for i in range(suite.n_tasks):
    print(i, suite.get_task(i).language)
PY
```

得到：

- `task_id=0`: `put both the alphabet soup and the tomato sauce in the basket`
- `task_id=1`: `put both the cream cheese box and the butter in the basket`
- `task_id=7`: `put both the alphabet soup and the cream cheese box in the basket`
- `task_id=5`: `pick up the book and place it in the back compartment of the caddy`



因此：

- 目标任务 tomato sauce 应使用 `task_id=0`
- 目标任务 cream cheese+butter 应使用 `task_id=1`

## 4. 当前 `libero_10` 对齐表

| libero_10 评测 `task_id` | meta `task_index` | task |
|---|---:|---|
| 0 | 5 | put both the alphabet soup and the tomato sauce in the basket |
| 1 | 7 | put both the cream cheese box and the butter in the basket |
| 2 | 3 | turn on the stove and put the moka pot on it |
| 3 | 8 | put the black bowl in the bottom drawer of the cabinet and close it |
| 4 | 0 | put the white mug on the left plate and put the yellow and white mug on the right plate |
| 5 | 9 | pick up the book and place it in the back compartment of the caddy |
| 6 | 1 | put the white mug on the plate and put the chocolate pudding to the right of the plate |
| 7 | 4 | put both the alphabet soup and the cream cheese box in the basket |
| 8 | 6 | put both moka pots on the stove |
| 9 | 2 | put the yellow and white mug in the microwave and close it |

## 5. 规避规范（必须执行）

1. 不要再把 `meta/tasks.jsonl task_index` 直接当作 `--args.task-ids`。
2. 评测前先按“任务文本”反查 `libero_10` 的 `task_id`。
3. 在评测日志开头打印 `task_id` 和任务文本，提交前人工二次确认。

## 6. 推荐做法（按任务文本自动求 `task_id`）

```bash
PYTHONPATH=/storage/yukaichengLab/lishiwen/jiayusun/openpi/third_party/libero \
examples/libero/.venv/bin/python - <<'PY'
from libero.libero import benchmark

TARGET = "put both the alphabet soup and the tomato sauce in the basket"
suite = benchmark.get_benchmark_dict()["libero_10"]()

matches = [i for i in range(suite.n_tasks) if suite.get_task(i).language == TARGET]
print(matches)
PY
```

输出 `0` 即可安全用于：

```bash
--args.task-ids 0
```

## 7. 本次修复动作

已新增并提交修正评测脚本（使用 `task_ids=1 0`）：

- 脚本：`/storage/yukaichengLab/lishiwen/jiayusun/openpi/eval_task5_ckpt50_100_on_ids0_1.sbatch`
- 修正评测作业：`33031`
