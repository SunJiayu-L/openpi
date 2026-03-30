# LIBERO 评测 Core Dump 事故记录（2026-03-30）

## 1. 现象

- 评测作业在第一个 task 的第一个 episode 就退出。
- 日志出现：
  - `INFO:openpi.serving.websocket_policy_server:Connection from (...) closed`
  - `Aborted (core dumped) python examples/libero/main.py ...`
- 典型作业：
  - `32017`（`eval_ft`）
  - `32018`（`eval_ft`）
  - `32780`（`eval_l10_3t`）

## 2. 容易误判的点

- `32780` 在 Slurm 里显示 `COMPLETED (0:0)`，但并不代表评测成功。
- 原因是脚本里用了：
  - `python examples/libero/main.py ... || true`
- 这会吞掉评测进程失败，导致作业“看起来成功”。

## 3. 根因判断

高概率是 2 卡评测时，策略推理进程（JAX）与仿真渲染进程（MuJoCo EGL）没有稳定分卡，导致冲突或不稳定崩溃。

证据：
- 多次崩溃都发生在 `episode 1` 刚开始。
- `serve_policy.py` 先成功启动并建立 websocket，随后连接被动关闭，说明更像评测端（仿真端）崩溃。
- 多个历史评测作业出现同类模式，不是单次偶发。

## 4. 修复方案

### 4.1 固定 GPU 分工（关键）

- 规则：`GPU0` 专门给仿真，`GPU1` 专门给策略。
- 在 `eval_wudi_libero10_3task.sbatch` 中：
  - 策略服务：
    - `CUDA_VISIBLE_DEVICES=1 ... uv run scripts/serve_policy.py`
  - 仿真评测：
    - `CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python examples/libero/main.py ...`

### 4.2 让失败真实暴露

- 去掉 `|| true`，避免把失败当成功。
- 增加：
  - `export PYTHONFAULTHANDLER=1`
  - `python -X faulthandler ...`

### 4.3 相关脚本同步修正

- 已修改：
  - `/storage/yukaichengLab/lishiwen/jiayusun/openpi/eval_wudi_libero10_3task.sbatch`
  - `/storage/yukaichengLab/lishiwen/jiayusun/openpi/wudi_eval_suite.sbatch`

## 5. 本次重提记录

- 旧任务（旧脚本）：`32794`
- 新任务（修复后脚本）：`32800`
- 资源约束：
  - `--gres=gpu:2`
  - `--nodes=1 --ntasks=1`
  - `--nodelist=gnho031,gnho034`
  - `--exclude=gnho009`

## 6. 后续建议（评测模板）

对于 2 卡 LIBERO 评测，固定使用以下模式：

```bash
# policy on GPU1
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/serve_policy.py ...

# simulator on GPU0
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python -X faulthandler examples/libero/main.py ...
```

并且不要在关键评测命令后追加 `|| true`。
