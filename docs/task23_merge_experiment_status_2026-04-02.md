# Task23 融合实验状态与规划（2026-04-02）

## 1. 目标

当前总目标有两个层次：

1. 诊断层：
- 解释为什么 merged 在 `task23 (bbq sauce)` 上不如 base。
- 判断主要问题来自 `VL` 还是 `action`。

2. 方法层：
- 验证“先训 B/C，再融合，再训 A”是否能稳定优于 `A from base`。

其中任务定义：
- TaskA（难）：`pick up the bbq sauce and place it in the basket`
- TaskB（易）：`pick up the alphabet soup and place it in the basket`
- TaskC（中）：`pick up the cream cheese and place it in the basket`

---

## 2. 统一设置（你关心的这套）

- 节点：仅 `gnho031`
- 训练：`batch_size=8`
- 评测：`task23(task_id=3)`
- 每次评测：`50 trials`
- seeds：`0/1/2`
- 指标：`success_rate`，最终汇总 `mean ± std`

结论：
- 这套设置 **部分已经做过**（4组方向诊断）。
- 但“B/C 迁移到 A”的完整方案 **还没开始**。

---

## 3. 已完成实验（明确）

### 3.1 两组 30 步对比（task23）
- base30 vs merged30
- 结果：两者都 `0.05`
- 结论：30 步下 merged 没有优势。

### 3.2 两组 50 步对比（task23）
- base50 vs merged50
- 结果：
  - base50：`0.25 (5/20)`
  - merged50：`0.05 (1/20)`
- 结论：在该设置下，base 明显优于 merged。

### 3.3 方向诊断 4 组（阶段A）已提交并修复流程
4 组定义：
1. `base_full`
2. `merged_full`
3. `mvl_bact` = merged VL + base action
4. `bvl_mact` = base VL + merged action

- seeds：`0/1/2`
- 每组训练 50 步，评测 task23 50 trials
- 说明：
  - 训练作业中有若干显示 `FAILED 0:15`，但多数 checkpoint( step50 )已成功落盘。
  - 已改为统一补评测流程，避免依赖链断裂。

关键提交记录：
- 提交映射：`trainlogs/submit_task23_ablation_4x3_50.csv`
- 统一补评测作业：`33102`（在 031 串行跑 12 组并重试）
- 结果汇总文件：`trainlogs/task23_ablation_eval_summary_recover_33102.csv`

当前状态（写文档时）：
- `33102` 仍在运行中，最终均值方差尚未产出。

---

## 4. 尚未开始的实验（你提的新主线）

即“B/C 迁移到 A”的方法层验证：

1. `A from base`（基线）
2. `B only -> A`
3. `C only -> A`
4. `B/C VL merge + base(action) -> A`（主方案）
5. （可选）`B/C full merge -> A`

这些还没有系统跑完（3 seeds + 50 trials + mean±std）。

---

## 5. 下一步执行顺序（建议）

### Step 1：先收口阶段A
- 等 `33102` 跑完。
- 输出 4 组 `mean ± std`。
- 给出一句诊断：
  - `action 主因` / `VL 主因` / `两者都导致`。

### Step 2：启动 B/C->A 主线
按统一设置跑 4 组核心对照（每组 3 seeds）：
- `A from base`
- `B only -> A`
- `C only -> A`
- `B/C VL merge + base(action) -> A`

### Step 3：形成最终结论
产出总表：
- `config, data_size, steps, seed, success_rate`
并输出：
- 每组 `mean ± std`
- 最终一句结论：
  - “action 导致” / “VL 导致” / “数据导致”

---

## 6. 你现在最关心的问题的直接回答

问题：
- “统一设置那套（031、bs8、task23、50trials、3 seeds、mean±std）是不是做过了？”

回答：
- **做过一部分**：4组方向诊断（阶段A）在跑，结果还在收集。
- **还没做完整**：你新提的 `B/C -> A` 路线对照实验还没正式开始。

