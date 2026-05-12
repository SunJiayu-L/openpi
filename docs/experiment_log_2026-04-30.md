# 实验日志 — 2026-04-30

> 创建：2026-04-30 CST  
> 项目路径：`/storage/yukaichengLab/lishiwen/jiayusun/openpi`

---

## 一、本次实验背景

接续 `experiment_log_2026-04-29.md`，本次主要工作：

1. **base checkpoint 全量评测**（20k / 25k / 30k）：评测 `pi05_libero/my_experiment/` 各阶段 checkpoint 的全 4 suite 成绩，寻找更好的训练基底。
2. **`ft_from_wudi_4task_1k/9000` 重测**：验证 task8=74% 的可重复性（结论：存在较大随机方差）。
3. **task8 × 200 次精确测评**：尝试用 200 次减小方差（发现 LIBERO eval 代码 bug，已修复）。
4. **task8 能否 SFT 到 90% 的可行性分析**：结论为否。

---

## 二、base checkpoint 全量评测

### 2.1 评测配置

- Checkpoint 路径：`checkpoints/pi05_libero/my_experiment/{20000, 25000, 29999}`
- 注：`29999` 即训练末端（30k 步），`20000` 已删除（仅有旧 eval 记录）
- 全 4 suite，每 task 50 episodes

### 2.2 评测结果

| Checkpoint | spatial | object | goal | libero_10 | **avg(4)** | task8 ★ | eval Job |
|---|---|---|---|---|---|---|---|
| base_10k（历史） | ~96% | ~98% | ~97% | ~91% | ~95.5% | — | — |
| base_15k（历史） | — | — | — | — | — | — | — |
| **base_20k** | 97.8% | 98.6% | 96.8% | 93.4% | **96.7%** | 33/50 = **66%** | 34678 |
| **base_25k** | 98.4% | 99.0% | 98.4% | 94.0% | **97.5%** ⭐ | 35/50 = **70%** | 34702 |
| **base_30k** | 99.0% | 98.8% | 97.4% | 93.2% | **97.1%** | 32/50 = **64%** | 34719 |

### 2.3 关键发现

1. **base_25k 是新历史最优基底**：avg=**97.5%**，超过所有之前记录（之前最佳 96.6% 来自 l10_mix50@2k FT）
2. **25k 是甜点，30k 开始退化**：
   - libero_10：20k=93.4% → 25k=94.0% → 30k=93.2%（25k 峰值）
   - libero_goal：25k=98.4% → 30k=97.4%（-1.0pp）
   - libero_spatial：持续提升（97.8→98.4→99.0%）
3. **task8 随 base 训练步数无明显改善**：20k=66%，25k=70%，30k=64%，天花板约 70%

### 2.4 可用 checkpoint 步数

| 步数 | 路径 | 存在 |
|---|---|---|
| 5000 | `my_experiment/5000` | ✅ |
| 10000 | `my_experiment/10000` | ✅ |
| 15000 | `my_experiment/15000` | ✅ |
| 20000 | `my_experiment/20000` | ✅ |
| 25000 | `my_experiment/25000` | ✅ |
| 29999 (30k) | `my_experiment/29999` | ✅ |

---

## 三、`ft_from_wudi_4task_1k/9000` 重测（Job 34730）

### 3.1 评测结果

| Suite | 成功率 | 备注 |
|---|---|---|
| libero_spatial | 97.8% | 489/500 |
| libero_object | 98.8% | 494/500 |
| libero_goal | 95.8% | 479/500 |
| libero_10 | 90.8% | 434/478（task9 被截断，仅 28 eps）|
| **avg(4)** | ~96.0%（估算）| libero_10 不完整 |

**task8（★）= 30/50 = 60%**

### 3.2 与原始记录对比

| 评测 | Job | task8 | libero_10 | avg(4) |
|---|---|---|---|---|
| 原始（4月16日） | 33834 | **37/50 = 74%** | 93.0% | 96.5% |
| 重测（4月30日） | 34730 | **30/50 = 60%** | 90.8%（不完整） | ~96.0% |

**结论**：50 次 episode 评测的随机方差较大。以 p≈0.67 估计，95% CI ≈ ±13pp，74% 和 60% 均在置信区间内。**task8 真实成功率估计约 67%**。

### 3.3 `pi05_libero_10_from_pi05libero_10k/20000` 历史记录

该 checkpoint（libero_10 专精 FT，从 10k base 训 20k 步）已被删除，仅有 eval 记录（Job 33920）：

| Suite | 成功率 | 说明 |
|---|---|---|
| libero_10 | **92.6%** | 专精良好 |
| libero_spatial | 17.4% | **灾难性遗忘** |
| libero_object | 34.2% | **灾难性遗忘** |
| libero_goal | 10.0% | **几乎全部遗忘** |
| avg(4) | 38.5% | 不可用 |

task8 = 31/50 = **62%**。单 suite 专精代价极大，不可取。

---

## 四、task8 × 200 次评测（未完成 + Bug 修复）

### 4.1 经过

- Job 34742/34743：使用 `--args.task-ids 8 --args.num-trials-per-task 200`
- 跑到第 50 次时崩溃：`IndexError: index 50 is out of bounds for axis 0 with size 50`
- **根因**：LIBERO 每个 task 只有 50 个预设初始状态，`initial_states[episode_idx]` 越界

### 4.2 Bug 修复

```python
# examples/libero/main.py line 102
# 修复前
obs = env.set_init_state(initial_states[episode_idx])
# 修复后
obs = env.set_init_state(initial_states[episode_idx % len(initial_states)])
```

### 4.3 测试结论

200 次测试本质上是 50 个初始状态循环 4 遍，不增加新初始状态多样性。鉴于 task8 的核心问题不是统计方差而是能力上限，200 次测试意义有限，不再追测。

---

## 五、task8 能否通过 SFT 达到 90% —— 可行性分析

### 5.1 现有最优成绩汇总（task8: "put both moka pots on the stove"）

| Checkpoint | task8 成功率 | 来源 |
|---|---|---|
| `ft_from_wudi_4task_1k/9000` | ~67%（74%/60% 两次均值） | 全 4 suite 联合 FT 9k 步 |
| base_25k | 70% | 纯预训练 25k 步 |
| base_20k | 66% | 纯预训练 20k 步 |
| t89 @ step300 | 62% | task8+9 专精 FT |
| ft_l10_t8_from_avg_x07/300 | **46%** | task8 单独 FT（退化）|

### 5.2 为什么 SFT 无法达到 90%

1. **数据是硬上限**：只有 **29 个 demo**，LIBERO 也只有 **50 种初始状态**，覆盖不足
2. **任务结构性难度**：需两次连续抓取+放置，若每步成功率 90%，串行后上限仅 81%；达到 90% 整体成功率要求每步 94.9%
3. **过拟合风险极高**：500 步 batch=32 → 每个 demo 被重复约 550 次，必然过拟合
4. **历史证明**：单 task 专精 FT 反而退化（46% < 基线 54%）
5. **当前天花板约 67-70%**：任何 SFT 方法都在这个区间波动，无法突破

### 5.3 若要达到 90% 需要的条件

| 方案 | 可行性 | 说明 |
|---|---|---|
| 补充更多 demo（100+ eps） | ✅ 必要条件 | 当前 29 eps 不够覆盖初始状态空间 |
| RL 在线训练 | ✅ 理论可行 | 不依赖 demo 数量，需在线 MuJoCo 环境 |
| SFT（当前数据） | ❌ | 天花板约 70-75%，且退化风险高 |

**结论：task8 的 90% 目标在当前数据和方法下不可达。**

---

## 六、SLURM 任务汇总（2026-04-30）

| Job | 类型 | 说明 | 节点 | 状态 |
|---|---|---|---|---|
| 34671 | eval | `ft_from_wudi_4task_1k/9000` 全 4 suite（失败） | gnho009 | ❌ GCS 网络错误 |
| 34674 | eval | base_20k 全 4 suite（失败） | gnho031 | ❌ GCS 网络错误 |
| 34675 | eval | base_25k 全 4 suite（失败） | gnho031 | ❌ GCS 网络错误 |
| 34676 | eval | base_30k 全 4 suite（失败） | gnho031 | ❌ GCS 网络错误 |
| 34678 | eval | base_20k 全 4 suite（重提） | gnho031 | ✅ spatial=97.8/object=98.6/goal=96.8/l10=93.4，avg=96.7% |
| 34702 | eval | base_25k 全 4 suite（重提） | gnho031 | ✅ spatial=98.4/object=99.0/goal=98.4/l10=94.0，avg=**97.5%** ⭐ |
| 34719 | eval | base_30k 全 4 suite（重提） | gnho031 | ✅ spatial=99.0/object=98.8/goal=97.4/l10=93.2，avg=97.1% |
| 34730 | eval | `ft_from_wudi_4task_1k/9000` 全 4 suite（重提） | gnho031 | ⚠️ 部分完成（libero_10 task9 截断），task8=30/50=60% |
| 34742/34743 | eval | task8 × 200 次 | gnho009 | ❌ IndexError（50 初始状态越界，已修复代码） |
| 34745 | eval | task8 × 200 次（修复后） | gnho009 | ❌ EOF（gnho009 环境问题，已取消） |

---

## 七、关键结论

### 7.1 最优基底确认：base_25k

`pi05_libero/my_experiment/25000` 以 **avg(4)=97.5%** 成为历史最优基底，适合作为后续所有 FT 的起点。

### 7.2 全局最优 checkpoint 汇总（更新）

| 方案 | avg(4) | task8 | 说明 |
|---|---|---|---|
| **base_25k** | **97.5%** ⭐ | 70% | 历史最优基底（无 FT） |
| base_30k | 97.1% | 64% | 过训练轻微退化 |
| base_20k | 96.7% | 66% | — |
| l10_mix50@2k | 96.6% | 48% | 之前最佳 FT |
| t89@step300 | 96.5% | 62% | — |
| ft_from_wudi_4task_1k/9000 | 96.5% | ~67% | task8 历史最优 |

### 7.3 task8 结论

- **真实成功率约 67%**（50-trial 单次测评方差约 ±13pp）
- **SFT 无法达到 90%**：数据（29 eps）、任务结构（双物体串行）、过拟合三重限制
- 若要突破 70%：需补充 demo 或走 RL 路线

---

## 八、待办事项

1. **从 base_25k 出发探索**：作为迄今最优基底，考虑用 l10_mix50 配方（500 eps 混合）从 25k 继续 FT
2. **task8 demo 补充**：若要攻克 task8，需录制 100+ 个新 demo
3. **代码修复已合并**：`examples/libero/main.py` 的 `episode_idx % len(initial_states)` 修复已生效
