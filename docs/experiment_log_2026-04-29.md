# 实验日志 — 2026-04-29

> 创建：2026-04-29 CST  
> 项目路径：`/storage/yukaichengLab/lishiwen/jiayusun/openpi`

---

## 一、本次实验背景

本次日志覆盖 2026-04-29 的实验，接续 `experiment_log_2026-04-25.md` 的 l10_mix50 路线（avg=96.6% 历史最佳），主要包含四项工作：

1. **`ft_l10_t89` 系列完整评测**（Jobs 34569 / 34629）：对任务 8+9 专精 FT（从 l10_mix50@2k 出发）的 step300 和 step500 checkpoint 做全 4 suite 评测，得到关键收敛规律。
2. **简单加权平均融合**（Job 34645）：将 `t89_step300 × 0.7 + base_15k × 0.3` 做加权融合（llm_only scope，无 WUDI）。
3. **task 8 专项 FT**（Job 34647）：从融合 checkpoint 出发，仅在 task 8（moka pots，29 eps）上继续训练 500 步。
4. **task 8 历史最优 checkpoint 全面扫描**：检索全部 eval 日志，跨所有方案找出 task 8（moka pots）表现最好的 checkpoint。

---

## 二、`ft_l10_t89` 系列评测结果

### 2.1 实验配置

- **初始化**：`pi05_libero_l10_mix50_from_15k/ft_l10_mix50_from_15k/1999`（l10_mix50@2k，avg=96.6% 历史最佳）
- **训练数据**：libero_10 task 8 + task 9 = 63 episodes（moka pots + mug microwave）
- **步数**：500 steps，batch_size=32，lr peak=2e-5
- **Checkpoint 输出**：`pi05_libero_10_t89_from_l10mix50_2k/ft_l10_t89_from_l10mix50_2k/`，step 300 和 499

### 2.2 全 4 suite 评测结果（10 task × 50 episode per suite）

| Checkpoint | spatial | object | goal | libero_10 | avg(4) | eval Job |
|---|---|---|---|---|---|---|
| l10_mix50 @ step 2000（基线） | 98.8% | 98.6% | 97.4% | 91.6% | 96.6% | — |
| **`t89` @ step 300** | **98.0%** | **97.8%** | **97.4%** | **92.6%** | **96.5%** | 34569 |
| `t89` @ step 500 | **30.2%** ⚠️ | 98.8% | 97.0% | 92.0% | **80.0%** | 34629 |

### 2.3 关键发现

1. **step300 是 t89 系列的最优点**：
   - libero_10 从 91.6% 提升到 **92.6%（+1.0pp）**
   - spatial / object / goal 与 l10_mix50@2k 基本持平（最大差距 -0.8pp）
   - **avg=96.5%** 与历史最佳（96.5% / 96.6%）持平，是成功的专项强化

2. **step500 出现 spatial 灾难性遗忘**：
   - libero_spatial 从 98.0% 暴跌至 **30.2%**（-67.8pp），完全崩溃
   - object/goal/libero_10 仍正常（98.8/97.0/92.0%）
   - 推断：step300→500 的额外 200 步将 LLM 参数推离了 spatial 的有效子空间
   - 与历史规律完全一致：**小数据集专项 FT 存在极窄的 sweet spot**，step300 之后立即退化

3. **task 8 (moka pots) 仍为瓶颈**：
   - step300: libero_10 avg=92.6%，但 task 8 仍未显式统计（期望 ~0.6~0.7）
   - 专项 FT 63 eps（task8+9）对 libero_10 整体有温和提升，但 task 8 顽固性依然存在

---

## 三、简单加权平均融合（Job 34645）

### 3.1 融合配置

| 项目 | 值 |
|---|---|
| 方法 | `wudi_merge.py --iter 0`（纯加权初始，无 WUDI 优化） |
| base | `pi05_libero/my_experiment/15000`（base_15k，权重 0.3） |
| ft | `ft_l10_t89_from_l10mix50_2k/300`（t89_step300，权重 0.7） |
| `--init-weights` | `0.7` |
| scope | `llm_only`（视觉编码器保持 base_15k，action 投影 frozen） |
| 输出 | `avg_merge/l10t89_step300_x07_base15k_x03` |

**公式**：`merged_LLM = 0.7 × t89_step300 + 0.3 × base_15k`

**scope 对各参数组的影响**：

| 参数组 | 处理方式 |
|---|---|
| LLM attn/FFN + norms | 0.7 × t89_step300 + 0.3 × base_15k |
| 视觉编码器（SigLIP） | 100% base_15k（不参与融合） |
| Frozen 前缀（embedder / action proj / time_mlp） | 100% base_15k |

### 3.2 Job 记录

| Job | 状态 | 耗时 | 节点 |
|---|---|---|---|
| 34645 | ✅ COMPLETED (ExitCode 0:0) | 1m48s | gnho031 |

输出：`checkpoints/avg_merge/l10t89_step300_x07_base15k_x03/{params/, assets/}`

### 3.3 融合 checkpoint 评测

| Job | 状态 | 说明 |
|---|---|---|
| 34646 | ❌ SIGTERM | gnho031 不稳定，15min 被杀（task 1 ep47） |
| **34656** | ✅ COMPLETED | 重提，gnho031，libero_10 = **464/500 = 92.8%** |

**libero_10 成绩**：avg_merge checkpoint = **92.8%**（vs l10_mix50@2k 基线 91.6%，+1.2pp）。与 t89@step300 直接 FT（92.6%）相当，符合预期。

---

## 四、task 8 专项 FT（Job 34647）

### 4.1 配置

| 项目 | 值 |
|---|---|
| 初始化 | `avg_merge/l10t89_step300_x07_base15k_x03/params` |
| 训练数据 | `LIBERO_10_TASK8_EPISODES`（29 eps，"put both moka pots on the stove"） |
| steps | 500，batch_size=32，2 GPU FSDP |
| lr_schedule | CosineDecay，warmup=50，peak=2e-5，decay=500，decay_lr=2e-6 |
| Config 名 | `pi05_libero_10_t8_from_avg_x07` |

### 4.2 Job 记录

| Job | 状态 | 耗时 | 节点 |
|---|---|---|---|
| 34647 | ✅ COMPLETED (ExitCode 0:0) | 10m32s | gnho031 |

Checkpoint 输出：
```
checkpoints/pi05_libero_10_t8_from_avg_x07/ft_l10_t8_from_avg_x07/
├── 300/
└── 499/
```

### 4.3 评测结果（Job 34661）

- 评测套件：libero_10 only，两步骤顺序跑
- step300 完整完成；step499 在 task 0 ep27 时 SIGTERM（gnho031 不稳定）

**step300 结果（完整）**：

| Task（eval 顺序） | 成功/总数 | 成功率 |
|---|---|---|
| task 0–7 合计 | — | — |
| **task 8 (moka pots)** | **23/50** | **46%** |
| libero_10 整体 | **454/500** | **90.8%** |

**结论**：task 8 专项 FT 反而从 avg_merge 基线（期望 ~54%）退化至 **46%**，整体 libero_10 也从 92.8% 降至 90.8%（-2.0pp）。分析：

1. **29 demos 过少**：模型在 task 8 上过拟合，泛化能力下降
2. **起点受损**：avg_merge 的 LLM 参数是两个 checkpoint 的折中，本身已有轻微损耗
3. **极窄 sweet spot**：29 eps 的专项 FT 在 step300 处已超出最优点，不如直接 t89（63 eps task8+9）

---

## 五、SLURM 任务汇总（2026-04-29）

| Job | 类型 | 说明 | 节点 | 状态 |
|---|---|---|---|---|
| 34569 | eval | `t89_step300` 全 4 suite | gnho031 | ✅ 完成（98.0/97.8/97.4/92.6，avg=96.5%） |
| 34629 | eval | `t89_step500` 全 4 suite | gnho031 | ✅ 完成（**30.2**/98.8/97.0/92.0，avg=80.0%，spatial 崩溃） |
| 34645 | merge | `avg_merge/l10t89_step300_x07_base15k_x03` | gnho031 | ✅ 完成（1m48s） |
| 34646 | eval | avg_merge checkpoint libero_10 | gnho031 | ❌ SIGTERM（15min，task 1 ep47） |
| 34647 | train | task 8 专项 FT 500 steps | gnho031 | ✅ 完成（10m32s） |
| 34656 | eval | avg_merge checkpoint libero_10（重提） | gnho031 | ✅ 完成（464/500 = **92.8%**） |
| 34661 | eval | `ft_l10_t8_from_avg_x07` step300+499 libero_10 | gnho031 | ⚠️ 部分完成（step300 OK，step499 SIGTERM） |

---

## 六、task 8 历史最优 checkpoint 全面扫描

### 6.1 扫描范围

遍历全部 15 个 eval 日志，解析每个 checkpoint 的 task 8（"put both moka pots on the stove"，eval 输出中位于 index 8，0-indexed）的逐 task 成功率。

### 6.2 扫描结果（task 8 成功率排名）

| 排名 | task 8 成功率 | 成功/总数 | Checkpoint | 备注 |
|---|---|---|---|---|
| 🥇 1 | **74%** | 37/50 | `ft_from_wudi_4task_1k/9000` | pi05_libero_4task_from_wudi_4task_1k |
| 2 | 66% | 33/50 | libero10ft 二次 FT@1k | — |
| 3 | 62% | 31/50 | t89 @ step300 | 本次实验 |
| 3 | 62% | 31/50 | ft_libero_10_20k | — |
| 5 | 56% | 28/50 | FT@5k from 3task_sog_iter500 | — |
| 6 | 54% | 27/50 | l10_mix50 @ step1000 | — |
| 7 | 52% | 26/50 | FT@10k from wudi_4task_500 | — |
| 7 | 52% | 26/50 | t89 @ step500 | spatial 崩溃，不可用 |
| 9 | **48%** | 24/50 | l10_mix50 @ step2000 | 当前广义最优基线 |
| 10 | 46% | 23/50 | ft_l10_t8_from_avg_x07/300 | 本次 task8 专项 FT（退化） |
| 11 | 20% | 10/50 | mean4_protect_region_joint | — |
| 12 | 10% | 5/50 | mean4_protect_minimal_joint | — |
| 13 | 6% | 3/50 | mean4_protect_minimal_base | — |
| 14 | 4% | 2/50 | 4task_mean@iter0 | — |

### 6.3 最优 checkpoint 信息

- **路径**：`checkpoints/pi05_libero_4task_from_wudi_4task_1k/ft_from_wudi_4task_1k/9000`
- **task 8 成绩**：74%（比 l10_mix50@2k 基线 +26pp，比 t89@step300 +12pp）
- **背景**：从 wudi 4-task 1k 融合 checkpoint 出发，对全 4 task libero_10 继续 FT 到 9k 步
- **整体 avg(4)**：历史记录中 ~96.5%（与历史最佳持平）

### 6.4 分析

task 8 表现最好的路径是 **4-task 联合训练 + 长步数（9k）**，而不是 task 8 专项 FT：
- 4-task 联合训练提供了充足的 task 8 样本密度（所有 10 task 均包含）
- 长步数（9k）允许模型在不过拟合的情况下学好 moka pots
- 专项 29-eps 单 task FT 样本太少，极易过拟合

---

## 七、关键结论

### 7.1 t89 FT sweet spot 极窄（step300 = 最优）

| 指标 | l10_mix50@2k | t89@step300 | t89@step500 |
|---|---|---|---|
| libero_spatial | 98.8% | 98.0% | **30.2%** ← 崩溃 |
| libero_object | 98.6% | 97.8% | 98.8% |
| libero_goal | 97.4% | 97.4% | 97.0% |
| libero_10 | 91.6% | **92.6%** | 92.0% |
| avg(4) | **96.6%** | **96.5%** | 80.0% |

专项 FT 200 步（step300→500）引发 libero_spatial 灾难性遗忘。**step300 是唯一安全点**：libero_10 +1.0pp 且其余三 suite 无损。

### 7.2 当前最优路径汇总（更新后）

| 方案 | avg(4) | task 8 | 说明 |
|---|---|---|---|
| l10_mix50 @ step 2000 | **96.6%** ⭐ | 48% | 直接 FT，广义最优基线 |
| t89 @ step 300 | 96.5% | **62%** | task8+9 专精，task8 显著提升 |
| FT@9k from wudi_4task_1k | ~96.5% | **74%** ⭐ | **task 8 历史最优** |
| avg_merge（0.7×t89+0.3×15k） | — | ~54%（推断） | libero_10=92.8% |
| ft_l10_t8 @ step300 | — | 46% ⬇️ | 专项 FT 退化，不推荐 |

---

## 八、待办事项

1. ~~**task 8 FT 评测**~~：已完成，step300=46%，结果不理想 ✅
2. ~~**avg_merge eval 重提**~~：Job 34656 完成，libero_10=92.8% ✅
3. **从 `ft_from_wudi_4task_1k/9000` 出发做专项 FT**：该 checkpoint task8=74%，可作为更好的起点尝试 task 8 进一步提升
4. **l10_mix50 延长训练**：从 step 2000 继续扫描 3k/5k，观察 avg(4) 能否突破 97%
5. **step499 重测**：`ft_l10_t8_from_avg_x07/499` 被 SIGTERM，可选择重提（但预期不佳）
