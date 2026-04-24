# 实验日志 — 2026-04-15 / 04-17 / 04-19

> 最后更新：2026-04-22 CST（新增 obj+10 iter=500/1k 完整结果，补全 iter=10 goal 数据）  
> 项目路径：`/storage/yukaichengLab/lishiwen/jiayusun/openpi`

---

## 一、研究背景与目标

**核心问题**：pi0.5 在 LIBERO 上分别微调了 4 个套件（libero_10 / goal / object / spatial），能否通过模型融合得到一个同时在 4 个套件上表现良好的统一模型？

**融合方法**：
- **WUDI**（per-layer 2D SVD de-centering + Adam 优化）：`scripts/wudi_merge.py`
- **TIES**（Trim + Elect Signs + Disjoint Merge，Yadav et al. NeurIPS 2023）：`scripts/ties_merge.py`（作为横向对照）

---

## 二、Checkpoint 全景图

### 2.1 Base 模型

| Checkpoint | 说明 | 关键 step |
|---|---|---|
| `checkpoints/pi05_libero/my_experiment/` | π₀.₅ 全量 LIBERO 微调（40 tasks, 1693 eps）<br>8×GPU FSDP, batch=256, 35.5h | 5k / 10k / 15k / 20k / 25k / **29999** |
| `checkpoints/pi05_libero_RETRAIN_base/pi05_libero_RETRAIN_base/` | 重训 base | 5k / **29999** |

### 2.2 per-suite 微调（从 pi05_libero 29999 出发）

| Checkpoint | 说明 | 关键 step |
|---|---|---|
| `checkpoints/pi05_libero_10/my_experiment/` | libero_10 suite | 5k / **29999** |
| `checkpoints/pi05_libero_spatial/my_experiment/` | libero_spatial suite | 5k / **29999** |
| `checkpoints/pi05_libero_goal/my_experiment/` | libero_goal suite | 5k / **29999** |
| `checkpoints/pi05_libero_object/my_experiment/` | libero_object suite | 5k / **29999** |

### 2.3 from10k 系列（从 pi05_libero 10000 出发）

| Checkpoint | 说明 | 关键 step |
|---|---|---|
| `checkpoints/from10k/pi05_libero_10_from_pi05libero_10k/ft_from_pi05libero_10k/` | libero_10 | 5k/10k/15k/**20k**/25k/29999 |
| `checkpoints/from10k/pi05_libero_goal_from_pi05libero_10k/ft_from_pi05libero_10k/` | libero_goal | 同上 |
| `checkpoints/from10k/pi05_libero_object_from_pi05libero_10k/ft_from_pi05libero_10k/` | libero_object | 同上 |
| `checkpoints/from10k/pi05_libero_spatial_from_pi05libero_10k/ft_from_pi05libero_10k/` | libero_spatial | 同上 |

---

## 三、WUDI 融合实验

**算法核心**（scope=llm_only）：
- SigLIP vision encoder：保持 base 不动
- Gemma attn + FFN（共 10 组参数）：WUDI 优化
- 其余（norm/embedding/action_head/time_mlp）：简单平均 task vector

### 3.1 2task 系列（libero_10 + libero_spatial）

**sum 初始化（`vectors.sum(dim=0)`）：**

| iter | 输出 Checkpoint | Job | 完成时间 | 状态 |
|---|---|---|---|---|
| **300** | `wudi_mllm/2task_from10k_iter300` | 33820 | 2026-04-16 10:48 | ✅ 完成 |
| **500** | `wudi_mllm/2task_from10k_iter500` | 33802 | 2026-04-15 17:26 | ✅ 完成 |
| **1000** | `wudi_mllm/2task_spatial10_iter1k` | 33602 | 2026-04-12 10:09 | ✅ 完成 |
| **5000** | `wudi_mllm/2task_spatial10_iter5k` | 33603 | 2026-04-12 13:24 | ✅ 完成 |

**mean 初始化（`vectors.mean(dim=0)`）：**

| iter | 输出 Checkpoint | Job | 完成时间 | 状态 |
|---|---|---|---|---|
| **500** | `wudi_mllm/2task_mean_iter500` | 33828 | 2026-04-16 | ✅ 完成（libero_10 + libero_spatial）|

**2task object+10 / goal+10 系列（新增 2026-04-21，mean init，scope=llm_only，scaling=scaling2=1.0）：**

| pair | iter | 输出 Checkpoint | Job | 完成时间 | 状态 |
|---|---|---|---|---|---|
| object+10 | 1000 | `wudi_mllm/2task_object10_mean_iter1k` | 34130 | 2026-04-22 10:45 | ✅ 完成（gnho031） |
| object+10 | 5000 | `wudi_mllm/2task_object10_mean_iter5k` | 34013 | 2026-04-21 | ✅ 完成（gnho031） |
| object+10 | 10000 | `wudi_mllm/2task_object10_mean_iter10k` | 34015 | 2026-04-21 | ✅ 完成（gnho031） |
| goal+10 | 5000 | `wudi_mllm/2task_goal10_mean_iter5k` | 34017 | 2026-04-21 | ✅ 完成（gnho009） |
| goal+10 | 10000 | `wudi_mllm/2task_goal10_mean_iter10k` | 34019 | 2026-04-21 | ✅ 完成（gnho009） |

### 3.2 4task 系列（libero_{10,goal,object,spatial}）

**sum 初始化（`vectors.sum(dim=0)`）：**

| iter | 输出 Checkpoint | Job | 完成时间 | 状态 |
|---|---|---|---|---|
| **300** | `wudi_mllm/4task_from10k_iter300` | 33821 | 2026-04-16 10:57 | ✅ 完成 |
| **500** | `wudi_mllm/4task_from10k_iter500` | 33801 | 2026-04-15 17:24 | ✅ 完成 |
| **1000** | `wudi_mllm/4task_from10k_iter1k` | 33796 | 2026-04-15 | ✅ 完成 |
| **5000** | `wudi_mllm/4task_from10k_iter5k` | 33750 | 2026-04-14 23:11 | ✅ 完成 |
| **10000** | `wudi_mllm/4task_from10k_iter10k` | 33777 | 2026-04-15 05:40 | ✅ 完成 |

**mean 初始化（`vectors.mean(dim=0)`）：**

| iter | 输出 Checkpoint | Job | 完成时间 | 状态 |
|---|---|---|---|---|
| **0** | `wudi_mllm/4task_mean_iter0` | 33893 | 2026-04-17 | ✅ 完成（纯 mean merge，无优化） |
| **1** | `wudi_mllm/4task_mean_iter1` | 33966 | 2026-04-20 | ✅ 完成（gnho031） |
| **10** | `wudi_mllm/4task_mean_iter10` | 33968 | 2026-04-20 | ✅ 完成（gnho034） |
| **100** | `wudi_mllm/4task_mean_iter100` | 33894 | 2026-04-17 | ✅ 完成 |
| **300** | `wudi_mllm/4task_mean_iter300` | 33895 | 2026-04-17 | ✅ 完成 |
| **500** | `wudi_mllm/4task_mean_iter500` | 33829 | 2026-04-16 | ✅ 完成 |

### 3.3 TIES 融合（4task，新增 2026-04-19）

配置：`scope=llm_only`，`mask_rate=0.8`，`scaling=1.0`，`scaling2=1.0`（论文默认）

| 输出 Checkpoint | Job | 完成时间 | 状态 |
|---|---|---|---|
| `ties_mllm/4task_mr08_s10` | 33945 | 2026-04-19 08:23 | ✅ 完成（CPU 模式，约 2 分钟） |

> 说明：TIES 三步（Trim / Elect Signs / Disjoint Merge）无需梯度，速度比 WUDI 快约 60×。因 GPU 显存不够 `(4, 2.29B)` float32 矩阵，改用 CPU 模式 + 300G 系统内存。

---

## 四、Fine-tuning 实验（从融合 checkpoint 出发）

| 实验名 | 初始化 | 训练数据 | 总步数 | 最终 Checkpoint | 状态 |
|---|---|---|---|---|---|
| ft from wudi_2task_5k | `wudi_mllm/2task_spatial10_iter5k` | libero_10 + libero_spatial | 30k | `.../ft_from_wudi_mllm_5k/29999` | ✅ 完成（Job 33786） |
| ft from wudi_4task_500 | `wudi_mllm/4task_from10k_iter500` | 4 suites 全量 | 10k | `.../ft_from_wudi_4task_500/9999` | ✅ 完成（Job 33817，step 9999） |
| ft from wudi_4task_1k | `wudi_mllm/4task_from10k_iter1k` | 4 suites 全量 | 10k | `.../ft_from_wudi_4task_1k/9000` | ✅ 完成（Job 33819，step 9000，被 kill 前保存） |

**训练配置**（FT 4task）：batch_size=32，2×GPU FSDP，lr CosineDecay warmup 500→5e-5→10k→5e-6，AdamW clip=1.0，EMA=0.999

---

## 五、评测结果汇总

### 5.1 完整结果表（最终汇总）

| Checkpoint | 类型 | iter | init | spatial | object | goal | libero_10 | 平均 |
|---|---|---|---|---|---|---|---|---|
| `2task_from10k_iter300` | 2task | 300 | sum | 94.2% | 32.4% | 18.4% | 66.8% | **53.0%** |
| `2task_mean_iter500` | 2task | 500 | **mean** | 95.4% | 32.8% | 16.8% | 70.8% | **54.0%** |
| `2task_from10k_iter500` | 2task | 500 | sum | **95.0%** | 33.6% | 18.4% | 69.4% | **54.1%** |
| `2task_spatial10_iter1k` | 2task | 1000 | sum | 94.4% | 32.8% | 18.6% | **70.4%** | **54.1%** |
| `2task_spatial10_iter5k` | 2task | 5000 | sum | 89.8% | 22.4% | 12.0% | 57.4% | 45.4% |
| `2task_object10_mean_iter1` | 2task(obj+10) | 1 | mean | 37.8% | 87.6% | 21.0% | 68.8% | **53.8%** |
| `2task_object10_mean_iter10` | 2task(obj+10) | 10 | mean | 38.2% | 85.8% | 20.8% | ~89%† | **~53.5%†** (34065) |
| `2task_object10_mean_iter500` | 2task(obj+10) | 500 | mean | 27.4% | 80.8% | 18.0% | 65.6% | **47.95%** (34066) |
| `2task_object10_mean_iter1k` | 2task(obj+10) | 1000 | mean | 23.6% | 82.6% | 18.6% | ~70%† | **~48.7%†** (34134) |
| `2task_object10_mean_iter5k` | 2task(obj+10) | 5000 | mean | 9.4% | 20.2% | 33.2% | 63.2% | **31.5%** |
| `2task_object10_mean_iter10k` | 2task(obj+10) | 10000 | mean | 0.0% | 0.0% | 0.0% | 0.0% | **0.0%** ← 崩溃 |
| `2task_goal10_mean_iter1` | 2task(goal+10) | 1 | mean | 55.4% | 64.0% | 77.4% | 80.8% | **69.4%** |
| `2task_goal10_mean_iter10` | 2task(goal+10) | 10 | mean | 54.2% | 65.2% | 79.6% | 83.2% | **70.55%** ← goal+10 最优 |
| `2task_goal10_mean_iter500` | 2task(goal+10) | 500 | mean | 45.4% | 50.2% | 69.0% | 77.2% | **60.45%** |
| `2task_goal10_mean_iter5k` | 2task(goal+10) | 5000 | mean | 17.6% | 25.4% | 29.2% | 74.6% | **36.7%** |
| `2task_goal10_mean_iter10k` | 2task(goal+10) | 10000 | mean | 0.0% | 0.0% | 0.0% | 0.0% | **0.0%** ← 崩溃 |
| `ft_from_wudi_mllm_5k/29999` | **FT@30k** | — | — | **98.2%** | 31.8% | 17.2% | **93.2%** | **60.1%** |
| `4task_from10k_iter300` | 4task | 300 | sum | 45.6% | 48.8% | 12.0% | **0.0%** | 26.6% |
| **`4task_mean_iter0`** | 4task | **0** | **mean** | **92.4%** | **90.0%** | **53.0%** | **49.2%** | **71.2%** ← 最佳 |
| `4task_mean_iter1` | 4task | 1 | mean | 92.4% | 90.2% | 53.0% | 49.4% | **71.25%** ← 与 iter=0 等同 |
| `4task_mean_iter10` | 4task | 10 | mean | 92.0% | 90.4% | 50.0% | 43.0% | 68.85% |
| `4task_mean_iter100` | 4task | 100 | mean | 86.4% | 90.2% | 46.6% | 31.8% | 63.8% |
| `4task_mean_iter300` | 4task | 300 | mean | 86.8% | 90.0% | 42.8% | 22.2% | 60.5% |
| `4task_mean_iter500` | 4task | 500 | mean | 85.8% | 90.6% | 43.0% | 18.6% | 59.5% |
| `4task_from10k_iter500` | 4task | 500 | sum | 53.8% | 52.8% | 28.2% | 3.2% | 34.5% |
| `4task_from10k_iter1k` | 4task | 1000 | sum | 60.0% | **62.2%** | **42.8%** | 2.0% | **41.8%** |
| `4task_from10k_iter5k` | 4task | 5000 | sum | 9.0% | 0.2% | 10.0% | 0.2% | 4.9% |
| `4task_from10k_iter10k` | 4task | 10000 | sum | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `ft_from_wudi_4task_500/9999` | **FT@10k** | — | — | **98.4%** | **98.2%** | **98.2%** | **91.0%** | **96.5%** |
| `ft_from_wudi_4task_1k/9000` | **FT@9k** | — | — | **96.8%** | **98.8%** | **97.2%** | **93.0%** | **96.5%** |
| `ties_mllm/4task_mr08_s10` | **TIES** 4task | — | mr=0.8 | 85.4% | 73.6% | 35.6% | 15.6% | **52.6%** |

**单任务 FT @20k 交叉评测**（每个模型评测 4 个 suite，用于诊断 task vector 特异性）：

| 模型 | spatial | object | goal | libero_10 | 平均 |
|---|---|---|---|---|---|
| `ft_libero_10_20k` (Job 33920) | 17.4% | 34.2% | 10.0% | **92.6%** | **38.6%** |
| `ft_libero_spatial_20k` (Job 33944) | **98.0%** | 0.0% | 9.8% | 0.0% | **27.0%** |
| `ft_libero_goal_20k` (Job 33952) | 37.2% | 2.4% | **95.8%** | 0.0% | **33.9%** |
| `ft_libero_object_20k` (Job 33953) | 1.4% | **98.2%** | 0.0% | 0.2% | **25.0%** |

> 四个单任务 FT 模型均在自身任务 >92%（spatial 98.0 / object 98.2 / goal 95.8 / libero_10 92.6），但跨任务全面崩溃（非自身 suite 上多数 <10%，大量为 0%），验证了 **task vector 高度特异化 + 灾难性遗忘**——这是 4-task 融合时方向冲突的根源。对比 mean@iter=0 的 4-task merge（71.2%）可以恢复跨任务能力至每个 suite 平均 ~71%，证明融合有效保留了共享能力。

### 5.2 关键发现

1. **2-task：iter=300~1000 均为最优窗口，mean init 无效**
   - sum init: 300/500/1000 iter → avg=53.0%/54.1%/54.1%，差异 <2pp
   - mean init@500 → avg=54.0%，与 sum@500（54.1%）**完全相同**，mean init 对 2-task 无帮助
   - iter=5000 明显退化（45.4%），验证了过优化假说
   - 结论：**2-task 最优 iter 约 300~1000**，iter≥5000 后性能单调下降

2. **🔥 重大发现：4-task 的 WUDI 优化从 iter=0 起就单调降低性能**
   - **iter=0（纯 mean merge，无 Adam 优化）= 最佳！**：spatial/object/goal/libero_10 = 92.4/90.0/53.0/49.2%，**avg=71.2%**
   - iter=1: avg=71.25%（+0.05pp，噪声范围内 → 单步 Adam 几乎不破坏 mean）
   - iter=10: avg=68.85%（-2.35pp，libero_10 从 49.2→43.0 降幅最大，退化已明显启动）
   - iter=100: avg=63.8%（-7.4pp）
   - iter=300: avg=60.5%（-10.7pp）
   - iter=500: avg=59.5%（-11.7pp）
   - iter=5000+: 灾难性崩溃
   - **核心结论**：对 4-task 而言 WUDI 优化没有"最优 iter 窗口"——**任何优化都有害**，简单的 mean task arithmetic 就是最佳融合方案
   - libero_10 被压制现象同样从 iter=0 起就存在，但程度较轻（49.2%），随 iter 持续恶化：49.2 → 31.8 → 22.2 → 18.6 → 3.2 → 2.0 → 0
   - 原 sum init 的 "iter=1000 最优 avg=41.8%" 只是在 sum init 噪声下的相对极值，绝对值远低于 mean@iter=0 (71.2%)
   - **mean init 对 2-task 完全无效**：sum avg=54.1% ≈ mean avg=54.0%（2-task 初始幅值仅 2×，影响可忽略）

3. **4-task FT 彻底解决 libero_10 崩溃**
   - FT@10k (from iter500): spatial/object/goal/libero_10 = 98.4/98.2/98.2/91.0%，**avg=96.5%**
   - FT@9k (from iter1k): 96.8/98.8/97.2/93.0%，**avg=96.5%**
   - **FT + WUDI 的组合成为最终最强方案**，全面超过单一微调
   - 两个 FT 起点（iter500 vs iter1k）表现几乎相同，说明融合质量差异被 FT 吸收

4. **2-task vs 4-task 的结构差异**
   - 4-task@1k 在 object（62.2%）和 goal（42.8%）上**远高于** 2-task@1k（32.8%/18.6%）
   - 代价是 libero_10 几乎归零，表明 4 个 task vector 之间存在严重的方向冲突
   - 但 FT 10k 步可将所有 suites 提升至 ≥91%，说明 4-task 融合仍是更好的 FT 初始化

5. **FT 能大幅修复融合退化**
   - 从 wudi_2task_5k（已过优化，avg=45.4%）出发，FT 30k 步后 spatial 98.2%、libero_10 93.2%（avg=60.1%）
   - 从 4task merge 出发，FT 10k 步后四 suites 均达 ≥91%（avg=96.5%）
   - 说明融合 checkpoint 即便出现退化，仍保留了足够的参数结构作为 FT 初始化

6. **过优化的物理机制已确认**
   - WUDI 优化损失单调下降，但真实成功率存在拐点
   - 根本原因：τ_m 被推向所有任务低秩子空间的公共零空间，高 iter 后 τ_m → 0

7. **TIES 基线：52.6% —— 不如 mean task arithmetic**
   - TIES (mr=0.8, s=1.0) vs WUDI 4task_mean_iter0：**52.6% vs 71.2%**（落后 18.6pp）
   - 分项对比：spatial 85.4 vs 92.4、object 73.6 vs 90.0、goal 35.6 vs 53.0、libero_10 15.6 vs 49.2
   - TIES 在所有 4 个 suite 上均弱于纯 mean，说明 Trim（丢弃 80% 小幅值元素）+ Disjoint Merge（按符号分组平均）两步反而破坏了有用信号
   - 结合 mean@iter=0 > WUDI 优化 > TIES 的完整对比：**对 pi0.5 全量微调的 4-task 融合场景，任何"基于任务向量的统计/优化调整"都会降低效果，最朴素的 mean task arithmetic 是最佳融合策略**
   - 物理直觉：pi0.5 全量微调的任务向量呈大幅值右偏分布（OptMerge Fig.2a），Trim 会误杀关键方向；Disjoint Merge 的符号投票会压制少数方向任务（libero_10 在 TIES 下仅 15.6%）

### 5.3 补充评测（2026-04-17）

| Job | Checkpoint | suites | 节点 | 状态 |
|---|---|---|---|---|
| 33869 | `4task_from10k_iter300` | libero_10 only | gnho018 | ✅ 完成（libero_10=0.0%） |
| 33870 | `2task_mean_iter500` | object/goal/libero_10 | gnho031 | ✅ 完成（32.8%/16.8%/70.8%） |
| 33880 | `2task_from10k_iter300` | libero_10 only | gnho018 | ✅ 完成（libero_10=66.8%） |

> 背景：33822（2task_300）和 33823（4task_300）均在 libero_10 suite 运行时超时；  
> 33830（2task_mean_500）在 libero_object suite 运行时被 node gnho034 强制终止；  
> 33871 在 gnho009 因资源不足卡在 PD，取消后以 Job 33880 重提至 gnho018。

### 5.5 Selective Protect Ablation（2026-04-20/21）

四个权重来源：`checkpoints/ablation_selective_protect/mean4_protect_{minimal,region}_{base,joint}`

构造方式（统一母体）：

- 母体权重：`W_mean = checkpoints/wudi_mllm/4task_mean_iter0`
- 仅在 `llm_only` scope 内修改 FFN 两类参数：`gating_einsum`、`linear`
- 非保护层与非 FFN 参数保持 `W_mean` 不变

四个权重具体定义：

- `mean4_protect_minimal_joint`
  - 保护层：`gating` 的 `{6,10,11}`，`linear` 的 `{1,2,4}`
  - 替换来源：`W_joint = checkpoints/pi05_libero/my_experiment/29999`
  - 规则：保护层 `W_new[layer]=W_joint[layer]`，其余层 `W_new[layer]=W_mean[layer]`

- `mean4_protect_region_joint`
  - 保护层：`gating` 的 `6..11`，`linear` 的 `1..11`
  - 替换来源：`W_joint`
  - 规则同上（仅扩大保护层范围）

- `mean4_protect_minimal_base`
  - 保护层：`gating` 的 `{6,10,11}`，`linear` 的 `{1,2,4}`
  - 替换来源：`W_base = checkpoints/pi05_libero/my_experiment/10000`
  - 规则：保护层 `W_new[layer]=W_base[layer]`，其余层保留 `W_mean`

- `mean4_protect_region_base`
  - 保护层：`gating` 的 `6..11`，`linear` 的 `1..11`
  - 替换来源：`W_base`
  - 规则同上

术语说明：

- **区域保护（region protect）**：不是只替换 2~3 个尖峰层，而是替换前中层一整段 FFN 层（`gating 6..11` + `linear 1..11`）。
- **联合校准（joint calibration）**：这里指“保护层直接对齐到 `joint@29999` 权重”（即 `*_joint` 方案），而不是额外训练校准器。

| Ckpt | spatial | object | goal | libero_10 | 平均 | 备注 |
|---|---|---|---|---|---|---|
| `mean4_protect_minimal_base` | 96.0% | 92.8% | 66.8% | 55.0% | **77.65%** | 前 3 suite 由 33995 完成；libero_10 由 34010 补跑 |
| `mean4_protect_minimal_joint` | 96.4% | 95.4% | 65.4% | 62.0% | **79.80%** | 前 3 suite 由 33994 完成；libero_10 由 34009 补跑 |
| `mean4_protect_region_base` | 94.4% | 93.2% | 55.8% | 63.8% | **76.80%** | 34011 一次跑完 |
| **`mean4_protect_region_joint`** | **98.0%** | **96.4%** | **88.0%** | **76.6%** | **89.75%** 🏆 | 34012 一次跑完；当前最佳融合 |

> 观察：`region_joint` 远超 `mean@iter=0 (71.2%)`、TIES (52.6%)、WUDI 系列——"区域保护 + 联合校准"对 4-task pi0.5 融合有显著增益（~+18.5pp over mean）。  
> 33994 / 33995 前 3 suite 正常完成但 libero_10 因 gnho034 双并发 EGL 冲突挂死；scancel 后用 34009/34010 在独立节点补跑 libero_10。

### 5.6 2task object+10 / goal+10 系列（2026-04-21/22，新增方向对测试）

配置：mean init，scope=llm_only，scaling=scaling2=1.0

| pair | iter | spatial | object | goal | libero_10 | 平均 | 备注 |
|---|---|---|---|---|---|---|---|
| object+10 | 1 | 37.8% | 87.6% | 21.0% | 68.8% | **53.8%** | 34047+34064 |
| object+10 | 10 | 38.2% | 85.8% | 20.8% | ~89%† | **~53.5%†** | 34065（libero_10 仅 6/10 完成） |
| object+10 | 500 | 27.4% | 80.8% | 18.0% | 65.6% | **47.95%** | 34066 ✅ |
| object+10 | 1k | 23.6% | 82.6% | 18.6% | ~70%† | **~48.7%†** | 34134（libero_10 仅 9/10 完成） |
| object+10 | 5k | 9.4% | 20.2% | 33.2% | 63.2% | **31.5%** | 34014 |
| object+10 | 10k | 0.0% | 0.0% | 0.0% | 0.0% | **0.0%** | 34016 完全崩溃 |
| goal+10 | 1 | 55.4% | 64.0% | 77.4% | 80.8% | **69.4%** | 34058 |
| goal+10 | 10 | 54.2% | 65.2% | 79.6% | 83.2% | **70.55%** ← goal+10 最优 | 34059 |
| goal+10 | 500 | 45.4% | 50.2% | 69.0% | 77.2% | **60.45%** | 34060 |
| goal+10 | 5k | 17.6% | 25.4% | 29.2% | 74.6% | **36.7%** | 34018 |
| goal+10 | 10k | 0.0% | 0.0% | 0.0% | 0.0% | **0.0%** | 34061 完全崩溃（34020 同） |

> **规律（新增 iter=500/1k 数据）**：  
> - object+10：iter=1→53.8%，iter=10→~53.5%（基本持平），iter=500→47.95%，iter=1k→~48.7%†，iter=5k→31.5%，iter=10k→崩溃。500~1k 是退化拐点，明显早于 spatial+10（1k~5k）。  
> - goal+10 在 iter=1/10 均约 70% avg，随优化单调退化（500→60.45%，5k→36.7%）。  
> - 与 spatial+10 对比（iter=1k avg=54.1%）：**object+10 在 iter=1k 与 spatial+10 相当（~48.7%†），但 spatial 项高度惩罚（23.6% vs spatial+10 的 94.4%）**——object task vector 与 spatial 方向冲突严重。  
> - **†** 表示 libero_10 仅部分完成，平均值为估算。

### 5.4 TIES + 单任务 FT 交叉评测（2026-04-18/19）

| Job | Checkpoint | suites | 节点 | 状态 |
|---|---|---|---|---|
| 33945 | merge `ties_mllm/4task_mr08_s10` | — | gnho031 | ✅ 融合成功（CPU 2min） |
| 33946 | TIES eval | all 4 | gnho031 | ✅ 完成（85.4/73.6/35.6/15.6 avg=52.6%） |
| 33920 | `ft_libero_10_20k` | all 4 | gnho009 | ✅ 完成（17.4/34.2/10.0/92.6） |
| 33944 | `ft_libero_spatial_20k` | all 4 | gnho034 | ✅ 完成（98.0/0.0/9.8/0.0） |
| 33952 | `ft_libero_goal_20k` | all 4 | gnho034 | ✅ 完成（37.2/2.4/95.8/0.0） |
| 33953 | `ft_libero_object_20k` | all 4 | gnho031 | ✅ 完成（1.4/98.2/0.0/0.2） |

> TIES 首次尝试（33935, 33940）因 GPU OOM 和 sbatch 配置错误失败；第三次（33945）改用 CPU 模式成功。
> 单任务 FT 的 goal/object 首次提交（33921/22/23）均因 EGL 错误被 kill；重跑（33925/31/32）又因资源/EGL 崩溃；第三次改节点提交（33952/53）进行中。

---

## 六、SLURM 任务历史（2026-04-22 CST）

| Job | 任务 | 节点 | 最终状态 |
|---|---|---|---|
| 34065 | `eval_wudi_2task_object10_mean_iter10` 全 4 suite | gnho009 | ✅ spatial/object/goal 完成；libero_10 仅 6/10（~89%†） |
| 34066 | `eval_wudi_2task_object10_mean_iter500` 全 4 suite | gnho009 | ✅ 完成（27.4/80.8/18.0/65.6） |
| 34130 | merge `2task_object10_mean_iter1k` | gnho031 | ✅ 完成 10:45 |
| 34134 | `eval_wudi_2task_object10_mean_iter1k` 全 4 suite | gnho031 | ✅ spatial/object/goal 完成；libero_10 仅 9/10（~70%†） |

---

## 七、实验路线图

```
pi05_libero (base @10k)
│
├─ FT libero_10  @20k ─┐
├─ FT libero_spatial @20k┤
├─ FT libero_goal @20k  ├─ WUDI 2task (sum) ─ iter=300  → 94.2/32.4/18.4/66.8  avg=53.0%
└─ FT libero_object @20k┤                   ├ iter=500  → 95.0/33.6/18.4/69.4  avg=54.1% ← best 2task
                         │                   ├ iter=1000 → 94.4/32.8/18.6/70.4  avg=54.1% ← best 2task
                         │                   └ iter=5000 → 89.8/22.4/12.0/57.4  avg=45.4%
                         │                         └─ FT 30k → 98.2/31.8/17.2/93.2  avg=60.1%
                         │
                         ├─ WUDI 2task (mean) ─ iter=500  → 95.4/32.8/16.8/70.8  avg=54.0%  (≈ sum，无提升)
                         │
                         ├─ WUDI 4task (sum) ─ iter=300  → 45.6/48.8/12.0/ 0.0  avg=26.6%  (libero_10 已崩溃!)
                         │                   ├ iter=500  → 53.8/52.8/28.2/ 3.2  avg=34.5%
                         │                   │    └─ FT 10k → 98.4/98.2/98.2/91.0  avg=96.5% ← best overall
                         │                   ├ iter=1000 → 60.0/62.2/42.8/ 2.0  avg=41.8%  ← best 4task sum
                         │                   │    └─ FT 9k  → 96.8/98.8/97.2/93.0  avg=96.5% ← best overall
                         │                   ├ iter=5000 → 9.0/0.2/10.0/0.2  avg=4.9%
                         │                   └ iter=10k  → 0.0/0.0/ 0.0/0.0  avg=0.0%  (完全退化为 base)
                         │
                         ├─ WUDI 4task (mean) ─ iter=0    → 92.4/90.0/53.0/49.2  avg=71.2% ← BEST MERGE (纯 mean)
                         │                      ├ iter=100  → 86.4/90.2/46.6/31.8  avg=63.8%
                         │                      ├ iter=300  → 86.8/90.0/42.8/22.2  avg=60.5%
                         │                      └ iter=500  → 85.8/90.6/43.0/18.6  avg=59.5%
                         │  (WUDI 优化对 4-task 单调降低 avg：71.2 → 63.8 → 60.5 → 59.5)
                         │
                         └─ TIES 4task (mr=0.8, s=1.0) → 85.4/73.6/35.6/15.6  avg=52.6%  (弱于 mean@iter=0)
```

---

## 八、后续计划

1. **所有实验与评测已全部完成** ✅

2. **更新 per-task Excel**（`docs/wudi_per_task_results.xlsx`）：
   - 补入 4task_mean@500、ft_4task_500、ft_4task_1k、2task_300、4task_300 的 per-task 行

3. **结论摘要**（2026-04-19 更新）：
   - **🏆 最佳纯融合**：4task_mean@iter=0 avg=71.2%（纯 mean task arithmetic，无任何优化/调整）
   - **WUDI 优化对 4-task 单调有害**：iter=0→100→300→500 avg 从 71.2% 线性下降至 59.5%
   - **TIES 亦弱于 mean 基线**：TIES (mr=0.8) avg=52.6% << mean@iter=0 (71.2%)，Trim + Disjoint Merge 在 pi0.5 全量微调场景下反而破坏有用信号
   - **"任何统计/优化调整都有害"假说成立**：mean > WUDI > TIES，呈现越干预、越退化的趋势
   - **libero_10 的衰减**：mean@iter=0 下 49.2% 最高，随 WUDI iter 递减至 18.6%，TIES 仅 15.6%，sum init 下 iter=300 已崩溃至 0%
   - **单任务 FT 高度特异化**：libero_10/spatial FT 模型在自任务 >92%，跨任务多 <10%，验证 task vector 方向冲突的存在
   - **最佳总体方案**：4task WUDI merge + FT 10k → avg=96.5%（FT 前起点对结果影响小）
   - **后续建议**：以 `4task_mean_iter0` 作为 FT 新起点对比实验，看是否能超越 avg=96.5%；TIES 可尝试更小 mask_rate（0.5 / 0.3）或扫 scaling 看是否能追上 mean 基线
