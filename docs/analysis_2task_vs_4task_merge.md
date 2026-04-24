# 2-task vs 4-task WUDI 融合差异分析

> 作者：Jiayu Sun  
> 日期：2026-04-22  
> 项目路径：`/storage/yukaichengLab/lishiwen/jiayusun/openpi`

---

## 一、背景与问题

本文分析 pi0.5 在 LIBERO 4 个 suite 上进行 **2-task** 与 **4-task** WUDI 融合时性能差异的根本原因。

使用的实验结果来自 `docs/experiment_log_2026-04-15.md`，原始日志位于 `trainlogs/` 目录。

---

## 二、澄清：4-task 并非全面劣于 2-task

首先需要纠正一个常见误解。将最优配置对比：

| 配置 | spatial | object | goal | libero_10 | avg | 数据来源 |
|---|---|---|---|---|---|---|
| 2-task spatial+10, sum, iter=500 | 95.0% | 33.6% | 18.4% | 69.4% | **54.1%** | Job 33822 |
| 2-task spatial+10, sum, iter=1k  | 94.4% | 32.8% | 18.6% | 70.4% | **54.1%** | Job 33823 |
| 2-task goal+10, mean, iter=1     | 55.4% | 64.0% | 77.4% | 80.8% | **69.4%** | Job 34058 |
| 2-task goal+10, mean, iter=10    | 54.2% | 65.2% | 79.6% | 83.2% | **70.5%** | Job 34059 |
| **4-task mean, iter=0**          | 92.4% | 90.0% | 53.0% | 49.2% | **71.2%** | Job 33893 |

**4-task mean@iter=0 是所有非 FT 方案中均值最高的结果（71.2%）**，并不差于 2-task。

真正的问题是两个不同的现象：
1. **4-task sum init**：结果远差于 2-task（iter=300 avg=26.6%，libero_10=0%）
2. **4-task mean init**：虽然 iter=0 最优，但 WUDI 优化立即使性能退化（iter=10 avg=68.9%，iter=500 avg=59.5%，iter=5k 崩溃）

---

## 三、实验数据全景

### 3.1 2-task 融合结果

**配置**：base=`checkpoints/pi05_libero/my_experiment/10000`，ft=`from10k/ @20k`，scope=llm_only，scaling=1.0

| 配置 | spatial | object | goal | libero_10 | avg | 数据来源 |
|---|---|---|---|---|---|---|
| spatial+10, sum, iter=300 | 94.2% | 32.4% | 18.4% | 66.8% | 53.0% | Job 33822 |
| spatial+10, sum, iter=500 | 95.0% | 33.6% | 18.4% | 69.4% | 54.1% | Job 33822 |
| spatial+10, sum, iter=1k  | 94.4% | 32.8% | 18.6% | 70.4% | 54.1% | Job 33823 |
| spatial+10, sum, iter=5k  | 89.8% | 22.4% | 12.0% | 57.4% | 45.4% | Job 33603 |
| spatial+10, mean, iter=500 | 95.4% | 32.8% | 16.8% | 70.8% | 54.0% | Job 33830 |
| goal+10, mean, iter=1     | 55.4% | 64.0% | 77.4% | 80.8% | 69.4% | Job 34058 |
| goal+10, mean, iter=10    | 54.2% | 65.2% | 79.6% | 83.2% | 70.5% | Job 34059 |
| goal+10, mean, iter=500   | 45.4% | 50.2% | 69.0% | 77.2% | 60.5% | Job 34060 |
| goal+10, mean, iter=5k    | 17.6% | 25.4% | 29.2% | 74.6% | 36.7% | Job 34018 |
| object+10, mean, iter=1   | 37.8% | 87.6% | 21.0% | 68.8% | 53.8% | Job 34047+34064 |
| object+10, mean, iter=500 | 27.4% | 80.8% | 18.0% | 65.6% | 47.9% | Job 34066 |

### 3.2 4-task 融合结果

**配置**：同上，ft=4 个 suite 各自 @20k，scope=llm_only

| 初始化 | iter | spatial | object | goal | libero_10 | avg | 数据来源 |
|---|---|---|---|---|---|---|---|
| mean | 0 | 92.4% | 90.0% | 53.0% | 49.2% | **71.2%** | Job 33893 |
| mean | 1 | 92.4% | 90.2% | 53.0% | 49.4% | 71.2% | Job 33966 |
| mean | 10 | 92.0% | 90.4% | 50.0% | 43.0% | 68.9% | Job 33968 |
| mean | 100 | 86.4% | 90.2% | 46.6% | 31.8% | 63.8% | Job 33894 |
| mean | 300 | 86.8% | 90.0% | 42.8% | 22.2% | 60.5% | Job 33895 |
| mean | 500 | 85.8% | 90.6% | 43.0% | 18.6% | 59.5% | Job 33829 |
| sum  | 300 | 45.6% | 48.8% | 12.0% |  0.0% | 26.6% | Job 33821 |
| sum  | 500 | 53.8% | 52.8% | 28.2% |  3.2% | 34.5% | Job 33801 |
| sum  | 1k  | 60.0% | 62.2% | 42.8% |  2.0% | 41.8% | Job 33796 |
| sum  | 5k  |  9.0% |  0.2% | 10.0% |  0.2% |  4.9% | Job 33750 |
| sum  | 10k |  0.0% |  0.0% |  0.0% |  0.0% |  0.0% | Job 33777 |

---

## 四、理解差异的关键：单任务 FT 的灾难性遗忘

### 4.1 实验数据（来源）

以下数据来自单任务 FT @20k 的跨 suite 评测，每个 FT 模型在全部 4 个 suite 上各运行 10 task × 5 episode = 50 episode：

| 模型 | spatial | object | goal | libero_10 | 数据来源 |
|---|---|---|---|---|---|
| spatial FT @20k | **98.0%** | 0.0% | 9.8% | 0.0% | Job 33944 (trainlogs/eval_ft_libero_spatial_20k.33944.err) |
| object FT @20k  | 1.4% | **98.2%** | 0.0% | 0.2% | Job 33953 (trainlogs/eval_ft_libero_object_20k.33953.err) |
| goal FT @20k    | 37.2% | 2.4% | **95.8%** | 0.0% | Job 33952 (trainlogs/eval_ft_libero_goal_20k.33952.err) |
| libero_10 FT @20k | 17.4% | 34.2% | 10.0% | **92.6%** | Job 33920 (trainlogs/eval_ft_libero_10_20k.33920.err) |

**关键观察**：
- spatial / object FT：自身 suite >98%，其余 suite 接近 0%——**高度特化，严重灾难性遗忘**
- goal FT：自身 95.8%，但 spatial 方向仍保留 37.2%——**存在跨任务泛化**
- libero_10 FT：OOD 均值 20.5%，是 4 个模型中最高的——**任务向量方向最"分散"**

### 4.2 派生量：task vector 方向的代理表示

**定义**：将上表每行看作 task vector τ 在"4 个 suite 性能方向"构成的低维空间中的投影向量。按 L2 范数归一化后，可作为 τ 相对方向的代理（近似）：

| task vector | 归一化方向（代理） | 计算方式 |
|---|---|---|
| τ_spatial   | [0.995, 0.000, 0.100, 0.000] | [98.0, 0.0, 9.8, 0.0] / 98.49 |
| τ_object    | [0.014, 1.000, 0.000, 0.002] | [1.4, 98.2, 0.0, 0.2] / 98.21 |
| τ_goal      | [0.362, 0.023, 0.932, 0.000] | [37.2, 2.4, 95.8, 0.0] / 102.80 |
| τ_libero_10 | [0.173, 0.340, 0.099, 0.919] | [17.4, 34.2, 10.0, 92.6] / 100.73 |

> **注意**：这是低维代理（4 维），真实 task vector 维度约 2.3B。这里用于定性分析任务间干扰关系，而非精确量化。

由此计算**近似余弦相似度**：

| τ_i / τ_j | spatial | object | goal | libero_10 |
|---|---|---|---|---|
| spatial   | 1.000 | **0.014** | 0.453 | 0.182 |
| object    | 0.014 | 1.000 | **0.029** | 0.344 |
| goal      | 0.453 | 0.029 | 1.000 | 0.163 |
| libero_10 | 0.182 | 0.344 | 0.163 | 1.000 |

**结论**：spatial 与 object 的 task vector **近乎正交**（cos=0.014），spatial/object 也与 goal/libero_10 部分正交。4 个向量张成了近乎正交的独立方向。

---

## 五、为什么 4-task WUDI 优化立即有害

### 5.1 WUDI 梯度的"向零坍缩"效应

WUDI 在 mean init 处，对**完全正交**的 N 个 task vector 的梯度可精确推导：

```
初始化：τ_m = (1/N) Σ_i τ_i
对任意 τ_t：τ_m - τ_t = -(N-1)/N · τ_t  [正交条件下 Σ_{j≠t} τ_j 与 τ_t 无关]

WUDI 损失对 τ_m 的梯度：
∂L/∂τ_m = 2 Σ_t (τ_m - τ_t)(τ_t^T τ_t)/||τ_t||^2
         ≈ -2(N-1) · τ_m   [近正交时]
```

| N（task 数） | 梯度方向 | 梯度幅值 |
|---|---|---|
| N=2 | 与 τ_m 反向 | -2τ_m |
| N=4 | 与 τ_m 反向 | **-6τ_m**（3× 大）|

数值验证（上述代理模型计算）：

| 配置 | 梯度方向与 τ_m 的余弦 | 初始 WUDI 干扰量 L |
|---|---|---|
| 2-task goal+10，mean init | -1.000 | 0.350 |
| 4-task，mean init         | -0.999 | **1.453**（4.1× 更大）|

**两者梯度都指向 -τ_m 方向**——WUDI 的唯一"解"是零向量。区别在于速度：
- N=4：梯度幅值 3× 更大，初始干扰量 4× 更大，τ_m 被快速压缩 → iter=10 就开始明显退化
- N=2：梯度幅值更小，可以有一段 "微调窗口" 在缩向零之前先消除局部干扰

### 5.2 4-task sum init 为什么更差

4-task sum init：τ_m = τ_1 + τ_2 + τ_3 + τ_4

对于 task t，干扰项为：`τ_m - τ_t = Σ_{j≠t} τ_j`（另外 3 个 task vector 之和）

即使 task vector 两两近正交，3 个向量之和与 τ_t 的干扰来自 3 份叠加。  
对比 2-task sum init：干扰项 = 仅 1 个 task vector，且若两者近正交，干扰极小。

这解释了实验结果（Jobs 33821 vs 33822）：
- **2-task sum@300~1k**：从接近零干扰的点出发，WUDI 微调有效（avg≈54%）
- **4-task sum@300**：libero_10 在 iter=300 就崩溃为 0.0%（被 3 个方向的交叉干扰压垮）

---

## 六、为什么 libero_10 在 4-task 中最先崩溃

从归一化方向看各 task vector 的"特化程度"：

| task | 最大分量 | 其余分量均值 | 特化程度 |
|---|---|---|---|
| spatial   | 0.995（spatial方向） | 0.033 | 极高 |
| object    | 1.000（object方向）  | 0.005 | 极高 |
| goal      | 0.932（goal方向）    | 0.128 | 高（但有 spatial 分量 0.362）|
| libero_10 | 0.919（libero_10方向）| 0.204 | **最低，各方向均有分布** |

4-task mean 融合后，4 个 task vector 各占 1/4 权重。WUDI 压缩 τ_m 时：
- spatial / object task vector 方向最"尖锐"（高自相关）→ 在 WUDI 压缩过程中，它们的方向分量相对更稳定
- libero_10 task vector 最"分散"→ 在 4 方向竞争下贡献最小 → 最先被 WUDI 梯度消除

这与实验数据吻合（来源：Jobs 33893/33966/33968/33894/33895/33829）：

```
iter=0:   spatial=92.4%  object=90.0%  goal=53.0%  libero_10=49.2%
iter=10:  spatial=92.0%  object=90.4%  goal=50.0%  libero_10=43.0%   ← libero_10 先跌
iter=100: spatial=86.4%  object=90.2%  goal=46.6%  libero_10=31.8%   ← libero_10 跌幅最大
iter=500: spatial=85.8%  object=90.6%  goal=43.0%  libero_10=18.6%   ← libero_10 持续崩溃
```

注意 spatial 和 object 在 iter=10~500 之间几乎保持稳定（90%），而 libero_10 从 49.2% 单调下降至 18.6%。

---

## 七、为什么 goal+10 是最优的 2-task pair

同样基于单任务 FT 数据（Jobs 33920/33944/33952/33953）：

| 2-task pair | OOD 覆盖机制 |
|---|---|
| spatial+10 | spatial FT OOD≈0%；仅靠 libero_10 的 OOD(object=34.2%) 覆盖其他 suite → 3 个 OOD suite 均差 |
| object+10  | object FT OOD≈0.5%；libero_10 覆盖 spatial(17.4%) → spatial 只有 37.8% |
| **goal+10** | goal FT OOD@spatial=37.2%（自身保留了跨任务能力）+ libero_10 OOD@object=34.2% → 4 个 suite 均有底线 |

goal task vector 方向更"居中"（cos(goal, spatial)=0.453 是所有非自身 pair 中最大的），mean merge 后的 τ_m 在各 suite 方向上的投影更均衡：

```
2-task goal+10 mean init 的τ_m 方向（代理）：
  [0.267, 0.181, 0.516, 0.460]
→ 4 个方向都有显著分量（最小值 0.181）

对比 2-task spatial+10 mean init：
  [(0.995+0.173)/2, (0+0.340)/2, (0.100+0.099)/2, (0+0.919)/2]
  = [0.584, 0.170, 0.100, 0.460]
→ goal 方向分量仅 0.100，解释了 goal suite 在 spatial+10 merge 后仅 18% 的结果
```

---

## 八、结论汇总

| 现象 | 根本原因 |
|---|---|
| 4-task mean@iter=0 = 71.2%（非 FT 最优） | 4 个 task vector 覆盖 4 个 suite 方向，mean 是无偏起点 |
| 4-task WUDI 优化从 iter=1 即退化 | N=4 近正交向量 → WUDI 梯度 ≈ -6τ_m，唯一最优解是零向量；初始干扰量是 2-task 的 4.1× |
| 2-task sum@300~1k 有有效窗口（avg≈54%） | 近正交 2-task sum init 干扰极小，WUDI 微调可消除残余干扰，iter>1k 后才过优化 |
| 4-task sum@300 libero_10=0% | 3 个方向的交叉干扰叠加，sum init 起点已是高干扰状态 |
| libero_10 在 4-task 中最先被 WUDI 压垮 | libero_10 task vector 最分散（OOD 均值 20.5%），在 4 方向竞争中抗压缩能力最弱 |
| goal+10 OOD 覆盖最好 | goal FT 在 spatial 方向保留 37.2%（其他 FT ≈ 0~3%），goal task vector 方向居中（cos≈0.45 with spatial） |

### 实践建议

1. **最优非 FT 融合**：4-task mean@iter=0 = 71.2%，无需任何 WUDI 优化
2. **如需 FT**：从 4-task mean@iter=0 出发，FT 10k 步可达 avg=96.5%（Jobs 33817/33819）
3. **2-task 最优 pair**：goal+10（avg≈70%），接近 4-task mean@iter=0
4. **不要对 4-task 做 WUDI 优化**：任何 iter > 0 都是纯损失；4-task 是"已过约束"系统，mean 本身就是 Pareto 最优点

---

## 附录：关键日志文件索引

| 数据 | 日志文件 |
|---|---|
| spatial FT @20k 跨 suite 评测 | `trainlogs/eval_ft_libero_spatial_20k.33944.err` |
| object FT @20k 跨 suite 评测 | `trainlogs/eval_ft_libero_object_20k.33953.err` |
| goal FT @20k 跨 suite 评测 | `trainlogs/eval_ft_libero_goal_20k.33952.err` |
| libero_10 FT @20k 跨 suite 评测 | `trainlogs/eval_ft_libero_10_20k.33920.err` |
| 4-task mean iter=0/1/10/100/300/500 | `trainlogs/eval_wudi_*_mean_iter*.*.err` (Jobs 33893/33966/33968/33894/33895/33829) |
| 2-task spatial+10 sum iter=300/500/1k/5k | `trainlogs/eval_wudi_2task_300.33822.err` 等 |
| 2-task goal+10 mean iter=1/10/500/5k | `trainlogs/eval_wudi_2task_goal10_mean_iter*.34058-34060.err` |
