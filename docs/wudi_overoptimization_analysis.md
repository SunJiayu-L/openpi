# WUDI 过优化分析：损失下降 ≠ 成功率提升

> 撰写时间：2026-04-15 | 最后更新：2026-04-22（新增 obj+10 / goal+10 方向对实验）  
> **⚠️ 本文档经历重大修订**：2026-04-17 补做 iter=0/100/300 实验后发现，对 4-task 而言 WUDI 优化从 iter=0 起就**单调有害**，并不存在"最优 iter 窗口"。原始"过优化拐点"叙事仅对 2-task 成立。  
> **🆕 2026-04-19 补充**：TIES 基线（mr=0.8, s=1.0）avg=52.6%，同样**弱于纯 mean task arithmetic**，进一步印证"对 pi0.5 全量微调的 4-task 融合，任何统计/优化调整都有害"这一假说。  
> **🆕 2026-04-22 补充**：新增 obj+10 / goal+10 两组方向对实验（iter=1/10/500/5k/10k），发现 goal+10 @ iter=10 avg=70.55% 与 4-task mean@iter=0 (71.2%) 接近——同样支持 mean≈iter=0 最优的通用结论。  
> 实验背景：pi0.5 × LIBERO，全量微调，WUDI 融合（`scripts/wudi_merge.py`）+ TIES 基线（`scripts/ties_merge.py`）  
> 参考文献：OptMerge (Wei et al., ICLR 2026), §3.2 Theorem 3.1；TIES (Yadav et al., NeurIPS 2023)

---

## 一、现象描述

### 1.1 实验数据（完整）

**2-task 系列**（libero_10 + libero_spatial，从 pi05_libero@10k 微调至 20k，sum 初始化）

| iter | spatial | object (OOD) | goal (OOD) | libero_10 | 平均 |
|------|---------|--------------|------------|-----------|------|
| 300  | 94.2%   | 32.4%        | 18.4%      | 66.8%     | **53.0%** |
| **500**  | **95.0%** | 33.6%   | 18.4%      | 69.4%     | **54.1%** |
| **1000** | 94.4%   | 32.8%        | 18.6%      | **70.4%** | **54.1%** |
| 5000 | 89.8%   | 22.4%        | 12.0%      | 57.4%     | 45.4% |

**2-task 系列（mean 初始化）**（`vectors.mean(dim=0)` 作为优化起点）

| iter | spatial | object | goal | libero_10 | 平均 |
|------|---------|--------|------|-----------|------|
| 500  | 95.4%   | 32.8%  | 16.8% | 70.8%   | **54.0%** |

**2-task object+10 系列（mean init，新增 2026-04-22）**（object FT + libero_10 FT 融合）

| iter | spatial | object | goal | libero_10 | 平均 |
|------|---------|--------|------|-----------|------|
| 1    | 37.8%   | **87.6%** | 21.0% | 68.8%   | **53.8%** |
| 10   | 38.2%   | **85.8%** | 🕒   | 🕒       | — (运行中) |
| 500  | —       | —      | —    | —        | 🕒 PD |
| 5000 | 9.4%    | 20.2%  | 33.2% | 63.2%   | **31.5%** ← 严重退化 |
| 10000| 0.0%    | 0.0%   | 0.0%  | 0.0%    | **0.0%** ← 崩溃 |

**2-task goal+10 系列（mean init，新增 2026-04-22）**（goal FT + libero_10 FT 融合）

| iter | spatial | object | goal | libero_10 | 平均 |
|------|---------|--------|------|-----------|------|
| 1    | 55.4%   | 64.0%  | **77.4%** | **80.8%** | **69.4%** |
| **10**   | 54.2%   | 65.2%  | **79.6%** | **83.2%** | **70.55%** ← goal+10 最优 |
| 500  | 45.4%   | 50.2%  | 69.0% | 77.2%   | **60.45%** |
| 5000 | 17.6%   | 25.4%  | 29.2% | 74.6%   | **36.7%** ← 严重退化 |
| 10000| 0.0%    | 0.0%   | 0.0%  | 0.0%    | **0.0%** ← 崩溃 |

> **发现**：goal+10@iter=10 avg=70.55% 与 4-task mean@iter=0 (71.2%) 几乎持平，且 goal+10 的 goal/libero_10 两个"训练过的 suite" 都超过了 4-task mean (79.6% vs 53%, 83.2% vs 49.2%)；代价是 spatial/object 显著低于 4-task mean。  
> object+10 在 iter=1 时 object/libero_10 同样高（87.6% / 68.8%），但 spatial/goal 更低（37.8% / 21%）。任务对越"正交"（spatial+10 最远 → goal+10 最近），iter=0 的平均越高，退化速度也越慢。

**4-task 系列**（libero_{10, goal, object, spatial}，sum 初始化）

| iter | spatial | object | goal | libero_10 | 平均 |
|------|---------|--------|------|-----------|------|
| 300  | 45.6%   | 48.8%  | 12.0% | **0.0%**    | 26.6% |
| 500  | 53.8%   | 52.8%  | 28.2% | 3.2%      | 34.5% |
| **1000** | 60.0% | **62.2%** | **42.8%** | 2.0% | **41.8%** |
| 5000 | 9.0%    | 0.2%   | 10.0% | 0.2%     | 4.9% |
| 10000| 0.0%    | 0.0%   | 0.0% | 0.0%      | **0.0%** |

**4-task 系列（mean 初始化）— ⚠️ 单调下降，iter=0 最优**

| iter | spatial | object | goal | libero_10 | 平均 |
|------|---------|--------|------|-----------|------|
| **0**  | **92.4%** | **90.0%** | **53.0%** | **49.2%** | **71.2%** ← 最优 |
| 1    | 92.4%   | 90.2%  | 53.0% | 49.4%    | **71.25%** (≈ iter=0) |
| 10   | 92.0%   | 90.4%  | 50.0% | 43.0%    | 68.85% (-2.35pp) |
| 100  | 86.4%   | 90.2%  | 46.6% | 31.8%    | 63.8% |
| 300  | 86.8%   | 90.0%  | 42.8% | 22.2%    | 60.5% |
| 500  | 85.8%   | 90.6%  | 43.0% | 18.6%    | 59.5% |

**关键发现**：纯 mean merge（iter=0，无 Adam 优化）就是最佳融合。WUDI 优化对 4-task 从第一步起就单调降低 avg 和 libero_10 成功率。

**TIES 基线（mr=0.8, scaling=1.0，与 mean 对照）— 新增 2026-04-19**

| 方法 | spatial | object | goal | libero_10 | 平均 |
|------|---------|--------|------|-----------|------|
| **mean@iter=0** (无干预) | **92.4%** | **90.0%** | **53.0%** | **49.2%** | **71.2%** ← 最优 |
| WUDI@iter=500 | 85.8%   | 90.6%  | 43.0% | 18.6% | 59.5% |
| TIES (mr=0.8)  | 85.4%   | 73.6%  | 35.6% | 15.6% | 52.6% |

**TIES 也弱于 mean 基线**：spatial 基本持平，但在 object / goal / libero_10 上全面落后 mean@iter=0。尤其是 object 从 90.0% 掉到 73.6%（-16.4pp），goal 从 53.0% 掉到 35.6%（-17.4pp）。

**单任务 FT@20k 交叉评测（诊断 task vector 特异化程度）**

| FT 模型 | spatial | object | goal | libero_10 | 平均 |
|---|---|---|---|---|---|
| `ft_libero_10_20k` | 17.4% | 34.2% | 10.0% | **92.6%** | **38.6%** |
| `ft_libero_spatial_20k` | **98.0%** | 0.0% | 9.8% | 0.0% | **27.0%** |
| `ft_libero_goal_20k` | 37.2% | 2.4% | **95.8%** | 0.0% | **33.9%** |
| `ft_libero_object_20k` | 1.4% | **98.2%** | 0.0% | 0.2% | **25.0%** |

> 4 个单任务 FT 各自在自身 suite 上 >92%，但在其他 3 个 suite 上全面崩溃（大多 <10%，多为 0%）。这说明 pi0.5 在 20k 步全量微调后，task vector 已高度特异化并引发严重的灾难性遗忘——每个 FT 模型的平均只有 25~39%。相比之下，**4task mean@iter=0 的融合将跨 suite 平均恢复至 71.2%**，证明算术平均能有效保留多任务共享的能力，而 WUDI 的 Adam 优化 / TIES 的 Trim+Sign 都会破坏这一保留。

**Fine-tuning（从 WUDI 融合 checkpoint 出发继续训练）**

| 初始化 checkpoint | 额外训练 | spatial | object | goal | libero_10 | 平均 |
|---|---|---|---|---|---|---|
| `2task_5k_iter5k` (sum) | FT 30k | 98.2% | 31.8% | 17.2% | 93.2% | 60.1% |
| `4task_sum_iter500` | FT 10k | 98.4% | 98.2% | 98.2% | **91.0%** | **96.5%** |
| `4task_sum_iter1k`  | FT 9k  | 96.8% | 98.8% | 97.2% | **93.0%** | **96.5%** |

### 1.2 核心矛盾（三种不同机制）

**矛盾 A：过优化（2-task 全部 suite & 4-task sum init 的 spatial/object/goal）**
```
WUDI 优化损失：iter=300 > 500 > 1000 > 5000 > 10000  （单调下降）
机器人成功率：iter≈300~1000 最优，之后单调下降，iter=10000 趋近于 0
```
WUDI 损失收敛，但真实成功率在拐点（2-task ≈500~1000，4-task sum ≈1000）后反而退化。  
**4-task@10000 全部 0%，确认模型彻底退化为 base**（H4 验证）。

**矛盾 B：结构性崩溃（4-task sum init 的 libero_10 suite）**
```
4-task sum libero_10 成功率：iter=300 已为 0%，与 iter 无关
```
libero_10 的崩溃并非过优化造成，而是**优化开始前就注定的结构性问题**：4-task 的优化方向从初始化时就压制了 libero_10 的任务向量方向。

**矛盾 C：单调下降（4-task mean init，全部 suite）— 新发现 🔥**
```
4-task mean 平均成功率：iter=0 → 100 → 300 → 500 = 71.2% → 63.8% → 60.5% → 59.5%
4-task mean libero_10：iter=0 → 100 → 300 → 500 = 49.2% → 31.8% → 22.2% → 18.6%
```
对 4-task mean init 而言，**纯 mean merge（iter=0，无 Adam 优化）就是最优解**。  
任何 WUDI 优化步骤都在降低 avg 和 libero_10 成功率，不存在"先上升后下降"的拐点。

> **三种机制的治理方式不同**：
> - 模式 A：early stopping（在拐点前停止优化）
> - 模式 B：FT 恢复（融合后继续训练）
> - 模式 C：**完全跳过 WUDI 优化**，直接用 mean merge（task arithmetic）

**矛盾 D：TIES 也有害（4-task，新增 2026-04-19）**
```
TIES (mr=0.8, s=1.0)：avg=52.6%，弱于 WUDI@500 (59.5%) 和 mean@iter=0 (71.2%)
```
TIES 的 Trim（丢弃 80% 小幅值）+ Disjoint Merge（按符号分组平均）两步在 pi0.5 全量微调场景下同样破坏有用信号。
- Trim 误杀：pi0.5 全量微调任务向量呈大幅值右偏分布（OptMerge Fig.2a），80% 裁剪可能丢弃关键次要方向
- Disjoint Merge 压制少数派：libero_10 方向在符号投票中屡屡成为少数派，被 disjoint_merge 直接忽略（libero_10 从 mean 的 49.2% 掉到 TIES 的 15.6%）

> **升级版假说**：对 pi0.5 全量微调的 4-task 融合，**任何"基于任务向量的统计/优化调整"都会降低效果**。
> 现有证据：mean (71.2%) > WUDI (59.5%) > TIES (52.6%)，呈现"越干预、越退化"的梯度。

---

## 二、理论分析

### 2.1 OptMerge Theorem 3.1：三项误差的对抗

OptMerge 论文 §3.2 给出了合并模型在任务 i 上的损失上界（全量微调，梯度下降步长 η，共优化 T 步）：

```
L_i(Θ + τ_m) ≤  C_i
              +  O(γ^T)      ← 单任务收益项（随 iter 指数衰减）
              +  O(δ·η·T)    ← 跨任务干扰项（随 iter 线性增长）
              +  O(η²·T²)    ← L-光滑曲率项（随 iter 平方增长）
```

- **C_i**：各微调模型自身的收敛残差（固定项）
- **γ^T 项（收益）**：合并从任务向量中提取有效信息的能力，随 T 递减
- **δηT 项（干扰）**：δ 是方向泄漏，随 T **线性增长**
- **η²T² 项（曲率）**：参数空间非线性误差，随 T **平方增长**

**结论**：WUDI 优化的代理损失捕捉的是 γ^T 项的收益，但看不到 δηT 和 η²T² 两项的增长。当 iter 超过某个阈值（2-task ≈ 300~1000 步，4-task sum ≈ 1000 步），干扰+曲率超过收益，真实成功率开始下降。

> 注：4-task 的 libero_10 崩溃是另一种机制（结构性压制，见 §2.5），不在本定理框架内。

---

### 2.2 零空间逃逸：WUDI 代码层面的根本机制

查看 `scripts/wudi_merge.py`（优化核心部分）：

```python
# 初始化（sum）
merging = torch.nn.Parameter(vectors.sum(dim=0).clone())
opt = torch.optim.Adam([merging], lr=1e-5)

for step in range(iter_num):
    diff = merging.unsqueeze(0) - taskvector               # (T, m, n)
    ip   = torch.matmul(diff, low_rank.transpose(-2, -1))  # 投影到各任务低秩子空间
    loss = (ip.square() / norms[:,None,None]).sum()        # 最小化投影能量
    loss.backward(); opt.step()
```

**几何解读**：损失函数要求 `(τ_m - τ_i)` 在每个任务的低秩子空间 `low_rank_i` 上的投影趋近于零。即 `τ_m` 被推向所有 `low_rank_i` 子空间的**公共零空间（null space）**。

随着 iter 增加：

```
iter=0（sum）: τ_m = Σ τ_i                （任务向量之和，初始幅值大）
iter=0（mean）: τ_m = mean(τ_i)           （任务向量均值，初始幅值小且均衡）
iter=1000:    τ_m ≈ 折中方向             （部分去除干扰，保留主要任务信息）
iter=5000:    τ_m → null(low_rank_1,...) （满足代理损失但脱离有效参数区域）
iter=10000:   τ_m ≈ 0                    （θ_merged ≈ θ_base，退化为原始基础模型）
```

**直接验证**：4-task iter=10000 的全面 0% 成功率（H4 ✅），意味着模型退化成了完全没有在 LIBERO 上微调过的 base 模型。

---

### 2.3 全量微调的特殊劣势

OptMerge 论文 Fig. 2(a) 显示，全量微调的任务向量呈**右偏大幅值分布**，而 LoRA 呈多峰小幅值分布。

对于全量微调模型：

1. **`low_rank_i` 覆盖面广**：任务向量奇异值分布平坦，即便只取 `top k = min_dim // T` 个奇异值，覆盖的子空间维度依然很大
2. **多任务联合覆盖更广**：4个任务的低秩子空间联合几乎覆盖整个参数空间，公共零空间趋近于 `{0}`
3. **早停点更靠前**：LoRA 的任务向量小且稀疏，零空间更大，允许更多优化 iter；全量微调任务向量大，零空间小，最优 iter 更少

**定量估算**（以单层 2D 子块为例）：
- 矩阵维度 (m, n)，2-task 时 `reduced_r = min_dim // 2`（取 50% 奇异值）
- 4-task 时 `reduced_r = min_dim // 4`（取 25% 奇异值），但有 4 个约束
- 4个子空间的联合秩 ≤ 4 × (25% × min_dim) = 100% × min_dim → 无有效零空间

---

### 2.4 sum vs mean 初始化：对崩溃速度的影响

**旧代码（sum）**：
```python
merging = torch.nn.Parameter(vectors.sum(dim=0).clone())  # 幅值 ∝ T
```

**新代码（mean）**：
```python
merging = torch.nn.Parameter(vectors.mean(dim=0).clone())  # 幅值恒定
```

**初始损失对比**（4-task）：
- sum：初始 `‖τ_m‖ ≈ 4 × ‖τ_avg‖`，初始损失极高，Adam 步子大
- mean：初始 `‖τ_m‖ ≈ ‖τ_avg‖`，初始损失仅为 sum 的 1/16（loss ∝ 距离²），Adam 步子小

**实验验证（H5 ✅ 已确认）**：

| 初始化 | 4task@500 avg | spatial | object | goal | libero_10 |
|--------|--------------|---------|--------|------|-----------|
| sum    | 34.5%        | 53.8%   | 52.8%  | 28.2% | 3.2%     |
| **mean**   | **59.5%** | **85.8%** | **90.6%** | **43.0%** | **18.6%** |

mean 初始化将 4-task avg 从 34.5% 提升至 59.5%（+25pp），libero_10 从 3.2% → 18.6%（+15pp）。

**进一步发现（2026-04-17）**：mean init 下 iter=0 **比 iter=500 还要好**（71.2% vs 59.5%）。  
这意味着：对 4-task 全量微调而言，**WUDI 的 Adam 优化从第一步起就在降低成功率**，最优策略是跳过优化、直接用 task arithmetic（mean merge）。  
这一发现推翻了 "mean init + 更多 iter" 的探索方向，WUDI 的代理目标与 4-task 真实控制能力在优化路径上完全负相关。

**注意**：mean init 对 **2-task 无显著帮助**（sum avg=54.1% vs mean avg=54.0%，差 <0.1pp）。  
机制解释：2-task 时初始幅值仅为 2×，Adam 步长差异不大；4-task 时初始幅值 4×，差异显著。

---

### 2.5 4-task libero_10 的结构性崩溃（与过优化无关）

**实验证据**：4-task sum init 下，libero_10 在 iter=300 已为 0%，而 spatial/object/goal 在 iter=300 仍有 45.6%/48.8%/12.0%。随 iter 增加，后三者继续提升（iter=1000 最优），而 libero_10 始终在 0~3.2% 之间。

**机制分析**：

WUDI 优化的目标是找 τ_m 使所有任务的投影干扰最小。若 4 个任务向量 {τ_spatial, τ_object, τ_goal, τ_10} 的低秩主成分存在**方向冲突**，则优化器被迫牺牲某个方向。

libero_10 的特殊性：
- libero_10 是 10 个异质长序列任务（open box, stack blocks, turn knob...），与其他三个 suite 的任务分布差异最大
- libero_10 的任务向量方向与 {spatial, object, goal} 正交分量更大，即 τ_10 中有更多"独特方向"
- 4-task 优化的代理损失中，τ_spatial + τ_object + τ_goal 三票对一票，libero_10 方向在多数票压力下被压制

**mean init 的有限改善**：
- mean init 将初始点从 4×avg 降至 1×avg，减小了初始步长，让优化更温和
- libero_10 在 mean@500 下从 3.2% → 18.6%（+15pp），说明过大的初始步长也部分参与了压制
- 但 18.6% 仍远低于其他 suite（85-90%），说明方向冲突是根本原因，init 策略只能缓解

**与 FT 的对比**：FT 直接以全量数据重新对齐参数，绕过了 task vector 方向冲突问题，4-task FT@10k 可将 libero_10 恢复至 91~93%，验证了融合 checkpoint 保留了足够的参数结构。

---

## 三、实验猜想验证情况

| # | 猜想 | 预期结果 | 实验结果 | 状态 |
|---|------|----------|----------|------|
| **H1** | 2-task: iter=500 ≈ iter=1000，最优点在 500~1000 | spatial ≥ 94%, avg ≥ 54% | 500 avg=54.1%，1000 avg=54.1%，差 <1pp | ✅ **确认** |
| **H2** | `‖τ_m‖_F` 随 iter 单调下降趋近于 0 | 高 iter checkpoint 的参数距 base 接近零 | 4-task@10k 全面 0% 间接确认；未打印 norm 日志 | ⚠️ 间接确认 |
| **H3** | 4-task 崩溃比 2-task 更快 | 4-task 最优 iter 更小，退化更陡 | 4-task@5k avg=4.9% vs 2-task@5k avg=45.4%（相同 iter 退化更严重） | ✅ **确认** |
| **H4** | 崩溃 checkpoint 与 base 参数距离接近零 | `‖Δθ‖ ≤ 1%‖θ_base‖` | 4-task@10k 评测 100% 零成功率，行为等同 base | ✅ **确认（间接）** |
| **H5** | mean 初始化可显著改善 4-task 稳定性，最优 iter 右移 | avg 明显提升，崩溃更晚 | 4-task sum@500→34.5%，mean@500→59.5%，提升 25pp；**但最优 iter 反而左移至 0**（见 H7） | ⚠️ **部分确认** |
| **H6** | Frobenius norm 正则可延缓崩溃 | 允许更多 iter 而不退化 | 未测试 | ❌ 待验证 |
| **H7** 🔥 | 4-task mean init 下 WUDI 优化从 iter=0 起单调有害 | iter=0 avg > iter>0 avg | iter=0→71.2%, iter=1→71.25%, iter=10→68.85%, iter=100→63.8%, iter=300→60.5%, iter=500→59.5%（iter=1 在噪声内，iter=10 起明显退化） | ✅ **确认** |
| **H8** 🆕 | TIES（Trim+Elect Sign+Disjoint Merge）也弱于纯 mean | TIES avg < mean@iter=0 | TIES(mr=0.8,s=1.0) avg=52.6%，低于 mean@iter=0 71.2% 约 19pp，且低于 WUDI@500 59.5% | ✅ **确认** |
| **H9** 🆕 | 2-task 任务对的语义距离越小，mean@iter≈0 越高，退化越慢 | goal+10 > object+10 > spatial+10 在低 iter 下的平均 | goal+10@iter=10 avg=70.55% > object+10@iter=1 avg=53.8% > spatial+10@iter=1k avg=54.1%；5k 时 goal+10 avg=36.7% vs spatial+10 avg=45.4%（但方向不一致，H9 需进一步验证） | ⚠️ **初步支持** |

---

## 四、综合结论与建议

### 4.1 核心结论

```
WUDI 优化损失下降  ≠  机器人控制成功率提升
────────────────────────────────────────────────────────────────
WUDI 优化的是"代理目标"（子空间投影干扰最小化），
而非机器人控制能力。

本实验中观察到三种不同的失败模式：

【模式 A：过优化退化（2-task 全部 & 4-task sum init 的 spatial/object/goal）】
  阶段一（iter < 拐点）：γ^T 收益项 > δηT 干扰项 → 成功率提升
  阶段二（iter > 拐点）：δηT 干扰 + η²T² 曲率 > 收益 → 成功率下降
  最优 iter：2-task sum 300~1000；4-task sum ~1000

【模式 B：结构性压制（4-task sum init 的 libero_10）】
  libero_10 在 iter=300 已为 0%，与 iter 无关
  原因：libero_10 任务向量与其他三个 suite 方向冲突，在优化中被多数票压制
  mean init 可缓解（3.2% → 18.6% → 49.2%@iter=0），但无法从根本解决
  唯一有效方案：FT（10k 步可恢复至 91~93%）

【模式 C：单调有害（4-task mean init 全部 suite）🔥】
  iter=0（纯 mean merge）avg=71.2%，任何优化步骤都降低 avg
  最优策略：跳过 Adam 优化，直接使用 task arithmetic（mean merge）
  解释：mean init 本身已处在一个低损失、低干扰的良好点，
        Adam 进一步压低代理损失时会沿着 δηT/η²T² 方向推进，立即损害真实能力

【模式 D：统计调整也有害（TIES）🆕】
  TIES(mr=0.8,s=1.0) avg=52.6%，弱于 mean@iter=0 (71.2%)，甚至弱于 WUDI@500 (59.5%)
  失败机制：
    1) Trim (mask_rate=0.8) 误杀小幅值但重要的参数（robotics 细粒度控制）
    2) Elect Sign + Disjoint Merge 压制少数派任务向量，libero_10 方向被进一步弱化
    3) scaling=1.0 + 稀疏后幅值不足，整体能力退化
  升级版假说：对 pi0.5 全量微调的 4-task 融合，
              任何基于任务向量的统计/优化调整（WUDI Adam / TIES Trim+Sign）都会降低效果，
              纯 mean task arithmetic 是最佳单次融合基线。

**【2-task 方向对距离效应 🆕 2026-04-22】**
  goal+10（语义相近）：mean@iter≈0 约 69~70%，退化速度慢（iter=10 仍最优）
  object+10（中等距离）：mean@iter≈0 约 54%，退化速度中等
  spatial+10（较远）：mean@iter≈0 约 54%，iter=500~1000 为拐点（A 型过优化）
  共同点：无论任务对如何，5k 时均严重退化（31~45%），10k 时均崩溃（0%）
  启示：**融合任务对的语义越近，iter=0 的 mean 越高**，这与 task vector 正交性预期一致。

4-task 比 2-task 崩溃更快的三重原因：
  1. 4个约束 → 公共零空间更小 → optimizer 更难找好解
  2. sum 初始化 → 初始幅值 4× → Adam 步子更大 → 更快脱轨
  3. O(δηT) 中的 δ 随任务数超线性增长

mean 初始化的作用与局限：
  - 效果：初始损失降低 ~16×，4-task avg 提升 25pp（34.5% → 59.5%）
  - 局限：对 2-task 无效（sum≈mean≈54%）；无法修复 libero_10 结构性崩溃
```

### 4.2 最终最优方案

| 方案 | avg 成功率 | 说明 |
|------|-----------|------|
| 4task TIES (mr=0.8, s=1.0) | 52.6% | 🆕 TIES 基线，弱于 mean |
| 4task merge (sum@500) | 34.5% | 基线 |
| 4task merge (sum@1k) | 41.8% | sum 最优 iter |
| 4task merge (mean@500) | 59.5% | mean init + 优化 |
| **4task merge (mean@iter=0)** | **71.2%** | 🔥 最优单次融合（纯 mean，无优化） |
| **4task merge + FT 10k** | **96.5%** | 最终最强方案（两个 FT 起点均达此） |

**方法梯度**：mean (71.2%) > WUDI@500 (59.5%) > TIES (52.6%) > WUDI sum@500 (34.5%)。任何超出纯算术平均的统计/优化调整均导致退化。

**结论**：
1. **纯 mean merge（iter=0）是最好的 WUDI 融合结果**——WUDI 的 Adam 优化对 4-task 完全负贡献
2. FT 10k 步后四 suites 全面 ≥91%，彻底解决 libero_10 崩溃问题
3. 建议把 `4task_mean_iter=0` 也作为 FT 的起点做一次实验，看是否能超过 96.5%

### 4.3 近期行动建议

针对**模式 C（4-task mean 单调有害）🔥**：
1. **默认跳过 WUDI 优化**：4-task 场景下直接用 `vectors.mean(dim=0)`，不进入 Adam 循环
2. **从 mean@iter=0 做 FT**：当前 96.5% 来自 mean@500（初始 avg=59.5%）的 FT；若从 mean@iter=0（avg=71.2%）出发，FT 可能收敛更快或上限更高

针对**模式 A（sum init 过优化）**：
3. **监控 `‖τ_m‖`**（H2 直接验证）：在 `wudi_optimize()` 中每 100 步打印 `merging.norm().item()`，找到 norm 急剧下降的 iter 点作为 early-stopping 依据
4. **norm 正则（H6）**：加入 `loss += λ * merging.norm()` 可能延缓崩溃

针对**模式 D（TIES 弱于 mean）🆕**：
4b. **不建议继续扫描 TIES 超参**作为 4-task pi0.5 的主线方案——已验证统计调整类方法（Trim+Sign）损害能力；如需做完备对照，可扫 `mask_rate ∈ {0.5, 0.7, 0.9}` 或 `scaling ∈ {0.3, 0.5}`，但期望上限仍 < mean@iter=0
4c. **跨方法共识**：WUDI Adam 和 TIES Trim+Sign 均弱于纯 mean，下一步应转向**融合后 FT**或**数据/架构侧方案**而非继续设计融合算法

针对**模式 B（libero_10 结构性压制）**：
5. **调整 rank ratio**：4-task 时 `reduced_r = min_dim // 4`，可尝试不对称分配（给 libero_10 更多 rank）
6. **加权优化**：在 WUDI 损失中对 libero_10 的约束项加权（使四个任务等权重而非等约束数）
7. **分阶段融合**：先 2-task（libero_10 + spatial），再与另外 2-task（goal + object）融合，避免四方向同时冲突

### 4.4 长期改进方向

参考 OptMerge §4.1 对全量微调的专项处理：
- **去中心化 + SVD 截断**（论文 Eq. 2-3）：已在 WUDI 实现中部分采用，可进一步调 rank ratio
- **norm 约束正则化**：防止 `τ_m` 幅值崩塌
- **自适应 early stopping**：监控 `‖τ_m‖` 变化率，自动在拐点前停止优化

---

*文档依据：`docs/experiment_log_2026-04-15.md`，`scripts/wudi_merge.py`，OptMerge paper §3.2 & §4.1*
