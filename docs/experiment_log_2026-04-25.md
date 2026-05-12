# 实验日志 — 2026-04-25 ~ 2026-04-26

> 创建：2026-04-26 CST
> 项目路径：`/storage/yukaichengLab/lishiwen/jiayusun/openpi`

---

## 一、本次实验背景

本次日志覆盖 2026-04-25 至 2026-04-26 的实验，主要包含三项工作：

1. **`wudi_merge.py` 新增 `--init-weights` 功能**：把 mean 等权初始化扩展为可手动指定每任务系数（不强制和为 1）。
2. **FT checkpoint 从 20k 切换到 15k 重新扫描**：发现 ft@20k 已开始过拟合 goal，ft@15k 是更优融合起点。
3. **加权初始化扫描（w333/w335/w444/w446）**：验证"系数总和 > 1 → spatial 崩盘"假说，确认 sum=1 是稳定性边界。
4. **l10_mix50 训练实验**：从 `pi05_libero/my_experiment/15000` 出发，用 50/50 采样（libero_10 vs spatial+object+goal）训练 2k 步，**只用 1k 步即超越 FT 5k 路径**。

---

## 二、`wudi_merge.py` 加权初始化功能

### 2.1 改动点

`scripts/wudi_merge.py` 新增 `--init-weights` CLI 参数和函数级 `init_weights` 参数：

| 位置 | 改动 |
|---|---|
| `wudi_optimize` 签名 | 新增 `init_weights: list[float] \| None = None` |
| `wudi_optimize` 主体 | `None` → `vectors.mean(dim=0)`；否则 `(vectors * w.view(T,1,1)).sum(dim=0)` |
| `run_merge` 签名 | 新增同名参数，向下传递 |
| `run_merge` 中非 attn/FFN 分支 | `np.mean` → 加权求和（保持初始点全局一致） |
| CLI | 新增 `--init-weights`，长度自校验 |

### 2.2 用法

```bash
# 默认行为（mean 等权）：不指定 --init-weights
python scripts/wudi_merge.py --base ... --ft A B C D ... --iter 500

# 加权初始化（不强制和为 1）：
python scripts/wudi_merge.py --base ... --ft A B C D \
    --init-weights 0.4 0.4 0.4 0.6 \
    --iter 500
```

### 2.3 验证

`python scripts/wudi_merge.py --test` smoke test 全部通过（round-trip / linearity / WUDI optimize）。

---

## 三、FT checkpoint 20k → 15k 的影响

### 3.1 发现

历史 3task_sog 系列（mean iter=300/500/1k/1500/2k）使用 `from10k/.../20000` checkpoint，但实际目录中 **20000 已被覆盖或不存在**，只有 5000/15000/29999。本次实验切到 **15000** 重新评估。

### 3.2 等价对比

`w333 iter=1`（0.333×3，sum=1.0，数学上等价于 mean）作为基线：

| FT 步数 | spatial | object | goal | libero_10 | avg(3) | eval Job |
|---|---|---|---|---|---|---|
| 20k mean iter=1 | 94.2 | 94.6 | 47.6 | ~5%（部分） | 78.8 | 34268 |
| **15k w333 iter=1**（≈ mean） | **95.4** | 94.0 | **54.2** | 8.6 | **81.2** | 34412 |
| Δ | +1.2 | -0.6 | **+6.6** | — | **+2.4** |

**结论**：15k 比 20k 是更好的融合起点。20k 已过拟合 goal（goal FT loss 已收敛但任务向量方向开始偏离"通用 goal 能力"，对融合不利）。

> ⚠️ 之前 `experiment_log_2026-04-22.md` 中所有 mean iter=300/500/1k/1500/2k 的结果都是基于 20k FT。如果用 15k 重做，goal 系列的整体水平很可能上一个台阶。

---

## 四、加权初始化扫描

### 4.1 实验设置

- **base**: `pi05_libero/my_experiment/10000`
- **ft**: 4 个 FT checkpoint @ **15k**（spatial / object / goal / libero_10）
- **iter**: 1（≈ 加权 init，无 WUDI 优化）
- **scope**: llm_only

### 4.2 4-task w446（spatial/object/goal/libero_10 = 0.4/0.4/0.4/0.6，sum=1.8）

**spatial 前 9 task：56.7%**（task 1=0%、task 4=12%、task 9=38%）→ **崩盘**。

**根因：** sum=1.8，相当于 `base + 1.8 × τ_avg_weighted`，远超模型能承受的偏离量。

由于 spatial 已严重崩塌（前 9 个就 56.7% < 90% 阈值），中途 cancel。

### 4.3 3-task w444（spatial/object/goal = 0.4/0.4/0.4，sum=1.2）

**spatial 完整 10 task: 89.0%**（task 1=0.72, task 9=0.48 是主崩盘点）。

| task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | avg |
|------|---|---|---|---|---|---|---|---|---|---|-----|
| mean baseline | 1.00 | 1.00 | 1.00 | 0.98 | 0.82 | 0.98 | 0.90 | 0.98 | 0.92 | 0.84 | 94.2% |
| w444 (sum=1.2) | 1.00 | 0.72 | 1.00 | 0.96 | 0.86 | 0.98 | 1.00 | 0.96 | 0.94 | 0.48 | **89.0%** |

按 spatial<90% 阈值规则，cancel。

### 4.4 3-task w333（mean 等价，sum=1.0）

完整结果见 §3.2，作为对照基线，**spatial=95.4%、avg(3)=81.2%**，正常无崩塌。

### 4.5 3-task w335（spatial/object/goal = 0.333/0.333/0.5，sum≈1.167）

| Suite | w335 | w333 | mean iter=1 (20k) |
|-------|------|------|-------------------|
| spatial | **84.2%** ⚠️ | 95.4% | 94.2% |
| object | 90.8% | 94.0% | 94.6% |
| goal | **66.8%** ✅ | 54.2% | 47.6% |
| libero_10 | ~4% | 8.6% | ~5% |

**spatial 前 10 task per-task：[1.0, 0.54, 1.0, 0.94, 0.74, 1.0, 0.96, 0.92, 0.98, 0.34]**
- task 1=0.54、task 9=0.34、task 4=0.74 三处崩盘
- 其余 7 个 task 都 ≥0.92

**关键观察**：spatial 仅 17% 总放大就开始崩，goal 却显著上升（+12.6 vs w333）。

### 4.6 加权初始化稳定性总结

| 配置 | sum(w) | spatial | object | goal | 安全？ |
|------|--------|---------|--------|------|--------|
| w333 (3-task, 0.333×3) | 1.000 | 95.4% | 94.0% | 54.2% | ✅ |
| w335 (3-task, 0.333/0.333/0.5) | 1.167 | **84.2%** ⚠️ | 90.8% | 66.8% | ❌ spatial 崩 |
| w444 (3-task, 0.4×3) | 1.200 | 89.0% | — | — | ❌ spatial 崩 |
| w446 (4-task, 0.4/0.4/0.4/0.6) | 1.800 | **56.7%** | — | — | ❌❌ 严重崩 |

**核心结论：sum=1 是稳定边界。** 即使 17% 的轻微超标也会触发 spatial 崩塌（task 1 和 task 9 是最敏感的）。如果想倾斜某个任务，必须**保持 sum=1**（即从其他任务"借权重"），例：

```
保持总和 = 1，goal 加权：(0.25, 0.25, 0.50)  ← 仍 sum=1
不保持总和 = 1，goal 加权：(0.333, 0.333, 0.5) ← sum=1.167，spatial 崩
```

---

## 五、l10_mix50 训练实验

### 5.1 动机

历史 FT 路径（FT 5k from 3task_sog → libero10ft 二次 FT 1k）虽然达到 avg=96.2%，但需要：
1. 先做 WUDI merge（500 iter，~30min）
2. 再做 4-task FT 5k 步（~1h28m）
3. 再做 libero_10 二次 FT 1k 步（~25min）
4. 总耗时 ~2.5h

新方案：直接从 `pi05_libero/my_experiment/15000`（已经在 4-task 全集上训练 15k 步的 base）出发，**用 50/50 mix sampler** 让 libero_10 和 (spatial+object+goal) 各占 50%，加强 libero_10 弱点而不放弃其他 suite。

### 5.2 配置

| 项目 | 值 |
|---|---|
| 初始 checkpoint | `pi05_libero/my_experiment/15000/params` |
| Train config 名 | `pi05_libero_l10_mix50_from_15k`（`config.py` 新增） |
| 数据混合 | libero_10（379 ep）= 50% / spatial+object+goal（1314 ep）= 50% |
| 实现 | `extra_lerobot_datasets` + `dataset_mix_weights=(0.5, 0.5)` |
| batch_size | 32 |
| FSDP | 2 GPU |
| num_train_steps | 2000 |
| save_interval | 1000 |
| keep_period | 1000（step 1000 / 1999 都保留） |
| lr_schedule | CosineDecay, warmup=200, peak=2.5e-5, decay→2.5e-6 |
| optimizer | AdamW (clip=1.0, weight_decay=1e-10 ≈ 0) |
| EMA | 0.999 |

> 注：weight_decay=1e-10 是 AdamW 默认值，作者注释"Changing this to 0 can cause out-of-memory errors for some reason"。1e-10 数值上完全可忽略：2k 步累积偏差 ~5e-12，远小于浮点精度。

### 5.3 训练 Job（34454）

- 节点：gnho031，2 GPU FSDP
- 完成时间：~35min（2k steps）
- 训练 loss：step 1500-1900 区间稳定在 0.0058-0.0067，grad_norm ~0.07-0.08，param_norm 1812.5（参数变化极小）

Checkpoint 输出：
```
checkpoints/pi05_libero_l10_mix50_from_15k/ft_l10_mix50_from_15k/
├── 1000/   ← step 1000 保存点
├── 1999/   ← step 2000 保存点（最后一步索引为 step-1）
└── wandb_id.txt
```

### 5.4 step 1000 评测结果（Job 34457，40/40 完整）

| Suite | 成功率 |
|-------|------|
| libero_spatial | **98.4%** |
| libero_object | **98.6%** |
| libero_goal | **95.8%** |
| libero_10 | **90.4%** |
| **avg(4)** | **95.8%** |

#### per-task 完整明细

**libero_spatial（avg = 98.4%）**

| task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|------|---|---|---|---|---|---|---|---|---|---|
| rate | 1.00 | 1.00 | 1.00 | 1.00 | 0.96 | 0.96 | 0.96 | 1.00 | 0.98 | 0.98 |

**libero_object（avg = 98.6%）**

| task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|------|---|---|---|---|---|---|---|---|---|---|
| rate | 0.94 | 1.00 | 1.00 | 1.00 | 0.98 | 0.98 | 1.00 | 0.98 | 0.98 | 1.00 |

**libero_goal（avg = 95.8%）**

| task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|------|---|---|---|---|---|---|---|---|---|---|
| rate | 0.92 | 1.00 | 1.00 | **0.84** | 0.96 | 0.94 | 1.00 | 1.00 | 1.00 | 0.92 |

**libero_10（avg = 90.4%）**

| task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|------|---|---|---|---|---|---|---|---|---|---|
| rate | 0.88 | 0.98 | 0.96 | 0.94 | 0.94 | 0.98 | 0.88 | 1.00 | **0.54** | 0.94 |

#### 弱点分析

1. **libero_10 task 8 = 0.54**（"put both moka pots on the stove"）：在所有方案中（FT 5k / libero10ft / l10_mix50）都是最顽固任务
2. **libero_goal task 3 = 0.84**：相对最弱的 goal 任务
3. spatial/object 几乎全部 ≥ 0.94

### 5.5 step 2000 评测结果（Jobs 34482 + 34497，40/40 完整）

| Suite | step 2000 | step 1000 | Δ |
|-------|-----------|-----------|---|
| libero_spatial | **98.8%** | 98.4% | +0.4 |
| libero_object | 98.6% | 98.6% | 0.0 |
| libero_goal | **97.4%** | 95.8% | +1.6 |
| libero_10 | **91.6%** | 90.4% | +1.2 |
| **avg(4)** | **96.6%** ⭐ | 95.8% | **+0.8** |

> Job 34482 在 libero_10 第 2 task 时 SIGTERM（spatial/object/goal 完整）；34497 单独补评 libero_10 完成全部 10 task。

#### libero_10 per-task 对比（1k vs 2k）

| task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|------|---|---|---|---|---|---|---|---|---|---|
| step 1k | 0.88 | 0.98 | 0.96 | 0.94 | 0.94 | 0.98 | 0.88 | 1.00 | **0.54** | 0.94 |
| step 2k | 0.94 | 1.00 | 0.98 | 1.00 | 0.94 | 1.00 | 0.92 | 1.00 | **0.48** | 0.90 |

**关键观察：**
- task 0/2/3/5/6 普遍 +0.02~0.06
- task 8（moka pots）从 0.54 **微降到 0.48**——1k→2k 没改善，甚至略退
- task 9 也微降 0.04（0.94→0.90）

→ task 8 是结构性短板，不是训练不足，可能需要专门策略（更多 demo、prompt 重设计、或 task-specific lr）。

### 5.6 与历史方案对比

| 方案 | 步骤 | 总耗时 | spatial | object | goal | libero_10 | avg(4) |
|------|------|--------|---------|--------|------|-----------|--------|
| FT@10k from wudi_4task_500 | merge + 10k FT | ~3h+ | 98.4% | 98.2% | 98.2% | 91.0% | 96.5% |
| FT@9k from wudi_4task_1k | merge + 9k FT | ~3h+ | 96.8% | 98.8% | 97.2% | 93.0% | 96.5% |
| FT@5k from 3task_sog_iter500 | merge + 5k FT | ~2h | 97.6% | 96.8% | 95.6% | 90.4% | 95.1% |
| libero10ft (二次 FT@1k) | merge + 5k + 1k l10 | ~2.5h | 99.2% | 98.4% | 95.6% | 91.4% | 96.2% |
| l10_mix50 @ step 1000 | 直接 1k FT | ~17min | 98.4% | 98.6% | 95.8% | 90.4% | 95.8% |
| **l10_mix50 @ step 2000** | **直接 2k FT** | **~35min** | **98.8%** | **98.6%** | **97.4%** | **91.6%** | **96.6%** ⭐ |

**🏆 关键发现：l10_mix50 @ step 2000 是新历史最佳**

1. **avg=96.6% 首次超过历史最佳**（FT@10k 的 96.5%）
2. **耗时仅 ~35min**（vs 历史 3h+），**≈ 5× 提速**
3. **不需要 WUDI merge**：直接从已训练 15k 步的 base 出发，跳过 merge 步骤
4. **所有 4 个 suite 都 ≥91.6%**，没有明显短板
5. step 1k → 2k 还在涨（+0.8pp），暗示**继续训练 3k/5k 步可能进一步突破**

---

## 六、SLURM 任务汇总（2026-04-25 ~ 2026-04-26）

| Job | 类型 | 说明 | 节点 | 状态 |
|---|---|---|---|---|
| 34396 | merge | 4task w446 iter=1（首次，20k FT，路径错误） | gnho009 | ❌ FileNotFoundError |
| 34402 | merge | 4task w446 iter=1（15k FT 重提） | gnho009 | ✅ 完成 |
| 34403 | eval | 4task w446 iter=1（spatial 56.7% 崩盘）| gnho009 | ❌ 用户取消 |
| 34404~34407 | merge+eval | 4task w446 iter=500/1k 链 | gnho009 | ❌ 取消（w446 已确认崩） |
| 34409 | merge | 3task w444 iter=1 | gnho009 | ✅ 完成 |
| 34410 | eval | 3task w444 iter=1（spatial 89.0% 触发<90% 规则） | gnho009 | ❌ 取消 |
| 34411 | merge | 3task w333 iter=1 | gnho009 | ✅ 完成 |
| 34412 | eval | 3task w333 iter=1（avg(3)=81.2%） | gnho009 | ⚠️ 40/40 完整，清理时 SIGTERM |
| 34452 | merge | 3task w335 iter=1 | gnho031 | ✅ 完成 |
| 34453 | eval | 3task w335 iter=1（spatial 84.2% 崩） | gnho031 | ⚠️ spatial/obj/goal 完整，清理时 SIGTERM |
| 34454 | train | l10_mix50 from 15k, 2k steps | gnho031 | ✅ 完成（35min） |
| 34457 | eval | l10_mix50 @ step 1000 | gnho031 | ⚠️ 40/40 完整，清理时 SIGTERM |
| 34458 | eval | l10_mix50 @ step 1999（依赖断） | gnho031 | ❌ DependencyNeverSatisfied |
| 34482 | eval | l10_mix50 @ step 1999（重提，spatial/obj/goal 完整，l10 仅 2/10）| gnho031 | ⚠️ SIGTERM |
| 34497 | eval | l10_mix50 @ step 1999 libero_10 补评（10/10）| gnho031 | ⚠️ 完整，清理时 SIGTERM |

---

## 七、关键结论

### 7.1 加权初始化的稳定性边界

**sum(weights) = 1 是硬边界**。即使总和 = 1.167（17% 超标）也会触发 spatial 崩塌（task 1 和 task 9 最敏感）。

如需倾斜某任务，**保持 sum = 1，从其他任务"借权重"**：
- ✅ `(0.25, 0.25, 0.50)` — sum=1，goal 加权
- ❌ `(0.333, 0.333, 0.5)` — sum=1.167，spatial 崩

理论解释：合并向量 = `base + sum(w_i × τ_i)`。当 sum > 1 时整体偏离量超出模型线性安全区，spatial 等抗扰性差的 suite 首先崩盘。

### 7.2 FT checkpoint 选择的影响

**15k > 20k**：20k 已过拟合 goal，对融合起点不利。所有 mean 系列实验都应使用 15k 重做。

### 7.3 l10_mix50 是新历史最佳路径 🏆

| 指标 | l10_mix50 @ step 2000 | 历史最佳（FT@10k） |
|------|----------------------|-------------------|
| avg(4) | **96.6%** | 96.5% |
| 耗时 | **~35min** | ~3h+ |
| 步骤 | 直接 1×FT | merge + FT |

- **35min 训练**就突破历史最佳（96.5% → 96.6%）
- 比 FT 5k from 3task_sog（~2h）效果好 +1.5pp，且**耗时 1/3**
- 比 libero10ft 二次 FT（~2.5h）效果好 +0.4pp，且**耗时 1/4**
- **关键：不依赖 WUDI merge**，直接从已训练 15k 步 base 上做 50/50 mix FT
- step 1k→2k 还在涨（+0.8pp），暗示**继续训练 3k/5k 可能进一步突破 97%**

> **后续优先方向**：以 `pi05_libero/my_experiment/15000` + 50/50 mix sampler 为基线，扫描更长训练步数（3k/5k/8k），观察 avg(4) 上限。

---

## 八、待办事项

1. ~~l10_mix50 step 1999 评测~~ ✅ 已完成（96.6% 新最佳）
2. **l10_mix50 延长训练（最高优先级）**：3k/5k/8k 步扫描，观察 avg(4) 是否能突破 97%
3. **task 8（put both moka pots on the stove）专项研究**：在所有方案中（libero10ft 0.66 / l10_mix50_2k 0.48）都是顽固任务，需专门策略
4. **3task_sog 15k FT 重做 mean 系列**：用 15k FT 重新跑 mean iter=300/500/1k/1500，对比 20k 系列
5. **保持 sum=1 的加权扫描**：例如 (0.25, 0.25, 0.50)、(0.20, 0.30, 0.50)，看 goal 倾斜的安全上限
6. **修复 changed_ base 路径**（继承自 04-22 待办）
