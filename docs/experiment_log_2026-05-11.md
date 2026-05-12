# 实验日志 — 2026-05-11 PPO × SFT 权重融合 (Task Arithmetic)

> 创建：2026-05-11 CST
> 项目路径：`/storage/yukaichengLab/lishiwen/jiayusun/openpi`
> 关联前序：`docs/experiment_log_2026-04-30_rl.md`（PPO RL 训练）
> 相关历史：`docs/experiment_log_2026-04-29.md`（t89 = libero_10 task 8+9 SFT 微调；l10_mix50 + t89 加权融合）

---

## 一、背景与动机

PPO 训练（Job 34325 / 35279，从 base_25k 在 libero_10 上 RL 微调）已产出 step80 / step100 / step120 三个 checkpoint。step80 单点 4-suite 评测显示：

| Suite | step80 单独 |
|-------|------------|
| libero_spatial | 98.2% |
| libero_object | 99.4% |
| libero_goal | **92.8%** ↓ |
| libero_10 (训练目标 suite) | 97.2% |

**问题**：libero_goal 比 base_25k 下降明显（base 在该 suite ~99%）。RL 只在 libero_10 上训练，对其它 suite 出现负迁移。

**目标**：把 SFT 知识与 RL 收益按比例混合，控制 RL 信号注入强度，验证能否在不损伤其它 suite 的前提下保留 RL 在 libero_10 上的提升。

**核心公式**：
```
θ* = θ_SFT + α · (θ_RL - θ_SFT)
   = θ_SFT + α · Δ_RL
```
α 控制 RL 变化的注入比例：α=0 退回 SFT；α=1 等价 RL 单独。

---

## 二、方法实现

### 2.1 算法选择

不使用 WUDI/SVD（N=1 时退化为线性插值），直接采用 **Task Arithmetic** 闭式解。函数式风格对齐 `MLLMerging/InternVL/internvl_chat/model_merging.py` 中的 `TaskVector` + `task_arithmetic`，但操作对象改为 PyTorch `state_dict` 而非 `nn.Module`。

### 2.2 实测确定融合范围

通过张量级 diff 确认 PPO 实际改变的参数：

| 张量 diff (base_25k vs PPO step80) | 数量 |
|------------------------------------|------|
| 改变 (changed) | 208 / 812 |
| 未变 (unchanged) | 604 / 812 |

**改变的 5 类前缀**（与 RLinf `train_expert_only=True` 配置完全一致）：

| 前缀 | 数量 | 含义 |
|------|------|------|
| `paligemma_with_expert.gemma_expert.model.layers.*` | 198 | action expert 22 层 transformer |
| `paligemma_with_expert.gemma_expert.model.norm.*` | 2 | action expert 最终 RMSNorm |
| `action_in_proj.{weight,bias}` | 2 | 动作输入投影 |
| `action_out_proj.{weight,bias}` | 2 | 动作输出投影 |
| `time_mlp_in / out.{weight,bias}` | 4 | 时步 MLP |

**未改变**：全部 604 个 `paligemma_with_expert.paligemma.*`（vision + language backbone，RL 中冻结）。

### 2.3 显式白名单定义

在 `scripts/task_arithmetic_pt.py:32-49` 显式列出可融合前缀：

```python
PI05_PPO_TRAINABLE_PREFIXES = (
    "paligemma_with_expert.gemma_expert.model.layers.",
    "paligemma_with_expert.gemma_expert.model.norm.",
    "action_in_proj.",
    "action_out_proj.",
    "time_mlp_in.",
    "time_mlp_out.",
)
```

CLI 通过 `--include-preset {pi05_ppo, all}` 控制。默认 `pi05_ppo` 仅对 208 个 trainable key 计算 task vector，其余直接拷贝 base 值。

### 2.4 权重格式对齐

| 项目 | base_25k | PPO step80 |
|------|----------|-----------|
| 文件 | `model.safetensors` (7.23GB) | `model.safetensors` (7.47GB) |
| 张量数 | 812 | 812 |
| key 名 | 完全相同 | 完全相同 |
| shape | 完全相同 | 完全相同 |
| dtype | 全 bf16 | **122 张量 fp32**（layernorm + action_proj + gemma norm/embed）|
| `config.json` | 有 `precision: bfloat16` | 无 |

**脚本处理**：
- `TaskVector.__init__`：差值用 fp32 计算（避免 bf16 舍入累积）
- `combine_with_pretrained_model`：写回时 cast 到 base 的 dtype
- 保存时 `--cast-dtype bfloat16` 统一为 bf16，复制 base 的 `config.json` / `assets/` / `physical-intelligence/`

输出 merged checkpoint 与 base_25k 格式完全一致，可直接被 `serve_policy.py --policy.config pi05_libero` 加载。

---

## 三、文件清单

| 文件 | 作用 |
|------|------|
| `scripts/task_arithmetic_pt.py` | 融合主脚本（TaskVector + task_arithmetic） |
| `merge_base25k_ppo80_alphas.sbatch` | 一次扫 α=0.3/0.5/0.7/1.0 |
| `eval_merged_base25k_ppo80_a03.sbatch` | α=0.3 评测 |
| `eval_merged_base25k_ppo80_a05.sbatch` | α=0.5 评测 |
| `eval_merged_base25k_ppo80_a07.sbatch` | α=0.7 评测 |

输出：`checkpoints/merged/base25k_ppo80_a{03,05,07,10}/`

---

## 四、执行记录

### 4.1 融合（Job 35319，gnho031，0 GPU，~1 min）

```bash
cd /storage/yukaichengLab/lishiwen/jiayusun/openpi
sbatch merge_base25k_ppo80_alphas.sbatch
```

| α | 实际改变张量数 | 输出目录 |
|---|----------------|----------|
| 0.3 | 208/812 | `checkpoints/merged/base25k_ppo80_a03` |
| 0.5 | 208/812 | `checkpoints/merged/base25k_ppo80_a05` |
| 0.7 | 208/812 | `checkpoints/merged/base25k_ppo80_a07` |
| 1.0 | 208/812 | `checkpoints/merged/base25k_ppo80_a10` |

全部状态 COMPLETED，每个 alpha ~10 秒（含 I/O）。

### 4.2 评测

| Job | α | 节点 | Suite | 状态 | 结果 |
|-----|---|------|-------|------|------|
| 35321 | 0.7 | gnho031 | 4 suites | FAILED (libero_10 中 SIGTERM) | spatial 98.0 / object 98.6 / goal 96.6 / libero_10 部分 (5 task 98.4%) |
| 35355 | 0.7 | gnho031 | libero_10 only | FAILED (4 task 后 SIGTERM, 99.5%) | — |
| 35365 | 0.7 | gnho031 | libero_10 only | **COMPLETED (55 min)** | **libero_10 = 96.2%** |
| —     | 0.5 | (待提) | — | — | — |
| —     | 0.3 | (待提) | — | — | — |

α=1.0 等价 PPO step80 本身，已有结果（98.2/99.4/92.8/97.2），不重跑。

**注**：35321/35355 都是被 SIGTERM 外部 kill（exit 0:15，无 OOM/Traceback），疑似 gnho031 共享节点上别的用户/调度器导致。35365 第三次跑通。后续若再遇到，建议直接只跑 libero_10 单 suite + 限 1.5h walltime。

---

## 五、结果表

| ckpt | spatial | object | goal | libero_10 | avg | 评测 Job |
|------|---------|--------|------|-----------|-----|---------|
| **base_25k JAX** (原始 orbax) | 99.6 | 99.0 | **99.4** | 92.0 | **97.50** | 34702 |
| **base_25k PT** (safetensors) | 99.2 | 99.0 | **96.4** | 92.0 | 96.65 | 35396 |
| merged α=0.3           | TBD  | TBD  | TBD  | TBD  | TBD   | — |
| merged α=0.5           | TBD  | TBD  | TBD  | TBD  | TBD   | — |
| **merged α=0.7**       | **98.0** | **98.6** | **96.6** | **96.2** | **97.35** | 35321 + 35365 |
| **merged α=0.9**       | 96.8 | 99.2 | 93.2 | 96.4 | 96.40 | 35385 |
| PPO step80 (α=1)       | 98.2 | 99.4 | **92.8** | 97.2 | 96.90 | 35272 |

### 5.1 JAX → PT 转换的精度损失（重要修正）

JAX orbax → PyTorch safetensors 转换在 `libero_goal` 上掉 **3.0 pp**（99.4 → 96.4），其它 3 个 suite 几乎无差。avg 掉 0.85 pp。

**结论**：PT 链路的 PPO / merged 数字应当与 **base_25k PT** 比较，而不是 JAX 原版：

| 对比 | base_25k PT | merged α=0.7 | 差值 |
|------|-------------|--------------|------|
| spatial | 99.2 | 98.0 | −1.2 |
| object  | 99.0 | 98.6 | −0.4 |
| goal    | 96.4 | 96.6 | **+0.2** |
| libero_10 | 92.0 | 96.2 | **+4.2** |
| **avg** | **96.65** | **97.35** | **+0.7** |

→ α=0.7 在 PT 同源比较下 **超过 base_25k 0.7 pp**（不是之前以为的"接近"）。libero_10 +4.2 pp 是核心增益，goal 几乎追平 PT base。

### 5.2 α=0.9 反常现象

α=0.9 (avg 96.40) **比 α=1.0/PPO (avg 96.90) 还差**，违反单调直觉。可能原因：

- 线性插值在 α 接近但不等于 1 时引入了"非干净"中间表征，与 RL 的 BatchNorm/RMSNorm 统计量不完全一致；
- libero_goal 在 α=0.9 (93.2) 比 PPO (92.8) 略高、比 α=0.7 (96.6) 明显低 —— 说明 α 在 [0.7, 0.9] 之间有非单调性。

下一步可以扫 α=0.5 或 α=0.6 验证 sweet spot。

### 5.3 α=0.7 libero_10 逐 task 成功率（10 tasks × 50 trials）

| task | rate | task | rate |
|------|------|------|------|
| 1 | 98% | 6 | 100% |
| 2 | 100% | 7 | 98% |
| 3 | 100% | 8 | 100% |
| 4 | 98% | 9 | **80%** ← 最低 |
| 5 | 96% | 10 | 92% |

汇总：481/500 = 96.2%

### 5.4 libero_goal 逐 task 比较 — PPO 负迁移定位

| task | base_25k JAX | merged α=0.7 | PPO step80 | PPO 损失 | 融合恢复 |
|------|--------------|--------------|------------|----------|----------|
| **1** | 0.98 | **0.92** | **0.54** | **−0.44** | +0.38 |
| 10 | 1.00 | 0.96 | 0.82 | −0.18 | +0.14 |

其余 8 个 task 三方都在 0.90-1.00 区间。**libero_goal task1 是 PPO 负迁移的核心源**。

### 5.5 libero_10 最后两个 task — RL 增益定位

| task | base_25k JAX | merged α=0.7 | PPO step80 |
|------|--------------|--------------|------------|
| **9** | 0.70 | 0.80 | **0.92** |
| **10** | 0.96 | 0.92 | 0.90 |

→ task9 是 base 的短板（70%），RL 拉到 92%（+22），融合保留一半。task10 三方接近。

### α=0.7 libero_10 逐 task 成功率（10 tasks × 50 trials）

| task | rate | task | rate |
|------|------|------|------|
| 1 | 98% | 6 | 100% |
| 2 | 100% | 7 | 98% |
| 3 | 100% | 8 | 100% |
| 4 | 98% | 9 | **80%** ← 最低 |
| 5 | 96% | 10 | 92% |

汇总：481/500 = 96.2%

### 关键观察

1. **α=0.7 抢救回 libero_goal**：92.8% (PPO) → 96.6%（+3.8 pp），同时 libero_10 仅小幅下降 97.2 → 96.2%（-1.0 pp）。
2. **avg 几乎追平 base**：97.35% vs base 97.50%，仅差 0.15 pp，但 libero_10 比 base 高 4.2 pp。
3. **依然有 task9 短板**（80%）—— PPO 在 libero_10 训练目标 task 上的不均衡仍然继承到 merged 模型。后续可以看是否扫 α=0.5/0.3 能进一步缓解。

---

## 六、关键坑与决策

1. **N=1 不用 WUDI**：单一微调模型时 SVD 去中心化退化为线性插值，没有意义。
2. **不依赖隐式"差为 0"**：第一版脚本对所有 812 个 key 都做 `b - a`，依赖 paligemma 数值相同来自然得到 `tv=0`。改为显式 `PI05_PPO_TRAINABLE_PREFIXES` 白名单，更明确、更快、且未来面对部分解冻的 RL 训练也更稳。
3. **fp32 中转**：bf16 直接相减会丢精度，特别是 RL 改动量级较小时。
4. **节点策略**：gnho031 当前由别人占着 GPU 但 CPU 空闲，融合（0 GPU）可放 031；eval（2 GPU）也提交到 031（避开正被占用的 gnho034）。

---

## 七、相关历史实验对照（PPO 融合 vs 历史 SFT 微调路径）

详见 `docs/experiment_log_2026-04-29.md`。

### 7.1 t89 系列 — libero_10 task 8+9 only SFT 微调

- **基础权重**：`l10_mix50@2000`（avg=96.6% 历史最佳之一），不是 base_25k
- **训练数据**：libero_10 task 8 + task 9 = 63 episodes (moka pots + mug microwave)
- **训练**：500 step，bs=32，lr peak=2e-5
- **输出**：`pi05_libero_10_t89_from_l10mix50_2k/ft_l10_t89_from_l10mix50_2k/{300, 499}`

| ckpt | spatial | object | goal | libero_10 | avg | Job |
|------|---------|--------|------|-----------|-----|-----|
| l10_mix50 @ 2000 (基线) | 98.8 | 98.6 | 97.4 | 91.6 | 96.6 | — |
| **t89 @ step300** | **98.0** | **97.8** | **97.4** | **92.6** | **96.5** | 34569 |
| t89 @ step500 | **30.2** ⚠ | 98.8 | 97.0 | 92.0 | 80.0 | 34629 |

历史结论：
- **step300 是 sweet spot**：libero_10 +1.0 pp，其它 suite 持平
- **step500 spatial 灾难性遗忘**（98.0→30.2），证明小数据集专项 SFT 有极窄的 sweet spot

### 7.2 t89 + base_15k 加权融合（Job 34645）

`merged_LLM = 0.7 × t89_step300 + 0.3 × base_15k`，scope=llm_only（视觉编码器保持 base_15k，action 投影 frozen）

- 输出：`checkpoints/avg_merge/l10t89_step300_x07_base15k_x03/`
- 注意：scope 与本日实验不同 —— 本日 PPO 只动 action expert + projections，t89 历史融合是 LLM scope

### 7.3 路径对比

| 路径 | 数据规模 | libero_10 增益 | 副作用 | 目标 |
|------|---------|---------------|--------|------|
| t89 SFT (29+34=63 eps) | 极小 | +1.0 pp (91.6→92.6) | step500 spatial 崩 | 显式专项强化 |
| **PPO α=1** (libero_10 全 suite) | 全 RL rollout | +5.2 pp (92.0→97.2) | goal task1 −44 pp | 端到端 RL |
| **PPO + α=0.7 融合** | RL + SFT 插值 | +4.2 pp (92.0→96.2) | goal task1 −6 pp（基本恢复） | RL 信号 + SFT 安全网 |

→ PPO + 融合在 libero_10 增益维度上明显优于历史 t89 SFT 路径（+4.2 vs +1.0 pp），且无 spatial 崩盘。

---

## 八、下一步

- [x] α=0.7 完整结果
- [x] α=0.9 完整结果
- [x] base_25k PT 完整结果
- [ ] 提交 α=0.5 eval（验证 [0.7, 0.9] 间的非单调性）
- [ ] α=0.3 eval（接近 SFT 端，预测主要差异在 libero_10）
- [ ] 若 α 扫描确认 0.7 是 sweet spot，考虑用同样方法融合 step100 / step120
- [ ] 可选：参考 t89 历史经验，对 PPO step80 + α=0.7 merged 再做一次 task9 专项 FT（解决 80% 短板）
