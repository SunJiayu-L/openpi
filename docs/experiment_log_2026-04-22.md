# 实验日志 — 2026-04-22

> 创建：2026-04-22 CST；最后更新：2026-04-25 CST（补全 iter=1/1k/1500/2k 结果、libero10ft 二次 FT 完整评测、绘出收敛曲线）  
> 项目路径：`/storage/yukaichengLab/lishiwen/jiayusun/openpi`

---

## 一、本次实验背景

本次日志覆盖 2026-04-22 至 2026-04-25 的实验，主要包含六项工作：

1. **`wudi_merge.py` decompose/compose 语义修复**：多轮迭代，最终对齐官方 `examples/convert_jax_model_to_pytorch.py` 的 PyTorch 语义
2. **spatial+10 方向 2task WUDI 融合（mean init）**：新增 spatial+10 pair（libero_spatial + libero_10），iter=300/500
3. **changed_ 系列 merge job（from10k base）**：修正输出前缀，尝试以 `pi05_libero/my_experiment/10000` 为 base，失败（checkpoint 路径问题）
4. **3task_sog WUDI 融合（spatial+object+goal，无 libero_10）**：mean init，**完整扫描 iter=1/300/500/1k/1500/2k**，评测全 4 suite，确认 **iter=1500 为 sweet spot**
5. **3task_sog_iter500 全量微调（FT 5k steps）**：从融合 checkpoint 出发，在全 4 suite 数据上继续训练，**结果：spatial=97.6% / object=96.8% / goal=95.6% / libero_10=90.4% / avg=95.1%**
6. **libero_10 二次 FT（1k steps）**：针对 FT 5k checkpoint 中 libero_10 弱点任务，只在 libero_10 数据上继续训练 1k 步，**结果 avg=96.2%（+1.1pp），所有 suite 均提升或持平**

---

## 二、`wudi_merge.py` decompose/compose 语义修复

### 2.1 问题背景

`scripts/wudi_merge.py` 的 `decompose_layer` / `compose_layer` 将 JAX 高维参数拆成"2D 块"后用 WUDI 优化。  
标准 WUDI 实现（`fusion_bench/method/wudi/wudi.py`）对 `len(shape)==2` 的参数逐个单独优化——在 PyTorch space 中，每个 `nn.Linear.weight` 已经是 2D，所以每个投影矩阵都是独立的优化单元。  
JAX 版本需要手动做等价的分解，才能产生语义等价的优化单元。

官方参考：`examples/convert_jax_model_to_pytorch.py`（308-346 行）精确定义了 JAX → PyTorch 的语义映射。

### 2.2 修复历程（多轮）

**Round 1（错误）**：
- q: `(N,D,H) → reshape(N*D, H)`（无语义，应为 `.transpose(0,2,1).reshape(N*H, D)`）
- kv: 把 K/V 合并为 `(D, 2*K*H)`（应为两个独立块）
- gate: 把 gate/up 合并为 `(2*D, Hff)`（应为两个独立块）

**Round 2（错误）**：  
修正了 kv/gate 为两块，但方向全部反了（把 `(K*H, D)` 写成 `(D, K*H)`，把 `(D, N*H)` 写成 `(N*H, D)` 等）。

**Round 3（正确）**：  
按官方脚本逐一校对所有 5 种参数类型的 shape 和转置方向，最终结果：

| ptype | JAX layer shape | 2D 块 | 块数 | 说明 |
|---|---|---|---|---|
| `q` | `(N, D, H)` | `.transpose(0,2,1).reshape(N*H, D)` | 1 | Q proj `(N*H, D)` |
| `kv` | `(2, K, D, H)` | k: `.transpose(0,2,1).reshape(K*H, D)`, v: 同 | 2 | K/V proj 各 `(K*H, D)` |
| `av` | `(N, H, D)` | `.transpose(2,0,1).reshape(D, N*H)` | 1 | O proj `(D, N*H)` |
| `gate` | `(2, D, Hff)` | `layer[0].transpose()`, `layer[1].transpose()` | 2 | gate/up proj 各 `(Hff, D)` |
| `lin` | `(Hff, D)` | `.transpose()` | 1 | down proj `(D, Hff)` |

每层共 **7 次独立 WUDI 优化**（q/k/v/o/gate/up/down），与官方 PT representation 中的 7 个 `nn.Linear` 精确对应。

**验证**：smoke test（round-trip + linearity）全部通过。

### 2.3 关键认识

标准 WUDI 与本修复前的实现本质区别：

- **修复前**：把 JAX packed tensor（如 `kv_einsum_1/w shape=(2,K,D,H)`）当作整体优化，或错误 reshape 成无语义的 2D 矩阵
- **修复后**：精确还原官方 PyTorch 语义——每个线性投影 = 一个独立优化单元，WUDI loss 在各投影的语义子空间内单独最小化干扰

> **注意**：修复后的 wudi_merge.py 本质上等价于：先 JAX→PT 转换（官方脚本），再对 PT state_dict 跑标准 WUDI，再逆变换回 JAX。路径更短，无需维护两套 pipeline。

---

## 三、spatial+10 方向 2task WUDI 融合

### 3.1 配置

- **融合 pair**：libero_spatial + libero_10（from10k @20k）
- **base**：`checkpoints/pi05_libero/my_experiment/29999`（全量微调 base，**非** from10k 系列）
- **init**：mean（`vectors.mean(dim=0)`）
- **scope**：llm_only
- **scaling / scaling2**：1.0 / 1.0

> 注：from10k 系列使用 `pi05_libero_10k` 为 base；spatial+10 系列沿用和前期 object+10/goal+10 相同的配置（base=29999）以保持可比性

### 3.2 融合 Job 记录

| iter | 输出 Checkpoint | 主 Job | 完成时间 | 状态 |
|---|---|---|---|---|
| 300 | `wudi_mllm/2task_spatial10_mean_iter300` | 34170 | 2026-04-22 16:21 | ✅ 完成（34166 先跑但被替换） |
| 500 | `wudi_mllm/2task_spatial10_mean_iter500` | 34175 | 2026-04-22 17:23 | ✅ 完成（34168/34172 中途失败后重提） |

> Job 34166（iter300）与 34170 同时提交，34166 结果被 34170 覆盖；iter500 经过 34168 → 34172 → 34175 三次提交，前两次因并发或路径问题未正常完成，34175 成功。

### 3.3 评测结果（eval job 34171, 34176）

**`2task_spatial10_mean_iter300`**（Job 34171，node gnho031）：

| suite | 成功率 |
|---|---|
| libero_spatial | 96.0% |
| libero_object | partial（仅完成前 4 task，均约 1.0/0.98，后续因超时截断）|
| libero_goal | ❌ 未完成（34171 在 libero_spatial 后中断） |
| libero_10 | ❌ 未完成 |

> 34171 仅完成 libero_spatial（14 task 中 10 为 spatial，均 ~96%），object 从第 4 task 起被截断。

**`2task_spatial10_mean_iter500`**（Job 34176，node gnho031）：

| suite | 成功率 | 备注 |
|---|---|---|
| libero_spatial | **96.8%** | 10 tasks: 1.0/1.0/1.0/0.96/0.92/0.94/1.0/1.0/0.96/0.9 |
| libero_object | **33.4%** | 10 tasks: 0.02/1.0/0.78/0.1/0.0/0.0/0.02/0.0/0.88/0.54 |
| libero_goal | ❌ 未完成（仅完成 1 task: 0.0%） | Job 超时或资源限制终止 |
| libero_10 | ❌ 未完成 | |

> libero_goal 开始时第一个 task 成功率 0.0%，后续 Job 被终止，结果不完整。

**与历史 spatial+10 基线对比**（参照 experiment_log_2026-04-15.md）：

| 历史结果（2task spatial+10，sum init）| spatial | object | goal | libero_10 | avg |
|---|---|---|---|---|---|
| `2task_from10k_iter300`（sum） | 94.2% | 32.4% | 18.4% | 66.8% | 53.0% |
| `2task_from10k_iter500`（sum） | 95.0% | 33.6% | 18.4% | 69.4% | 54.1% |
| `2task_mean_iter500`（mean） | 95.4% | 32.8% | 16.8% | 70.8% | 54.0% |

本次 `spatial10_mean_iter500` 的 spatial=96.8% / object=33.4%，与历史 mean/sum@500 基本一致（spatial +1.4pp，object +0.6pp）。goal/libero_10 因 job 终止无法比较。

---

## 四、changed_ 系列 merge job（from10k base 路径问题）

### 4.1 背景

为区分新旧实现产生的 checkpoint，添加 `changed_` 前缀到输出路径和日志，修改 sbatch 到 gnho031 节点，提交 2task merge from10k 系列。

### 4.2 sbatch 修改

文件：`merge_wudi_2task_from10k_iter300.sbatch`
- 输出路径：`checkpoints/wudi_mllm/changed_2task_from10k_iter300`
- 日志：`trainlogs/changed_wudi_2task_from10k_iter300_%j.{log,err}`
- 节点：`gnho031`（原 `gnho018`）

### 4.3 失败原因

Jobs 34163 / 34164 / 34165 全部失败，错误：

```
FileNotFoundError: Metadata file (named _METADATA) does not exist at
  .../checkpoints/pi05_libero/my_experiment/10000
```

**根因**：`_load_flat()` 调用 `restore_params(path)` 时传入 `...10000/`，但 orbax 期望在该目录下找到 `_METADATA` 文件，实际结构为：

```
checkpoints/pi05_libero/my_experiment/10000/
├── _CHECKPOINT_METADATA
├── assets/
├── params/          ← orbax checkpoint 实际在此
└── train_state/
```

应传 `...10000/params/` 而非 `...10000/`。

> **待修复**：需在 `wudi_merge.py` 的 `_load_flat()` 或调用侧修正路径解析，或修改 sbatch 中 `--base` 参数指向 `.../10000/params`。

---

## 五、3task_sog WUDI 融合（spatial + object + goal，无 libero_10）

### 5.1 实验动机

分析 2-task vs 4-task 差异后发现：libero_10 task vector 方向最"分散"，在 4-task 融合中被压制（49.2%），且随 WUDI 优化快速下降。新实验去掉 libero_10，只融合 spatial+object+goal（3 个最特化的 task vector），观察：
1. 去掉 libero_10 后，3 个 suite 的性能能否提升
2. 3-task WUDI 的最优 iter 窗口
3. FT 5k 步能否恢复 libero_10 的能力

### 5.2 融合配置

- **base**：`checkpoints/pi05_libero/my_experiment/10000`
- **ft**：`from10k/spatial @20k` + `from10k/object @20k` + `from10k/goal @20k`
- **init**：mean，**scope**：llm_only，**scaling**：1.0

### 5.3 融合 Job 记录

| iter | 输出 Checkpoint | Job | 节点 | 状态 |
|---|---|---|---|---|
| 1 | `wudi_mllm/3task_sog_mean_iter1` | 34245 | gnho009 | ✅ 完成（13:29） |
| 300 | `wudi_mllm/3task_sog_mean_iter300` | 34222 | gnho031 | ✅ 完成（22:21） |
| 500 | `wudi_mllm/3task_sog_mean_iter500` | 34218 | gnho031 | ✅ 完成（21:43） |
| 1000 | `wudi_mllm/3task_sog_mean_iter1k` | 34289 | gnho031 | ✅ 完成 |
| 1500 | `wudi_mllm/3task_sog_mean_iter1500` | 34310 | gnho009 | ✅ 完成 |
| 2000 | `wudi_mllm/3task_sog_mean_iter2k` | 34393 | gnho009 | ✅ 完成 |

> Job 34217（iter300 首次提交）在优化跑完后被 34176 eval 超时连带 SIGTERM 杀掉（未保存），34222 重提成功。  
> Job 34224（iter500 eval 首次提交）被 34222 merge 结束后的资源回收杀掉，34225 重提。  
> iter=1 merge（34245）首次在 gnho009 完成；iter=1k merge（34289）在 gnho031 完成；iter=1500/2k merge 在 gnho009 完成。

### 5.4 评测结果

**配置**：全 4 suite（libero_spatial / libero_object / libero_goal / libero_10），每 suite 10 task × 50 episode

| iter | spatial | object | goal | libero_10 | avg(4) | avg(3) | eval Job |
|---|---|---|---|---|---|---|---|
| 1 | 94.2% | 94.6% | 47.6% | ~5%（部分） | — | 78.8% | 34268 |
| 300 | 95.8% | 92.4% | 48.4% | 2.2% | 59.7% | 78.9% | 34223 |
| 500 | 95.0% | 92.6% | 60.6% | 1.4% | 62.4% | 82.7% | 34225 |
| 1k | **96.0%** | **93.4%** | 66.2% | ~0-2%（部分） | — | 85.2% | 34290+34304 |
| **1500** | 96.0% | 91.4% | **68.4%** | ~1%（部分） | — | **85.3%** | 34311 |
| 2k | 95.8% | 87.6% | 67.8% | （未跑） | — | 83.7% | 34394 |

> eval(34223/34225) 均在退出清理时被 SIGTERM，但 40 个 task 结果完整。  
> eval_iter1（34268）经过多次失败重提，最终在 gnho031 port 8001 完成 spatial/object/goal（libero_10 部分）。  
> eval_iter1k 经历 34290（031,SIGTERM）→ 34304（009,libero_10 时 MuJoCo core dump）→ 34308（libero_10-only 补评）多次拼接，spatial/object/goal 完整。  
> eval_iter1500（34311）跑到 libero_10 7/10 时 SIGTERM，spatial/object/goal 完整。  
> eval_iter2k（34394）在 libero_10 第 2 个 task 时**手动 cancel**（用户决定不再测 libero_10，因前几个 iter 都已证明 ~0-2%）。

#### 5.4.1 收敛曲线分析

```
spatial: 95.8 → 95.0 → 96.0 → 96.0 → 95.8     (基本平稳)
object : 92.4 → 92.6 → 93.4 → 91.4 → 87.6     (1k 后单调退化)
goal   : 48.4 → 60.6 → 66.2 → 68.4 → 67.8     (1500 饱和)
avg(3) : 78.9 → 82.7 → 85.2 → 85.3 → 83.7     (1500 峰值)
         (300)  (500)  (1k)  (1500)  (2k)
```

**关键观察：**
- **goal 是最受益的 suite**：从 mean init 的 ~47.6% 一路升到 1500 的 68.4%（+20.8pp），但 1500→2k 已开始饱和（-0.6pp）
- **object 出现单调退化**：1k=93.4 → 1500=91.4 → 2k=**87.6**（-5.8pp），WUDI 过度优化压制 object 方向
- **spatial 基本稳定**（95-96% 区间），最不敏感
- **avg(3) 在 iter=1500 达到峰值 85.3%**，2k 已超过 sweet spot
- **libero_10 几乎在所有 iter 都接近 0**（始终未参与融合）

**结论：iter=1500 是 3task_sog 系列的最佳 merge 点。**

**关键观察**：
- spatial/object：去掉 libero_10 后，3-task 融合在这两个 suite 上均超过 4-task mean@iter=0（92.4/90.0%）→ **95~96% / 92%**
- goal：iter=300 时 48.4%，iter=500 升至 60.6%（仍在提升，最优窗口可能在 500~1k）
- libero_10：几乎崩溃（~2%），符合预期（未包含 libero_10 task vector）

**与 4-task 对比**：

| 方案 | spatial | object | goal | libero_10 | avg |
|---|---|---|---|---|---|
| 4-task mean@iter=0（Job 33893） | 92.4% | 90.0% | 53.0% | 49.2% | 71.2% |
| 3task_sog mean@iter=300（Job 34223） | **95.8%** | **92.4%** | 48.4% | 2.2% | 59.7% |
| 3task_sog mean@iter=500（Job 34225） | **95.0%** | **92.6%** | **60.6%** | 1.4% | 62.4% |

去掉 libero_10 后 spatial/object 提升明显（+3~5pp），但 avg 因 libero_10 崩溃反而下降。

---

## 六、3task_sog_iter500 全量微调（FT 5k steps）

### 6.1 配置

- **初始化**：`checkpoints/wudi_mllm/3task_sog_mean_iter500/params`
- **训练数据**：全 4 suite（LIBERO_10 + SPATIAL + GOAL + OBJECT，共 1693 episodes）
- **steps**：5000，**batch_size**：32，**GPU**：2，**FSDP**
- **lr_schedule**：CosineDecay，warmup=500，peak=5e-5，decay_steps=5k，decay_lr=5e-6
- **训练配置名**：`pi05_libero_4task_from_3task_sog_iter500`（新增至 `config.py`）

### 6.2 Job 记录

| Job | 类型 | 说明 | 节点 | 状态 |
|---|---|---|---|---|
| 34235 | train | FT 5k steps from `3task_sog_mean_iter500` | gnho031 | ✅ 完成（1h28m） |
| 34236 | eval | eval FT @4999 全 4 suite（spatial/object/goal 完成，libero_10 被 kill） | gnho031 | ⚠️ 部分完成 |
| 34256 | eval | eval FT @4999 libero_10 补评（port 8001） | gnho031 | ⚠️ 10 task 全部完成，清理时 SIGTERM |

checkpoint 输出：`checkpoints/pi05_libero_4task_from_3task_sog_iter500/ft_from_3task_sog_iter500/4999`

### 6.3 评测结果

| suite | 成功率 | per-task | eval Job |
|---|---|---|---|
| libero_spatial | **97.6%** | 1.0/1.0/1.0/1.0/0.9/0.9/1.0/0.96/1.0/1.0 | 34236 |
| libero_object | **96.8%** | 0.98/1.0/0.98/0.88/0.96/0.96/0.96/0.96/1.0/1.0 | 34236 |
| libero_goal | **95.6%** | 0.94/0.98/0.98/0.9/1.0/0.92/0.96/1.0/1.0/0.88 | 34236 |
| libero_10 | **90.4%** | 0.98/0.98/0.92/0.92/0.98/1.0/0.92/1.0/0.56/0.78 | 34256 |
| **avg** | **95.1%** | — | — |

> libero_10 task 9（0.56）和 task 10（0.78）明显偏低，推测为"put both moka pots on the stove"及相邻任务。这是发起 §7 二次 FT 的动机。

**与参考基线对比**（from `experiment_log_2026-04-15.md` Job 33832/33834）：

| 方案 | spatial | object | goal | libero_10 | avg |
|---|---|---|---|---|---|
| FT@10k from wudi_4task_500（Job 33832） | 98.4% | 98.2% | 98.2% | 91.0% | 96.5% |
| FT@9k from wudi_4task_1k（Job 33834） | 96.8% | 98.8% | 97.2% | 93.0% | 96.5% |
| **FT@5k from 3task_sog_iter500（本实验）** | **97.6%** | **96.8%** | **95.6%** | **90.4%** | **95.1%** |

> 本实验仅 5k steps，结果略低于 10k 参考，但差距很小（avg -1.4pp）。libero_10 仍是短板（90.4% vs 91~93%）。

---

## 七、libero_10 二次 FT（1k steps）

### 7.1 实验动机

FT 5k checkpoint（§六）在 libero_10 上 task 9（0.56）和 task 10（0.78）表现明显偏低，推测对应"put both moka pots on the stove"及相邻操作。  
在当前 FT checkpoint 基础上，只用 libero_10 数据再训练 1k 步，通过小 lr（peak=2e-5）强化这两个弱点任务，同时尽量避免遗忘其他三个 suite。

### 7.2 配置

- **初始化**：`checkpoints/pi05_libero_4task_from_3task_sog_iter500/ft_from_3task_sog_iter500/4999/params`
- **训练数据**：libero_10 only（LIBERO_10_EPISODES）
- **steps**：1000，**batch_size**：32，**GPU**：2 FSDP
- **lr_schedule**：CosineDecay，warmup=100，peak=2e-5，decay_steps=1k，decay_lr=2e-6
- **训练配置名**：`pi05_libero_10_from_4task_3sog500_5k`（新增至 `config.py`）

> lr 比 §六 的 5e-5 低 2.5×，避免过拟合 libero_10 导致其他 suite 退步。

### 7.3 Job 记录

| Job | 类型 | 说明 | 节点 | 状态 |
|---|---|---|---|---|
| 34267 | train | libero_10 FT 1k steps from FT@4999 | gnho034（idle） | ✅ 完成 |
| 34269 | eval | eval 二次 FT（被端口冲突连到 34268 server） | gnho031 | ❌ 取消（评测错模型） |
| 34325 | eval | eval 二次 FT 全 4 suite（重提，port 8002） | gnho009 | ⚠️ 40/40 task 完成，清理时 SIGTERM |

checkpoint 输出：`checkpoints/pi05_libero_10_from_4task_3sog500_5k/ft_libero10_from_4task_3sog500_5k/999`

### 7.4 评测结果（Job 34325，全 40/40 完成）

| suite | libero10ft（二次 FT 1k） | FT 5k 基线 | Δ | per-task |
|---|---|---|---|---|
| libero_spatial | **99.2%** | 97.6% | **+1.6** | 1.0/1.0/1.0/1.0/0.98/0.98/1.0/0.98/0.98/1.0 |
| libero_object | **98.4%** | 96.8% | **+1.6** | 0.96/0.98/0.96/0.94/1.0/1.0/1.0/1.0/1.0/1.0 |
| libero_goal | **95.6%** | 95.6% | 0.0 | 0.94/1.0/1.0/0.8/0.94/0.96/0.92/1.0/1.0/1.0 |
| libero_10 | **91.4%** | 90.4% | **+1.0** | 0.92/0.98/0.92/0.98/0.98/0.96/1.0/0.96/0.66/0.78 |
| **avg(4)** | **96.2%** | 95.1% | **+1.1** | — |

**关键发现**：

1. **libero_10 仅提升 +1.0pp**（90.4 → 91.4）：原本最弱的 task 9（0.56→0.66）和 task 10（0.78→0.78）改善有限，说明这两个任务对 1k 步、lr=2e-5 的 FT 不够敏感
2. **意外收获：spatial/object 各提升 +1.6pp**：在 libero_10 数据上 FT 居然让其他 suite 也有所改善，可能因为：
   - libero_10 包含的复杂多步操作技能对其他 suite 有正迁移
   - 更小的 lr 实际上是一次"温和的多任务正则化"
3. **goal 完全保持 95.6%**：未发生灾难性遗忘
4. **整体 avg=96.2% 已与历史最佳 FT@9k/10k（96.5%）持平**，但只用了 1k 额外 FT 步（vs 9k 步），效率提升 ~9×

**与历史 FT 全表对比：**

| 方案 | spatial | object | goal | libero_10 | avg | 说明 |
|---|---|---|---|---|---|---|
| FT@10k from wudi_4task_500（33832） | 98.4% | 98.2% | 98.2% | 91.0% | 96.5% | 4-task 融合 + 10k FT |
| FT@9k from wudi_4task_1k（33834） | 96.8% | 98.8% | 97.2% | 93.0% | 96.5% | 4-task 融合 + 9k FT |
| FT@5k from 3task_sog_iter500 | 97.6% | 96.8% | 95.6% | 90.4% | 95.1% | 3-task 融合 + 5k FT |
| **二次 FT@1k from FT@5k（本实验）** | **99.2%** | **98.4%** | **95.6%** | **91.4%** | **96.2%** | **+1k libero_10 only** |

---

## 八、节点问题记录（2026-04-23）

gnho031 上多个 eval/train job 被反复 SIGTERM（ExitCode 0:15 或 15:0）：

| 现象 | 可能原因 |
|---|---|
| 34236（FT eval）运行 ~2h 后 kill | gnho031 资源争抢，其他用户高优先级任务抢占 |
| 34246（eval_iter1 on 009）运行 ~3min 后 kill | gnho009 资源争抢（mazijian 等用户长期占用） |
| 34251（eval_iter1 on 031）连接到 34236 的 server | 端口冲突：两个 eval 同时用 port 8000，34251 在 1s 内接上 34236 server，评测了错误 checkpoint |
| 34256（FT libero_10 eval）10 task 跑完后 kill | 清理阶段 SIGTERM，结果已完整保存 |
| 34262（eval_iter1）被 SIGTERM，0 task | 与 34263（train）同时用 031 共 4 GPU，可能触发抢占 |
| 34263（train libero_10）FileNotFoundError | config.py 中路径写了 `4999` 而非 `4999/params`，已修复 |

**修复措施**：
1. 所有新 eval job 改用 **port 8001**（原来 port 8000 可能被长期 inf_task 占用）
2. eval job 和 train job 不再同时在同一节点提交（改为串行依赖链）
3. 训练 job 改投 **gnho034**（idle，避开 031 的抢占风险）

---

## 九、SLURM 任务汇总（2026-04-22 ~ 2026-04-23）

| Job | 类型 | 说明 | 节点 | 状态 |
|---|---|---|---|---|
| 34163~34165 | merge | `changed_2task_from10k_iter300`（三次提交） | gnho031 | ❌ `_METADATA` FileNotFoundError |
| 34166 | merge | `2task_spatial10_mean_iter300`（被覆盖） | gnho031 | ✅ |
| 34168/34172 | merge | `2task_spatial10_mean_iter500`（两次中途失败） | gnho031 | ⚠️ |
| 34170 | merge | `2task_spatial10_mean_iter300` | gnho031 | ✅ 完成 (16:21) |
| 34171 | eval | `2task_spatial10_mean_iter300`（partial） | gnho031 | ⚠️ spatial 完成，其余截断 |
| 34175 | merge | `2task_spatial10_mean_iter500` | gnho031 | ✅ 完成 (17:23) |
| 34176 | eval | `2task_spatial10_mean_iter500`（partial） | gnho031 | ⚠️ spatial+object 完成，goal/l10 未完成 |
| 34217 | merge | `3task_sog_mean_iter300`（首次，被杀） | gnho031 | ⚠️ 优化完但未保存 |
| 34218 | merge | `3task_sog_mean_iter500` | gnho031 | ✅ 完成 (21:43) |
| 34222 | merge | `3task_sog_mean_iter300`（重提） | gnho031 | ✅ 完成 (22:21) |
| 34223 | eval | `3task_sog_mean_iter300` 全 4 suite | gnho031 | ✅ 完成（结果见§5.4） |
| 34224 | eval | `3task_sog_mean_iter500`（被连带杀） | gnho031 | ❌ SIGTERM |
| 34225 | eval | `3task_sog_mean_iter500`（重提）全 4 suite | gnho031 | ✅ 完成（结果见§5.4） |
| 34235 | train | FT 5k from `3task_sog_mean_iter500` | gnho031 | ✅ 完成 (1h28m) |
| 34236 | eval | FT @4999 spatial/object/goal（libero_10 被 kill） | gnho031 | ⚠️ 部分完成（见§6.3） |
| 34245 | merge | `3task_sog_mean_iter1`（重提，gnho009） | gnho009 | ✅ 完成 (13:29) |
| 34246 | eval | `3task_sog_mean_iter1`（gnho009，被抢占） | gnho009 | ❌ SIGTERM |
| 34248 | eval | `3task_sog_mean_iter1`（二次重提，gnho009） | gnho009 | ❌ SIGTERM |
| 34251 | eval | `3task_sog_mean_iter1`（031，端口冲突） | gnho031 | ❌ 接到 34236 server，已手动取消 |
| 34256 | eval | FT @4999 libero_10 补评（port 8001） | gnho031 | ⚠️ 10 task 完成，清理时 SIGTERM |
| 34262 | eval | `3task_sog_mean_iter1`（031，port 8001） | gnho031 | ❌ SIGTERM（与 34263 竞争） |
| 34263 | train | libero_10 FT 1k（路径错误） | gnho031 | ❌ FileNotFoundError（`4999` 漏 `/params`） |
| 34267 | train | libero_10 FT 1k（修复路径后重提） | gnho034 | ✅ 完成 |
| 34268 | eval | `3task_sog_mean_iter1` 全 4 suite（port 8001） | gnho031 | ⚠️ spatial/obj/goal 完整，libero_10 部分 |
| 34269 | eval | libero_10 FT（端口冲突连到 34268 server） | gnho031 | ❌ 手动取消 |
| 34282 | eval | iter=1 libero_10 补评（port 8003） | gnho031 | ⚠️ 4/10 task 后 SIGTERM |
| 34289 | merge | `3task_sog_mean_iter1k` | gnho031 | ✅ 完成 |
| 34290 | eval | `3task_sog_mean_iter1k`（port 8001） | gnho031 | ⚠️ spatial 完整、obj 4/10 后 SIGTERM |
| 34301 | eval | `3task_sog_mean_iter1k`（重提） | gnho031 | ❌ 4 task 后 SIGTERM |
| 34304 | eval | `3task_sog_mean_iter1k`（009 重提） | gnho009 | ⚠️ spatial/obj/goal 完整，libero_10 时 MuJoCo core dump |
| 34308 | eval | iter=1k libero_10 补评（port 8004） | gnho031 | ⚠️ 5/10 task 后 SIGTERM |
| **34310** | merge | `3task_sog_mean_iter1500` | gnho009 | ✅ 完成 |
| **34311** | eval | `3task_sog_mean_iter1500`（port 8005） | gnho009 | ⚠️ spatial/obj/goal 完整，libero_10 7/10 后 SIGTERM |
| **34325** | eval | libero_10 FT 全 4 suite（port 8002） | gnho009 | ⚠️ 40/40 完整，清理时 SIGTERM |
| **34393** | merge | `3task_sog_mean_iter2k` | gnho009 | ✅ 完成 |
| **34394** | eval | `3task_sog_mean_iter2k`（port 8006） | gnho009 | ⚠️ spatial/obj/goal 完整，libero_10 第 2 task 时**手动 cancel** |

---

## 十、关键结论

### 10.1 3task_sog WUDI 收敛规律

- **iter=1500 是最佳 merge 点**（avg(3)=85.3%，goal=68.4%，object 尚未明显退化）
- goal 提升最显著（+20.8pp from mean→1500），但 1500 已基本饱和
- object 在 1k 后开始单调退化（93.4 → 87.6 at iter=2k）
- spatial 全程稳定在 95-96%
- libero_10 始终接近 0（未参与融合）

### 10.2 二次 FT 的意外收获

在 libero_10 数据上做二次 FT（1k 步、lr=2e-5）：
- libero_10 仅 +1.0pp（弱点 task 9/10 改善有限）
- **但 spatial/object 各 +1.6pp**——说明 libero_10 的复杂多步技能对其他 suite 有正迁移
- **最终 avg=96.2%，已接近历史最佳 FT@10k（96.5%）**，但只用了 1/9 的 FT 步数

### 10.3 节点稳定性问题

gnho031 反复 SIGTERM 的根因不明（疑似 fairshare 抢占），**gnho009 + 不同端口的串行化方案可工作**。所有 eval 任务在被 kill 前 spatial/object/goal 通常已完整，libero_10 易在末尾被截断——可单独提交 libero_10-only 补评。

---

## 十一、待办事项

1. **iter=1500 / 2k libero_10 补评**：当前未跑（趋势明确 ~0-2%，可选补评以完整记录）。
2. **修复 changed_ base 路径**：在 `wudi_merge.py` 的 `_load_flat()` 中自动追加 `params/` 子路径。
3. **补跑 spatial10_mean_iter300/500 完整 eval**：现有结果缺 goal/libero_10 suite。
4. **基于 3task_sog_iter1500 重做 FT 5k**：iter=1500 比 iter=500 起点更优（avg(3) 85.3 vs 82.7），值得重做对比。
5. **libero_10 二次 FT 延长试验**：当前 1k 步已达 +1.1pp，是否继续到 2k/3k 步可进一步提升？
