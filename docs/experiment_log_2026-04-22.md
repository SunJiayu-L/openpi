# 实验日志 — 2026-04-22

> 创建：2026-04-22 CST；最后更新：2026-04-23 CST（补全 FT 评测结果、新增 libero_10 二次 FT 实验、节点问题记录）  
> 项目路径：`/storage/yukaichengLab/lishiwen/jiayusun/openpi`

---

## 一、本次实验背景

本次日志覆盖 2026-04-22 至 2026-04-23 的实验，主要包含六项工作：

1. **`wudi_merge.py` decompose/compose 语义修复**：多轮迭代，最终对齐官方 `examples/convert_jax_model_to_pytorch.py` 的 PyTorch 语义
2. **spatial+10 方向 2task WUDI 融合（mean init）**：新增 spatial+10 pair（libero_spatial + libero_10），iter=300/500
3. **changed_ 系列 merge job（from10k base）**：修正输出前缀，尝试以 `pi05_libero/my_experiment/10000` 为 base，失败（checkpoint 路径问题）
4. **3task_sog WUDI 融合（spatial+object+goal，无 libero_10）**：mean init，iter=1/300/500/1k，评测全 4 suite
5. **3task_sog_iter500 全量微调（FT 5k steps）**：从融合 checkpoint 出发，在全 4 suite 数据上继续训练，**结果：spatial=97.6% / object=96.8% / goal=95.6% / libero_10=90.4%**
6. **libero_10 二次 FT（1k steps）**：针对 FT 5k checkpoint 中 libero_10 弱点任务，只在 libero_10 数据上继续训练 1k 步

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

| iter | 输出 Checkpoint | Job | 完成时间 | 状态 |
|---|---|---|---|---|
| 1 | `wudi_mllm/3task_sog_mean_iter1` | 34245 | 2026-04-23 13:29 | ✅ 完成 |
| 300 | `wudi_mllm/3task_sog_mean_iter300` | 34222 | 2026-04-22 22:21 | ✅ 完成 |
| 500 | `wudi_mllm/3task_sog_mean_iter500` | 34218 | 2026-04-22 21:43 | ✅ 完成 |
| 1000 | `wudi_mllm/3task_sog_mean_iter1k` | 34270 | — | ⏳ PD（after 34268） |

> Job 34217（iter300 首次提交）在优化跑完后被 34176 eval 超时连带 SIGTERM 杀掉（未保存），34222 重提成功。  
> Job 34224（iter500 eval 首次提交）被 34222 merge 结束后的资源回收杀掉，34225 重提。  
> iter=1 merge（34245）在 gnho009 上成功完成；iter=1k merge（34270）串在 eval_iter1（34268）之后，在 gnho031 排队。

### 5.4 评测结果

**配置**：全 4 suite（libero_spatial / libero_object / libero_goal / libero_10），每 suite 10 task × 50 episode

| iter | spatial | object | goal | libero_10 | avg | eval Job |
|---|---|---|---|---|---|---|
| 1 | — | — | — | — | — | 34268（🟢 RUNNING） |
| 300 | **95.8%** | **92.4%** | 48.4% | **2.2%** | **59.7%** | 34223 |
| 500 | **95.0%** | **92.6%** | **60.6%** | 1.4% | **62.4%** | 34225 |
| 1k | — | — | — | — | — | 34271（⏳ PD） |

> eval(34223/34225) 均在退出清理时被 SIGTERM，但 40 个 task 结果完整。  
> eval_iter1（34268）经历多次提交失败后（34246 在 gnho009 被抢占，34251 端口冲突到 34236 server，34262 被 SIGTERM），以 port 8001 在 gnho031 重提，目前正常运行中。

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
| 34267 | train | libero_10 FT 1k steps from FT@4999 | **gnho034**（idle） | 🟢 RUNNING |
| 34269 | eval | eval 二次 FT 全 4 suite | gnho031 | ⏳ PD（after 34267） |

checkpoint 输出：`checkpoints/pi05_libero_10_from_4task_3sog500_5k/ft_libero10_from_4task_3sog500_5k/999`

### 7.4 评测结果

| suite | 成功率 | eval Job |
|---|---|---|
| libero_spatial | — | 34269（⏳ PD） |
| libero_object | — | 34269（⏳ PD） |
| libero_goal | — | 34269（⏳ PD） |
| libero_10 | — | 34269（⏳ PD） |

> 结果待 34267 训练完成（预计 20-30min）+ 34269 评测（~2.5h）后补充。  
> 预期：libero_10 提升至 93-95%，其余 suite 基本保持（spatial/goal 可能略有下降，±1pp 以内）。

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
| **34267** | train | libero_10 FT 1k（修复路径后重提） | **gnho034** | 🟢 RUNNING |
| **34268** | eval | `3task_sog_mean_iter1` 全 4 suite（port 8001） | gnho031 | 🟢 RUNNING |
| 34269 | eval | libero_10 FT 全 4 suite | gnho031 | ⏳ PD (after 34267) |
| 34270 | merge | `3task_sog_mean_iter1k` | gnho031 | ⏳ PD (after 34268) |
| 34271 | eval | `3task_sog_mean_iter1k` 全 4 suite（port 8001） | gnho031 | ⏳ PD (after 34270) |

---

## 十、待办事项

1. **补充 3task_sog iter=1 结果**：Job 34268 完成后补充 §5.4 表格（~2.5h）。
2. **补充 3task_sog iter=1k 结果**：Job 34271 完成后补充 §5.4 表格（merge + eval ~5h）。
3. **补充 libero_10 二次 FT 结果**：Job 34269 完成后补充 §7.4 表格。
4. **修复 changed_ base 路径**：在 `wudi_merge.py` 的 `_load_flat()` 中自动追加 `params/` 子路径。
5. **补跑 spatial10_mean_iter300/500 完整 eval**：现有结果缺 goal/libero_10 suite。
