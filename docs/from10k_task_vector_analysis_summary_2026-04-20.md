# from10k Task Vector 分析总结（2026-04-20，更新至 Joint 对照 + 保护层构建）

## 1. 分析目标与设置

目标：解释为什么 `2task mean` 下 `libero_10` 保持较高，而 `4task mean` 下明显下降。  
统一设置：

- 基座：`checkpoints/pi05_libero/my_experiment/10000`
- 四个单任务 FT（20k）：
  - `pi05_libero_10_from_pi05libero_10k/.../20000`
  - `pi05_libero_spatial_from_pi05libero_10k/.../20000`
  - `pi05_libero_goal_from_pi05libero_10k/.../20000`
  - `pi05_libero_object_from_pi05libero_10k/.../20000`
- scope：`llm_only`（与 `scripts/wudi_merge.py` 对齐）

---

## 2. 已完成分析与结果

### 2.1 全局 pairwise cosine（A 实验）

产物：

- `docs/analysis/from10k_llm_only_global_cosine/global_cosine_6pairs.csv`
- `docs/analysis/from10k_llm_only_global_cosine/global_cosine_matrix_4x4.csv`

结果（6 对）：

- `cos(Δ10, Δsp) = 0.048576`
- `cos(Δ10, Δobj) = 0.052842`
- `cos(Δ10, Δgoal) = 0.049358`
- `cos(Δsp, Δobj) = 0.052164`
- `cos(Δsp, Δgoal) = 0.054143`
- `cos(Δobj, Δgoal) = 0.048924`

结论：四个任务向量全局上都近似正交（约 `0.05`），`libero_10` 并未在全局上显著“更孤立”。

### 2.2 retain rate（全局 + 分模块）

产物：

- `docs/analysis/from10k_llm_only_retain_rate/retain_rate_mean_2task_4task.csv`
- `docs/analysis/from10k_llm_only_retain_rate/retain_rate_by_module.csv`
- `docs/analysis/from10k_llm_only_retain_rate/retain_rate_drop_summary.csv`

定义：

- `Δm2 = (Δ10 + Δsp)/2`
- `Δm4 = (Δ10 + Δsp + Δobj + Δgoal)/4`
- `r_t(Δm) = <Δm,Δt> / ||Δt||²`

全局 retain 结果：

- `libero_10`: `r2 = 0.524384`, `r4 = 0.287673`, `drop = -0.236710`
- `spatial`: `r2 = 0.524193`, `r4 = 0.288498`, `drop = -0.235694`
- `object`: `r4 = 0.288611`
- `goal`: `r4 = 0.288223`

结论：`4task mean` 对各任务投影保留率存在近乎均匀压缩（约 `0.52 -> 0.29`）。

分模块（`libero_10`）`drop_r10 = r10_4task - r10_2task`：

- `gating_einsum`: `-0.240992`（最大）
- `linear`: `-0.239106`
- `q_einsum`: `-0.220163`
- `attn_vec_einsum`: `-0.211069`
- `kv_einsum`: `-0.201511`

结论：压缩最严重的是 FFN（`gating_einsum`, `linear`）。

### 2.3 FFN 分层 retain rate（gating + linear）

产物：

- `docs/analysis/from10k_llm_only_ffn_layerwise_retain_rate/layerwise_retain_rate_ffn.csv`
- `docs/analysis/from10k_llm_only_ffn_layerwise_retain_rate/layerwise_drop_summary_r10.csv`
- `docs/analysis/from10k_llm_only_ffn_layerwise_retain_rate/region_summary_r10.csv`
- `docs/analysis/from10k_llm_only_ffn_layerwise_retain_rate/candidate_protected_layers.csv`

按区域汇总（`drop_r10 = r10_4task - r10_2task`）：

- `gating_einsum`
  - front (0-5): `-0.240759`
  - middle (6-11): `-0.242239`（最重）
  - back (12-17): `-0.224734`（明显减轻）
- `linear`
  - front (0-5): `-0.240556`
  - middle (6-11): `-0.240363`
  - back (12-17): `-0.220550`（明显减轻）

关键层观察：

- 两个 FFN 模块在前中层呈“大面积高平台”（多数层约 `-0.24`）
- 并非“少数尖峰层主导”
- 最后两层（16/17）显著减轻，尤其 layer 17（`gating: -0.1581`, `linear: -0.1675`）

自动候选保护层（top-3 per module）：

- `gating_einsum`: layers `11, 10, 6`
- `linear`: layers `1, 4, 2`

### 2.4 Joint 目标解对照 + Repair Map（新增）

本轮新增两类试验：

1. **保护层 checkpoint 构建**（从 `mean4_iter0` 出发）  
2. **以 `joint@29999` 为目标解的几何/差分分析**

#### 2.4.1 保护层 checkpoint 构建

脚本：

- `scripts/gen_selective_merge_from_mean4.py`

策略严格分开：

- `*_joint`：保护层借用 `joint@29999` 对应层
- `*_base`：保护层回退到 `base@10000` 对应层

已生成 checkpoint：

- `checkpoints/ablation_selective_protect/mean4_protect_minimal_joint`
- `checkpoints/ablation_selective_protect/mean4_protect_region_joint`
- `checkpoints/ablation_selective_protect/mean4_protect_minimal_base`
- `checkpoints/ablation_selective_protect/mean4_protect_region_base`

#### 2.4.2 joint 对齐几何分析（含 iter100）

产物目录：

- `docs/analysis/joint_target_alignment_2026-04-20/`

关键文件：

- `geometry_alignment_to_joint_global.csv`
- `geometry_alignment_to_joint_by_module.csv`
- `geometry_alignment_to_joint_ffn_layer.csv`
- `repair_module_norm.csv`
- `repair_ffn_layer_norm.csv`
- `repair_vs_drop_correlation.csv`

关键结果（global）：

- `mean4_iter0`: `cos_to_joint = 0.1731`
- `mean4_iter100`: `cos_to_joint = 0.1687`（比 iter0 更差）
- `ft_from_500`: `cos_to_joint = 0.1532`
- `ft_from_1k`: `cos_to_joint = 0.1371`

注：这里显示“更长 FT checkpoint 并不等于更接近 joint@29999（以 base@10000 为参照）”，说明 joint 与 merge+FT 未必在同一几何目标上收敛。

Repair 模块占比（`joint - mean4_iter0`）：

- `gating_einsum`: `0.6164`
- `linear`: `0.2995`
- FFN 合计约 `0.9159`

Repair vs Drop 相关性（FFN 分层）：

- `gating_einsum`: Pearson `0.9262`, Spearman `0.7874`
- `linear`: Pearson `0.9532`, Spearman `0.5521`
- `ffn_all`: Pearson `0.7781`, Spearman `0.8427`

结论：retain-rate 找到的“受损层”与 `joint - mean` 的真实修复需求高度一致。

### 2.5 Joint 对照下的 selective protection 评测（新增，进行中）

在 `mean4_iter0` 上构建了 4 个保护权重（已完成）：

- `checkpoints/ablation_selective_protect/mean4_protect_minimal_joint`
- `checkpoints/ablation_selective_protect/mean4_protect_region_joint`
- `checkpoints/ablation_selective_protect/mean4_protect_minimal_base`
- `checkpoints/ablation_selective_protect/mean4_protect_region_base`

其中：

- `*_joint`：保护层替换为 `joint@29999` 对应层（目标解注入）
- `*_base`：保护层替换为 `base@10000` 对应层（错误更新消融）

评测状态：

- 已生成并提交四个 all-suites 评测脚本（节点 `gnho034` + `gnho018`）
- 首轮失败原因是新 checkpoint 缺少 `assets/norm_stats.json`
- 已补齐 `assets` 后重提，目前作业运行中：
  - `33993` (`mean4_protect_region_joint`)
  - `33994` (`mean4_protect_minimal_joint`)
  - `33995` (`mean4_protect_minimal_base`)
  - `33996` (`mean4_protect_region_base`)

---

## 3. 当前可证实机制结论（更新）

1. `libero_10` 在 4-task 下变差，不是“全局方向更离群”造成的。  
2. 主机制是：在近正交任务向量几何下，等权 4-task mean 将每个任务的保留投影系统性压缩至约 `0.29`。  
3. 性能敏感压缩主要集中在 FFN（`gating/linear`），提示 task-specific specialization 更多存储在 FFN，而非 attention。
4. 分层上更准确的模式是：**前中层广泛且强烈压缩，后层（尤其最后两层）明显减弱**，而非“仅中后层最关键”。
5. 以 `joint@29999` 为目标解时，`joint - mean4_iter0` 的修复量约 91.6% 集中在 FFN，且与层级 `drop_r10` 高度相关，进一步支持“FFN 前中层是主修复区域”。

---

## 4. 与现有 eval 现象的对应

- 已观测：`2task mean` 中 `libero_10` 仍高；`4task mean` 明显下降。  
- 现在几何解释已闭环：
  - 全局 cosine 约 `0.05`（近正交）
  - retain 从 `~0.52` 到 `~0.29` 的下降与 2task/4task 性能差异一致
  - 分模块显示 FFN 为主要压缩点
  - 分层显示 FFN 压缩是“前中层高平台 + 末层减弱”结构

---

## 5. 下一步（高优先级）

完成并汇总 selective merge 评测验证（不再做全局统计）：

1. 最小保护集（PoC）
   - `gating`: `6, 10, 11`
   - `linear`: `1, 2, 4`
2. 区域保护集（主推荐）
   - `gating`: `6-11`
   - `linear`: `1-11`
3. 模块消融对照
   - 只保护 `gating`
   - 只保护 `linear`
   - 同时保护两者
4. 核心 baseline 补充
   - `4task_mean_iter100`（已纳入 joint 对齐，评测侧也应纳入）

评估重点：

- `libero_10` 回升幅度
- 其余 suites 是否显著受损
- 是否存在“较小保护集即可恢复大部分 drop”的拐点
- `*_joint` vs `*_base`：区分“目标解注入”与“错误更新消融”

---

## 6. Joint 对照试验流程图

```text
[base@10000] -----------------------------+
                                          |
                                          v
                             定义目标解 [joint@29999]
                                          |
                                          v
                           几何对齐分析（global/module/layer）
                         compare: mean0 / mean100 / ft500 / ft1k
                                          |
                                          v
                     Repair Map: Δrepair = joint - mean0
                                          |
                                          v
                  与历史drop_r10做相关性（验证“受损层=修复层”）
                                          |
                                          v
                生成4个保护权重（从mean0出发，分两类策略）
                 /                                  \
                /                                    \
   borrow-joint（目标解注入）              fallback-base（错误更新消融）
   - minimal_joint                          - minimal_base
   - region_joint                           - region_base
                \                                    /
                 \                                  /
                  +----------> 四模型全套件评测 <---+
                               (gnho034 + gnho018)
                                          |
                                          v
                 对比结论：joint注入 vs base回退，minimal vs region
```

## 7. 运行记录（本轮）

- 全局 cosine CPU 作业：`job 33979`（COMPLETED）
- retain rate CPU 作业：`job 33980`（COMPLETED）
- FFN 分层 retain rate CPU 作业：`job 33981`（COMPLETED）
- 保护层 checkpoint 构建 CPU 作业：`job 33983`（COMPLETED）
- joint 对齐/repair 分析 CPU 作业：`job 33985`（COMPLETED）
