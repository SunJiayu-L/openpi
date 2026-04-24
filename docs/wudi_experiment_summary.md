# WUDI 模型融合实验总结

**日期**: 2026-03-24
**项目**: openpi / pi0.5 LIBERO 多任务模型融合

---

## 1. 方法概述

### 算法：WUDI（per-layer 2D + SVD de-centering）

移植自 `MLLMerging/InternVL/internvl_chat/model_merging.py` 的 `wudi_merging2`，针对 pi0.5 参数结构适配。

**核心公式**：`merged = scaling2 × base + scaling × wudi_tv`

**优化流程**（对每个 2D 子块）：
1. 计算各微调模型相对 base 的 task vector
2. 全局平均向量 de-centering：`v_i - avg`
3. 对原始 task vector 做 full SVD → 低秩方向基（low_rank）
4. 对去中心后的向量做 compact SVD → 任务参考向量（taskvector）
5. Adam 优化合并向量，最小化其在各任务低秩子空间的投影能量

### pi0.5 参数 2D 分解（循环 L=18 层，不折叠 L 维）

| 参数 | 原始形状 | 2D 分解方式 |
|------|---------|------------|
| `q_einsum/w` | `(L,N,D,H)` | `reshape(N*D, H)` |
| `kv_einsum/w` | `(L,2,K,D,H)` | split K/V → each `(D, K*H)` |
| `attn_vec_einsum/w` | `(L,N,H,D)` | `reshape(N*H, D)` |
| `gating_einsum` | `(L,2,D,Hff)` | split gate/value → each `(D, Hff)` |
| `linear` | `(L,Hff,D)` | already 2D |

### scope = llm_only：参数处理规则

| 参数组 | 数量 | 处理方式 |
|--------|------|---------|
| `PaliGemma/img/*`（SigLIP vision encoder） | 23 | **保持 base 不动** |
| attn/FFN（q/kv/av/gate/linear，expert0+1） | 10 | **WUDI 优化** |
| 其余（norm/embedding/action_head/time_mlp） | ~18 | **简单平均 task vector** |

### 实现文件

- `scripts/wudi_merge.py` — 主融合脚本（含 wandb loss 监控）
- `wudi_merge_5task.sbatch` — Slurm 融合作业（gpu:2, 160G）
- `eval_wudi_5task.sbatch` — Slurm 评测作业（4 suites × 10 tasks × 20 trials）

---

## 2. 核心发现：base 模型选择是关键

### base = pi05_base（预训练，无 LIBERO 能力）→ 全部失败

无论 scaling / scaling2 / scope 如何组合，成功率均为 **0%**。

| 配置 | scope | scaling | scaling2 | 结果 |
|------|-------|---------|----------|------|
| 2-model (libero+goal) 30k | expert1_only | 1.0 | 1.0 | 0% |
| 4-model 30k | expert1_only | 0.1 | 1.0 | 0% |
| 4-model 30k | llm_only | 0.1 | 1.0 | 0% |
| 4-model 30k | llm_only | 0.1~0.5 | 1.0 | 0% |
| 4-model 30k | llm_only | 0.8 | 0.2 | 0% |

**根本原因**：`merged_tv` 是 task vector（delta），不是完整权重。`scaling2 × base_val + scaling × merged_tv` 中 base 本身不具备 LIBERO 动作生成能力，delta 无论如何缩放都无法弥补。

### base = pi05_libero（全量 LIBERO 微调）→ 有效

| 配置 | ft | scope | scaling | 结果 |
|------|-----|-------|---------|------|
| 4-model 30k | spatial/object/goal/10 | llm_only | **0.1** | ✅ **96.8%** |
| 4-model 30k | spatial/object/goal/10 | llm_only | 0.5 | ❌ ~12% |

---

## 3. 最佳配置结果

**Checkpoint**: `checkpoints/merged/wudi_llm_5task_30k_s01`
**配置**: base=pi05_libero/29999，ft=spatial/object/goal/10，scope=llm_only，scaling=0.1，scaling2=1.0，iter=300

### 各 Suite 每任务成功率（20 trials/task）

| Suite | T0 | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 | **avg** |
|-------|----|----|----|----|----|----|----|----|----|----|---------|
| libero_spatial | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | **100%** |
| libero_object  | 1.0 | 1.0 | 1.0 | 0.95| 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | **99.5%** |
| libero_goal    | 0.9 | 1.0 | 1.0 | 0.7 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.95| **95.5%** |
| libero_10      | 1.0 | 1.0 | 0.95| 0.95| 0.95| 1.0 | 0.95| 0.95| 0.95| 0.5 | **92.0%** |
| **Overall**    |    |    |    |    |    |    |    |    |    |    | **🏆 96.8%** |

### 对比：pi05_libero 单模型全量微调基线

| Step | libero_10 | libero_goal | libero_object | libero_spatial | avg |
|------|-----------|-------------|---------------|----------------|-----|
| 5000 | 91.5% | 98.5% | 99.0% | 97.0% | 96.5% |
| 29999| 94.0% | 97.5% | 99.0% | 97.5% | **97.0%** |

**结论**：WUDI 融合（96.8%）与 pi05_libero 全量多任务训练（97.0%）基本持平，差距 0.2%。融合有效但未超越 oracle。

---

## 4. scaling 消融

| scaling | base | 结果 |
|---------|------|------|
| 1.0 | pi05_libero | 未测试 |
| **0.1** | pi05_libero | ✅ **96.8%** |
| 0.5 | pi05_libero | ❌ 12% |
| 0.1~0.5 | pi05_base | ❌ 0% |
| 0.8 (scaling2=0.2) | pi05_base | ❌ 0% |

scaling=0.1 是目前唯一有效值，scaling ≥ 0.5 会严重破坏 base 的原有能力。

---

## 5. 讨论

### 为什么 base=pi05_base 全部失败？

WUDI 在 ViT/RoBERTa 上有效，是因为 base（ImageNet/BERT 预训练）本身已具备特征提取能力，只需叠加任务 delta。但 VLA 的情况不同：

- pi0.5 使用 flow matching 生成连续动作，动作空间高度任务依赖
- pi05_base 对 LIBERO 环境一无所知，无法产生任何合理动作
- task vector 是相对 base 的小 delta，必须在 base 能力基础上才有意义

### 为什么 scaling=0.1 有效，0.5 失败？

base=pi05_libero 已经会所有 LIBERO 任务（97%），task vector 的幅度相对 base 很小。scaling=0.1 只是轻微调整权重，不破坏已有能力；scaling=0.5 幅度过大，严重干扰 base。

### 实验局限性

当前设置本质上是：**在一个已经会所有任务的模型上，加上 4 个子任务专家的微小扰动**。这与经典模型融合场景（多个单任务专家 → 多面手）不同。真正的挑战是：能否从 4 个单任务专家（不依赖全量多任务模型）直接融合出接近 97% 的结果？当前框架下无法实现，需要更强的 base 或不同的融合范式。

---

## 6. 下一步规划

1. **Simple average baseline**：`merged = mean(ft_spatial, ft_object, ft_goal, ft_10)`（不经过 WUDI，直接平均权重），验证 WUDI 相对简单平均的实际增益
2. **wandb loss 分析**：查看 300 步 Adam 优化的 loss 曲线，判断是否已收敛；若未收敛可增加 iter 至 500/1000
3. **5k/15k eval 结果**：补全不同训练步数的融合效果对比
4. **论文对比**：与 ViT 8-task WUDI 结果（ICML 2025）对比，分析 VLA 融合的特殊性
