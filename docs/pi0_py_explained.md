# `pi0.py` 逐函数讲解（第二篇）

本文对应源码：
- [src/openpi/models/pi0.py](/storage/yukaichengLab/lishiwen/jiayusun/openpi/src/openpi/models/pi0.py)

配套前置文档：
- [model_py_explained.md](/storage/yukaichengLab/lishiwen/jiayusun/openpi/docs/model_py_explained.md)

目标：你读完这篇后，能完整理解 `Pi0` 在训练与推理两条路径上“每个函数在做什么”。

---

## 1. 先记住整体结构

`pi0.py` 可以按函数分成 7 块：

1. `make_attn_mask`（第 31 行）
2. `posemb_sincos`（第 65 行）
3. `Pi0.__init__`（第 98 行）
4. `Pi0.embed_prefix`（第 153 行）
5. `Pi0.embed_suffix`（第 202 行）
6. `Pi0.compute_loss`（第 275 行）
7. `Pi0.sample_actions`（第 320 行）

其中最关键的是三件事：
- prefix 怎么构造（图像+文本）
- suffix 怎么构造（状态/动作/时间）
- flow matching 的训练目标与采样积分怎么实现

---

## 2. `make_attn_mask(input_mask, mask_ar)`

位置：第 31 行。

### 2.1 输入
- `input_mask: bool[B, N]`：哪些 token 有效（不是 padding）
- `mask_ar: bool[?B, N]`：分块/因果控制信号

### 2.2 核心逻辑
1. 先把 `mask_ar` broadcast 到 batch 维。
2. 对 `mask_ar` 做 `cumsum`，得到每个 token 的“块编号”。
3. 构造 `attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]`。
4. 再与 `valid_mask`（由 `input_mask` 构造）相与。

### 2.3 直觉
- `True` 会“开启新块”。
- 同块内可全互看，后块可看前块，前块不能看后块。
- 再叠加 padding 屏蔽。

---

## 3. `posemb_sincos(pos, embedding_dim, min_period, max_period)`

位置：第 65 行。

### 3.1 作用
把标量时间 `t`（或一般位置标量）映射成固定维度的 sin/cos 向量。

### 3.2 关键步骤
1. 维度必须是偶数（sin/cos 对）。
2. 生成从低频到高频的几何级数周期。
3. 用外积得到每个样本对每个频率的相位。
4. 拼接 `sin` 和 `cos`。

### 3.3 在 Pi0 里的用途
给扩散/流匹配时间步 `t` 编码，后续注入动作专家（pi0 与 pi0.5 的注入方式不同）。

---

## 4. `Pi0.__init__(config, rngs)`

位置：第 98 行。

### 4.1 构建了什么
1. 读取 `paligemma_config` 与 `action_expert_config`
2. 构建 Gemma 双专家模块（语言主干 + action expert）
3. 构建 SigLIP 图像编码器
4. 构建动作投影层：
   - `action_in_proj`
   - `action_out_proj`
5. 根据 `config.pi05` 分叉：
   - pi0.5：`time_mlp_in/out`（时间条件给 adaRMS）
   - pi0：`state_proj` + `action_time_mlp_in/out`（动作+时间显式融合）

### 4.2 这里决定了 pi0 和 pi0.5 的核心差异
- pi0：状态是 suffix 里的连续 token，时间与动作拼接后喂 expert
- pi0.5：状态通常离散进 prompt，时间走 adaRMS 条件，不和动作 token 直接拼接

---

## 5. `embed_prefix(obs)`

位置：第 153 行。

### 5.1 输入
`Observation`（来自 `model.py` 预处理后）。

### 5.2 输出
返回三件套：
- `tokens: float[B, S_prefix, E]`
- `input_mask: bool[B, S_prefix]`
- `ar_mask: bool[S_prefix]`

### 5.3 处理流程
1. 遍历 `obs.images`，每路图像进 SigLIP 得到 image tokens。
2. 把图像 mask 扩到 token 级别。
3. 每个图像 token 在 prefix 中都标成 `ar_mask=False`（同块全互看）。
4. 若有 `tokenized_prompt`，调用 `llm(..., method="embed")` 得到文本 embedding。
5. 文本 token 也并入 prefix，且同样 `ar_mask=False`。

### 5.4 结果语义
prefix = 图像 token + 文本 token。
它们彼此全互看，不做因果分割。

---

## 6. `embed_suffix(obs, noisy_actions, timestep)`

位置：第 202 行。

这是全文件最关键函数之一。

### 6.1 输入
- `obs`
- `noisy_actions: x_t`（训练时）或当前轨迹点（采样时）
- `timestep: t`

### 6.2 输出
- `tokens: float[B, S_suffix, E]`
- `input_mask: bool[B, S_suffix]`
- `ar_mask: bool[S_suffix]`
- `adarms_cond: float[B, E] | None`

### 6.3 pi0 分支（`not self.pi05`）
1. `state_proj(obs.state)` 生成一个 `state token`。
2. `action_in_proj(noisy_actions)` 生成动作 token。
3. `posemb_sincos(t)` 生成时间向量，再复制到每个动作步。
4. 动作 token 与时间 token 在特征维拼接后过 MLP。
5. `adarms_cond = None`。

### 6.4 pi0.5 分支（`self.pi05`）
1. 不添加连续 `state token`（状态信息通常已离散进文本 prompt）。
2. `action_in_proj(noisy_actions)` 作为动作 token。
3. `posemb_sincos(t)` 后经 `time_mlp_in/out + swish`。
4. 时间不与动作拼接，而是作为 `adarms_cond` 传给 action expert。

### 6.5 suffix 的 attention 规则
`ar_mask += [True] + [False] * (action_horizon - 1)`
表示 action 序列是一个因果块（第一位开块）。

---

## 7. `compute_loss(rng, observation, actions, train=False)`

位置：第 275 行。

这是训练主入口。

### 7.1 先做数据与随机准备
1. 拆分 RNG：预处理、噪声、时间。
2. `preprocess_observation(...)`。
3. 采样噪声 `noise ~ N(0, I)`。
4. 采样时间 `t ~ Beta(1.5, 1)`，并限制在 `(0.001, 1)`。

### 7.2 构造 flow matching 目标
- `x_t = t * noise + (1 - t) * actions`
- `u_t = noise - actions`

模型要学的是：给定 `x_t, t, condition` 预测速度场 `v_t`，逼近 `u_t`。

### 7.3 前向
1. `embed_prefix(observation)`
2. `embed_suffix(observation, x_t, t)`
3. 拼接 mask，构造 `attn_mask`
4. 调 LLM：`self.PaliGemma.llm([prefix_tokens, suffix_tokens], ...)`
5. 从 `suffix_out` 末尾取 action 段，过 `action_out_proj` 得 `v_t`

### 7.4 损失
`loss = mean((v_t - u_t)^2, axis=-1)`

即动作维上的 MSE（保留 batch 与 horizon 维）。

---

## 8. `sample_actions(rng, observation, num_steps=10, noise=None)`

位置：第 320 行。

这是推理/控制时的采样入口。

### 8.1 核心思想
从 `x_1 ~ N(0, I)` 出发，沿着学到的速度场反向积分到 `x_0`。

代码里采用：
- `t=1` 是噪声端
- `t=0` 是目标动作端
- `dt = -1/num_steps`

### 8.2 KV cache 优化
1. 先只跑 prefix 一次，得到 `kv_cache`。
2. 迭代时每步只送 suffix，利用 cache 避免重复计算 prefix。

### 8.3 每一步 `step(carry)`
carry 是 `(x_t, t)`。

每步做：
1. 用当前 `x_t, t` 调 `embed_suffix`
2. 构造 suffix 对 prefix+suffix 的联合 attention mask
3. 调 LLM（prefix 走 cache）得到 `suffix_out`
4. 投影得 `v_t`
5. Euler 更新：`x_{t+dt} = x_t + dt * v_t`

### 8.4 停止条件
`cond` 用 `time >= -dt / 2`，给浮点误差留容差。

最终返回 `x_0` 作为动作序列。

---

## 9. 一张训练/推理对照表

| 维度 | 训练 `compute_loss` | 推理 `sample_actions` |
|---|---|---|
| 输入动作 | 真值 actions | 初始噪声或给定 noise |
| 时间 | 随机采样 t | 从 1 迭代到 0 |
| 目标 | 拟合 `u_t = noise-actions` | ODE 离散积分生成动作 |
| 前向 | prefix+suffix 一次大前向 | prefix 预填缓存 + suffix 多步前向 |
| 输出 | loss | 采样动作 `x_0` |

---

## 10. pi0 与 pi0.5 最关键差异（在本文件里的落点）

1. `__init__`
- pi0: `state_proj` + `action_time_mlp_*`
- pi0.5: `time_mlp_*`

2. `embed_suffix`
- pi0: 状态 token 在 suffix；时间和动作显式拼接
- pi0.5: 状态通常已进文本；时间走 `adarms_cond`

3. `llm` 调用
- 都传 `adarms_cond=[None, adarms_cond]`
- pi0 时 `adarms_cond=None`
- pi0.5 时为时间条件向量

---

## 11. 读源码时的最小心智模型

你可以把 `Pi0` 看成：
1. 把多模态条件编码成 prefix（图像+文本）
2. 把待去噪动作轨迹编码成 suffix（含时间条件）
3. 让 action expert 预测速度场
4. 训练时做监督（Flow Matching），推理时做数值积分（Euler）

只要抓住这 4 步，细节就不容易迷路。

---

## 12. 建议下一步

如果你要继续深挖，建议按这个顺序看：
1. `src/openpi/models/pi0_config.py`：配置如何影响上述分支
2. `src/openpi/models/gemma.py`：`adarms_cond` 在 action expert 内部如何作用
3. `src/openpi/models/siglip.py`：图像 token 的来源与形状

