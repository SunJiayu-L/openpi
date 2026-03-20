# `gemma.py` 逐模块讲解（第三篇）

本文对应源码：
- [src/openpi/models/gemma.py](/storage/yukaichengLab/lishiwen/jiayusun/openpi/src/openpi/models/gemma.py)

配套文档：
- [model_py_explained.md](/storage/yukaichengLab/lishiwen/jiayusun/openpi/docs/model_py_explained.md)
- [pi0_py_explained.md](/storage/yukaichengLab/lishiwen/jiayusun/openpi/docs/pi0_py_explained.md)

本文重点回答：`adarms_cond` 在 action expert 里到底怎么起作用。

---

## 1. 文件定位与核心思想

`gemma.py` 是 `pi0.py` 里 `self.PaliGemma.llm` 的实现核心。它不是“单一 Transformer”，而是：

- 一个共享 Attention 的多专家结构（Mixture-of-experts style, 但不是路由 MoE）
- 每个 expert 有自己的 Norm/FFN 参数
- 支持把不同 token 序列喂给不同 expert（例如 prefix 走 PaliGemma expert，suffix 走 action expert）

最关键设计：
- **Attention 层在专家之间按 token 序列拼接后共同计算**
- **Norm + FFN 按 expert 分开参数**
- `adarms_cond` 通过 `RMSNorm(cond)` 进入每个 block 的残差门控

---

## 2. 结构总览（从上到下）

1. `Config` / `get_config`：定义各变体宽度、层数、头数、LoRA 配置
2. `RMSNorm`：普通 RMSNorm + 自适应 adaRMSNorm
3. `Embedder`：词表 embedding 编解码
4. `Attention`：多 expert token 合并后做注意力（支持 KV cache）
5. `FeedForward`：Gemma 风格门控 FFN
6. `Block`：Norm -> Attn -> Residual -> Norm -> FFN -> Residual
7. `Module`：堆叠 `Block`，提供 `embed` 和总前向
8. 辅助函数：`_apply_rope`、`_name`、`_gated_residual`

---

## 3. `Config` 与模型变体

位置：`Config` 在第 45 行，`get_config` 在第 58 行。

`Config` 字段：
- `width`, `depth`, `mlp_dim`
- `num_heads`, `num_kv_heads`, `head_dim`
- `lora_configs`

`get_config` 支持：
- `gemma_2b`
- `gemma_300m`
- 对应的 `*_lora` 版本
- `dummy`（测试）

在 `pi0.py` 里通常会创建两个 config：
- expert 0: PaliGemma 主干
- expert 1: action expert

---

## 4. `RMSNorm`：`adarms_cond` 的注入入口

位置：第 113 行。

### 4.1 两种模式

1. `cond is None`：普通 RMSNorm
- 计算方差
- 归一化
- 乘以可学习 `1 + scale`

2. `cond is not None`：自适应 RMSNorm（adaRMS）
- 用 `Dense(3 * D)` 投影 `cond`
- 切成 `scale, shift, gate`
- 输出：`normed * (1 + scale) + shift`
- 额外返回 `gate`

### 4.2 为什么它关键

在 `Block` 里，这个 `gate` 会进入 `_gated_residual`：
- 普通残差：`x + y`
- 自适应残差：`x + y * gate`

所以 `adarms_cond` 不是只做“偏置”，而是同时控制：
- 归一化后的缩放/平移
- 残差分支强度（门控）

这就是 `pi0.5` 时间条件化的核心实现点。

---

## 5. `Attention`：多专家 token 合并计算 + KV cache

位置：第 158 行。

### 5.1 输入形式
`xs` 是列表：每个 expert 一个 token 张量或 `None`。

示例（在 `pi0.py` 中）：
- 一次完整训练前向：`[prefix_tokens, suffix_tokens]`
- 采样时仅 suffix：`[None, suffix_tokens]`

### 5.2 核心流程

1. 对每个非空 expert 各自做 Q/K/V 投影（支持 LoRA）
2. 把各 expert 的 Q/K/V 在序列维拼接
3. 对 Q/K 应用 RoPE
4. 若有 `kv_cache`，把历史 K/V 拼接进来
5. 计算 masked attention
6. 得到编码后，再按原 expert 序列长度切回去
7. 每个 expert 用自己输出投影 `attn_vec_einsum`

### 5.3 关键意义

虽然有多个 expert，但注意力在“拼接后的统一序列”上做，所以跨 expert 的信息可以通过 attention 交互。

---

## 6. `FeedForward`：门控 MLP

位置：第 253 行。

结构是 Gemma 常见 gated-MLP 形式：
- 两路线性：`ff_gate` 和 `ff1`
- `gelu(ff_gate) * ff1`
- 再线性回投

在 `Block` 里是每个 expert 分开参数、分开计算。

---

## 7. `Block`：adaRMS 生效的具体位置

位置：第 284 行。

每层做两段子结构：

1. Attention 子层
- `pre_attention_norm`：`RMSNorm(x, adarms_cond[i]) -> (x_norm, gate)`
- Attention 前向
- 残差：`_gated_residual(x, attn_out, gate)`

2. FFN 子层
- `pre_ffw_norm`：再次 `RMSNorm(..., adarms_cond[i])`
- FFN 前向
- 残差：`_gated_residual(x, ffn_out, gate)`

所以 `adarms_cond` 在每一层、每个子层都会影响特征流。

---

## 8. `Module`：完整模型封装

位置：第 340 行。

### 8.1 `setup`
- 创建共享词嵌入器 `Embedder`
- 用 `nn.scan` 堆叠 `Block`（深度 = `config.depth`）
- 每个 expert 一个 `final_norm`

### 8.2 `embed(tokens)`
- 只用 expert 0 的词表 embedding（PaliGemma vocab）

### 8.3 `__call__(embedded, positions, mask, adarms_cond, kv_cache)`

关键点：
1. `embedded` 是 expert 列表（可含 `None`）
2. `mask` 被扩展到 attention 期望维度 `[B,1,T,S]`
3. 若不传 `adarms_cond`，默认全 `None`
4. 经过所有层后，再做每个 expert 的 final norm
5. 返回：
   - 每个 expert 的输出序列
   - 更新后的 `kv_cache`

### 8.4 `init(use_adarms)`
是 Linen 习惯下的“方便初始化函数”：
- 用假输入跑一遍 `embed` + `__call__`
- 按 `use_adarms` 决定哪些 expert 初始化 adaRMS 路径参数

---

## 9. 辅助函数

### 9.1 `_apply_rope`（第 424 行）
对 Q/K 应用 RoPE 旋转位置编码。

### 9.2 `_name`（第 443 行）
命名规则：
- expert 0 保持原名（兼容直接加载 PaliGemma 预训练权重）
- expert 1+ 添加后缀 `_1`, `_2`...

这点非常关键：确保“主干可无缝加载，新增 expert 可单独初始化”。

### 9.3 `_gated_residual`（第 453 行）
- 无 gate：`x + y`
- 有 gate：`x + y * gate`

`adarms_cond` 对网络行为的最终控制就落在这里。

---

## 10. 回到你最关心的问题：`adarms_cond` 在 Pi0/Pi0.5 里的实际路径

在 `pi0.py`：
- pi0：`adarms_cond = None`
- pi0.5：`adarms_cond = time_mlp(t)`

传入：
```python
self.PaliGemma.llm(..., adarms_cond=[None, adarms_cond])
```

含义：
- expert 0（语言主干）不使用 adaRMS 条件
- expert 1（action expert）使用时间条件

然后在 `gemma.py` 的每层 `RMSNorm` 里变成 `scale/shift/gate`，并影响 attention/ffn 两段残差。

一句话总结：
`pi0.5` 通过 `adarms_cond` 把时间条件直接注入 action expert 的“归一化与残差门控”。

---

## 11. 建议你下一步怎么读

1. 对照 [pi0.py](/storage/yukaichengLab/lishiwen/jiayusun/openpi/src/openpi/models/pi0.py) 里 `embed_suffix` 看 `adarms_cond` 来源。
2. 再回看本文件 `Block.__call__`，确认条件如何逐层生效。
3. 最后看 [siglip.py](/storage/yukaichengLab/lishiwen/jiayusun/openpi/src/openpi/models/siglip.py)，补全“prefix 图像 token 的来源”。

