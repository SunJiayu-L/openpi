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
3. 最后看 [siglip.py](/storage/yukaichengLab/lishiwen/jiayusun/openpi/src/openpi/models/siglip.py)，补全”prefix 图像 token 的来源”。

---

## 12. `Attention.__call__` 逐行完整图解

> 本节对照用户提供的带注释代码，将 `Attention.__call__` 的 **每一行** 都展示为张量形状变换图。
>
> 使用 pi0 真实配置做具体数字标注：
> - **Expert 0（PaliGemma）**：`width=2048, num_heads=8, num_kv_heads=1, head_dim=256`，序列长 `S0=816`
> - **Expert 1（Action Expert）**：`width=1024, num_heads=8, num_kv_heads=1, head_dim=256`，序列长 `S1=51`
> - 两者都走 **GQA 分支**（`num_kv_heads=1 < num_heads=8`），`G=8`（每个 KV 头对应 8 个 Q 头）
>
> 图中给出两种场景下的形状：
> - **训练**：`xs = [prefix(B,816,2048), suffix(B,51,1024)]`，无 kv_cache
> - **推理**：`xs = [None, suffix(B,51,1024)]`，有 kv_cache（前缀 KV 已缓存）

---

### 12.1 整体数据流一览

```
xs[0] = prefix_tokens      xs[1] = suffix_tokens        (训练)
(B, 816, 2048)              (B, 51, 1024)
        │                          │
        │  ══ PHASE 1: Q/K/V 投影 ══════════════════════════════
        │                          │
        ▼  q_einsum “BTD,NDH->BTNH”  ▼  q_einsum “BTD,NDH->BTNH”
   Q0(B,816,8,256)            Q1(B,51,8,256)
        │  kv_einsum “BSD,2KDH->2BSKH”  │  kv_einsum “BSD,2KDH->2BSKH”
   K0(B,816,1,256)            K1(B,51,1,256)
   V0(B,816,1,256)            V1(B,51,1,256)
        │                          │
        │  ══ PHASE 2: 序列拼接 ══════════════════════════════
        │                          │
        └──── concat(axis=1) ───────┘
                    │
              Q(B,867,8,256)
              K(B,867,1,256)
              V(B,867,1,256)
                    │
        ══ PHASE 3: RoPE + Scale ══════════════════════════
                    │
              Q_rope(B,867,8,256)  × head_dim^(-0.5) = × 0.0625
              K_rope(B,867,1,256)
                    │
        ══ PHASE 4: KV Cache（仅推理）══════════════════════
                    │
              K_full(B, S_prefix+867, 1, 256)   ← cache_k 在前
              V_full(B, S_prefix+867, 1, 256)
                    │
        ══ PHASE 5: GQA logits ═════════════════════════════
                    │
              q rearrange: “B T (K G) H -> B T K G H”
              Q_gqa(B,867,1,8,256)   K=1, G=8
                    │
              logits = einsum(“BTKGH,BSKH->BKGTS”)
              (B, 1, 8, 867, 867)    float32
                    │
        ══ PHASE 6: Mask + Softmax ════════════════════════
                    │
              masked_logits = where(attn_mask[:,None,:,:], logits, -2.38e38)
              probs = softmax(masked_logits, axis=-1).astype(dtype)
              (B, 1, 8, 867, 867)    bfloat16
                    │
        ══ PHASE 7: 加权聚合 ════════════════════════════════
                    │
              encoded = einsum(“BKGTS,BSKH->BTKGH”)
              (B, 867, 1, 8, 256)
                    │
              encoded rearrange: “B T K G H -> B T (K G) H”
              (B, 867, 8, 256)
                    │
        ══ PHASE 8: 切片 + 输出投影 ═════════════════════════
                    │
        ┌───────────┴──────────────┐
  encoded[:,0:816]          encoded[:,816:867]
  (B,816,8,256)             (B,51,8,256)
        │                          │
  attn_vec_einsum           attn_vec_einsum_1
  “BTNH,NHD->BTD”           “BTNH,NHD->BTD”
  (8,256,2048)              (8,256,1024)
        │                          │
  out[0](B,816,2048)        out[1](B,51,1024)
        │                          │
        └──────────────────────────┘
                    │
        return [out[0], out[1]], (K_full, V_full)
```

---

### 12.2 逐行代码 → 形状变换对照表

#### ① 前置断言（3 行）

```python
assert all(config.head_dim == self.configs[0].head_dim for config in self.configs)
# 检查: gemma_2b.head_dim=256  == gemma_300m.head_dim=256  ✓

assert all(config.num_heads == self.configs[0].num_heads for config in self.configs)
# 检查: gemma_2b.num_heads=8   == gemma_300m.num_heads=8   ✓

assert all(config.num_kv_heads == self.configs[0].num_kv_heads for config in self.configs)
# 检查: gemma_2b.num_kv_heads=1 == gemma_300m.num_kv_heads=1 ✓
```

> **意义**：共享注意力的前提——两个 expert 必须有相同的 head 结构，否则 Q 拼接后无法对齐。

---

#### ② 获取 dtype（1 行）

```python
dtype = next(x.dtype for x in xs if x is not None)
# dtype = bfloat16
# 从第一个非 None 的输入拿 dtype，后续 softmax 结果会 cast 回此 dtype
```

```
xs[0] = prefix(B,816,2048)  dtype=bfloat16
xs[1] = suffix(B,51,1024)   dtype=bfloat16
          ↓ next(x for x if x is not None)
dtype = bfloat16
```

---

#### ③ Q/K/V 投影（GQA 分支，每个 expert 各自独立）

两个 expert 的 `num_kv_heads=1 ≠ num_heads=8`，走 GQA 分支：

```python
# ── Expert 0（PaliGemma, i=0）──────────────────────────────────────────
q_einsum = lora.Einsum(
    shape=(config.num_heads, config.width, config.head_dim),  # (8, 2048, 256)
    name=_name(“q_einsum”, 0),   # → “q_einsum”  (无后缀，兼容预训练权重)
    ...
)
q = q_einsum(“BTD,NDH->BTNH”, x)
#   x: (B, 816, 2048)
#   weight: (8, 2048, 256)
#   q: (B, 816, 8, 256)
#   einsum 含义：每个位置的 D=2048 维 token，投影到 N=8 个 head，每个 head H=256 维

kv_einsum = lora.Einsum(
    shape=(2, config.num_kv_heads, config.width, config.head_dim),  # (2, 1, 2048, 256)
    name=_name(“kv_einsum”, 0),  # → “kv_einsum”
    ...
)
k, v = kv_einsum(“BSD,2KDH->2BSKH”, x)
#   x: (B, 816, 2048)
#   weight: (2, 1, 2048, 256)
#   结果: (2, B, 816, 1, 256)
#   k: (B, 816, 1, 256)   ← K 头只有 1 个
#   v: (B, 816, 1, 256)

qkvs.append((q, k, v))
# qkvs[0] = ( (B,816,8,256), (B,816,1,256), (B,816,1,256) )


# ── Expert 1（Action Expert, i=1）──────────────────────────────────────
q_einsum = lora.Einsum(
    shape=(config.num_heads, config.width, config.head_dim),  # (8, 1024, 256)
    name=_name(“q_einsum”, 1),   # → “q_einsum_1”
    ...
)
q = q_einsum(“BTD,NDH->BTNH”, x)
#   x: (B, 51, 1024)
#   weight: (8, 1024, 256)
#   q: (B, 51, 8, 256)

kv_einsum = lora.Einsum(
    shape=(2, config.num_kv_heads, config.width, config.head_dim),  # (2, 1, 1024, 256)
    name=_name(“kv_einsum”, 1),  # → “kv_einsum_1”
    ...
)
k, v = kv_einsum(“BSD,2KDH->2BSKH”, x)
#   x: (B, 51, 1024)
#   weight: (2, 1, 1024, 256)
#   k: (B, 51, 1, 256)
#   v: (B, 51, 1, 256)

qkvs.append((q, k, v))
# qkvs[1] = ( (B,51,8,256), (B,51,1,256), (B,51,1,256) )
```

---

#### ④ 各 expert 的 Q/K/V 在序列维拼接（1 行）

```python
q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))
```

```
zip(*qkvs) 将 qkvs 按 (q,k,v) 分组：

  zip(*qkvs) = [
    (Q0(B,816,8,256),  Q1(B,51,8,256)),   ← 第0组: 所有 expert 的 Q
    (K0(B,816,1,256),  K1(B,51,1,256)),   ← 第1组: 所有 expert 的 K
    (V0(B,816,1,256),  V1(B,51,1,256)),   ← 第2组: 所有 expert 的 V
  ]

  concat(Q0, Q1, axis=1): (B,816,8,256) + (B,51,8,256) → Q(B,867,8,256)
  concat(K0, K1, axis=1): (B,816,1,256) + (B,51,1,256) → K(B,867,1,256)
  concat(V0, V1, axis=1): (B,816,1,256) + (B,51,1,256) → V(B,867,1,256)

  ┌─────────────────────────────────┐
  │  Q (B, 867, 8, 256)             │
  │  ├── prefix Q ──── 816 tokens   │
  │  └── suffix Q ─── 51 tokens     │
  └─────────────────────────────────┘
  S_total = S0 + S1 = 816 + 51 = 867
```

---

#### ⑤ RoPE + Scale（3 行）

```python
q = _apply_rope(q, positions=positions)
#   内部: freq = 10000^(2i/256), i=0..127
#         radians = positions / freq   → (B, 867, 1, 128) broadcast
#         [x1, x2] = split(q, axis=-1) → 各(B,867,8,128)
#         result = [x1*cos-x2*sin, x2*cos+x1*sin]
#   输入: q(B,867,8,256), positions(B,867)
#   输出: q(B,867,8,256)  float32 (RoPE 强制 float32)

q *= self.configs[0].head_dim ** -0.5
#   head_dim=256, 256^(-0.5) = 0.0625
#   q(B,867,8,256) 每个元素乘以 0.0625（scaled dot-product attention 的缩放）

k = _apply_rope(k, positions=positions)
#   输入: k(B,867,1,256), positions(B,867)
#   输出: k(B,867,1,256)  float32

assert q.dtype == k.dtype == v.dtype == dtype
#   此处仍应为 bfloat16（_apply_rope 内 float32 临时计算，返回时 cast 回）
```

```
positions(B,867): 每个 token 的位置 ID
  前 816 位：prefix 的位置（图像/文本）
  后  51 位：suffix 的位置（从 prefix_len 开始计数）
```

---

#### ⑥ KV Cache 拼接（推理专用，3 行）

```python
if kv_cache is not None:
    cache_k, cache_v = kv_cache
    k = jnp.concatenate([cache_k, k], axis=1)
    v = jnp.concatenate([cache_v, v], axis=1)
```

```
训练时（kv_cache=None）:
  ┌──────────────────────────────────────┐
  │  跳过，K/V 只含当前 step 的内容         │
  │  K(B, 867, 1, 256)                   │
  │  V(B, 867, 1, 256)                   │
  └──────────────────────────────────────┘

推理时（kv_cache != None, xs=[None, suffix]）:
  ┌──────────────────────────────────────────────────────┐
  │  cache_k: (B, S_prefix, 1, 256)  ← 已缓存的 prefix KV│
  │  cache_v: (B, S_prefix, 1, 256)                      │
  │  当前 k:  (B, 51, 1, 256)        ← 仅 suffix（xs[0]=None）│
  │                                                      │
  │  concat(axis=1):                                     │
  │  K_full: (B, S_prefix+51, 1, 256)                   │
  │  V_full: (B, S_prefix+51, 1, 256)                   │
  │          ├─── prefix 部分 (S_prefix) ──┤             │
  │          └─── suffix 部分 (51) ────────┘             │
  └──────────────────────────────────────────────────────┘

  S_key = S_prefix + 51    （S_query = 51，asymmetric attention）
```

---

#### ⑦ GQA Rearrange + Attention Logits（2 行）

```python
q = einops.rearrange(q, “B T (K G) H -> B T K G H”, K=self.configs[0].num_kv_heads)
```

```
目的：把 N=8 个 Q head 按 KV 分组，每组 G 个 Q head 共享 1 个 KV head

  K = num_kv_heads = 1
  G = num_heads / num_kv_heads = 8 / 1 = 8

  q(B,867,8,256) → q_gqa(B,867,1,8,256)
                        │ │ │ │  │
                        B T K G  H
```

```python
logits = jnp.einsum(“BTKGH,BSKH->BKGTS”, q, k, preferred_element_type=jnp.float32)
```

```
  q: (B, T=867, K=1, G=8, H=256)
  k: (B, S=867, K=1, H=256)       ← 训练时 S=T=867
                                     推理时 S=S_prefix+51

  einsum “BTKGH,BSKH->BKGTS”:
    ∑_H q[b,t,k,g,h] * k[b,s,k,h]
    → logits[b,k,g,t,s]

  logits: (B, K=1, G=8, T=867, S=867)  float32
           │   │   │    │       │
           B   KV组 Q组  query  key/value 序列长
```

---

#### ⑧ Attention Mask 校验 + 应用（3 行）

```python
if attn_mask.shape != (q.shape[0], 1, q.shape[1], k.shape[1]):
    raise ValueError(...)
# 期望 attn_mask: (B, 1, T=867, S=867)
# Module.__call__ 里做了 mask = mask[:, None, :, :]  [B,1,T,S]
```

```
attn_mask(B,1,867,867):           logits(B,1,8,867,867):
  True  = 可见（正常注意力）          ┌─────────────────────────────┐
  False = 不可见（填充极小值）         │  attn_mask[:, :, None, :, :]│
                                    │  broadcast (B,1,1,867,867)  │
  结构（基于 make_attn_mask）：                │                           │
  ┌─────────────────────────────┐   │ → 对 G 维广播                │
  │        │ prefix │ suffix    │   └─────────────────────────────┘
  │ prefix │  True  │  False    │
  │ suffix │  True  │  True     │   masked_logits = where(mask, logits, -2.38e38)
  └─────────────────────────────┘   # -2.3819763e38 ≈ float32 极小值
                                    # softmax 后这些位置概率 ≈ 0
```

```python
big_neg = -2.3819763e38
masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)
# attn_mask[:, :, None, :, :] shape: (B, 1, 1, 867, 867)
# logits shape:                      (B, 1, 8, 867, 867)
# broadcast: mask 在 G 维广播到 8 组
# masked_logits: (B, 1, 8, 867, 867)  float32
```

---

#### ⑨ Softmax → Probs（1 行）

```python
probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)
```

```
  masked_logits: (B, 1, 8, T=867, S=867)  float32
                                  axis=-1 ↑   (沿 key 维做 softmax)

  softmax(axis=-1): 每个 query token 对 key 维归一化，∑_s probs[...,t,s]=1

  .astype(bfloat16): cast 回原始精度
  probs: (B, 1, 8, 867, 867)  bfloat16
```

---

#### ⑩ 加权聚合 + Rearrange（2 行）

```python
encoded = jnp.einsum(“BKGTS,BSKH->BTKGH”, probs, v)
```

```
  probs: (B, K=1, G=8, T=867, S=867)
  v:     (B, S=867, K=1, H=256)

  einsum “BKGTS,BSKH->BTKGH”:
    ∑_s probs[b,k,g,t,s] * v[b,s,k,h]
    → encoded[b,t,k,g,h]

  encoded: (B, T=867, K=1, G=8, H=256)
```

```python
encoded = einops.rearrange(encoded, “B T K G H -> B T (K G) H”)
```

```
  (B, 867, 1, 8, 256) → (B, 867, 8, 256)
  K*G = 1*8 = 8 = num_heads ← 还原成完整头维
```

---

#### ⑪ 切片 + 输出投影（逐 expert）

```python
out = []
start = 0
for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
    if x is not None:
        end = start + x.shape[1]
        out_einsum = lora.Einsum(
            shape=(config.num_heads, config.head_dim, config.width),
            name=_name(“attn_vec_einsum”, i),
            ...
        )
        out.append(out_einsum(“BTNH,NHD->BTD”, encoded[:, start:end]))
        start = end
    else:
        out.append(None)
```

```
encoded(B, 867, 8, 256):
  ├── [0:816]  → encoded_prefix(B, 816, 8, 256)
  └── [816:867]→ encoded_suffix(B, 51,  8, 256)

── Expert 0（i=0, x=prefix, start=0, end=816） ───────────────────────
  out_einsum weight: (num_heads=8, head_dim=256, width=2048)
  name: “attn_vec_einsum”  (无后缀，兼容 PaliGemma 预训练)

  einsum “BTNH,NHD->BTD”:
    encoded_prefix(B,816,8,256) × weight(8,256,2048)
    → out[0](B, 816, 2048)
  start = 816

── Expert 1（i=1, x=suffix, start=816, end=867） ─────────────────────
  out_einsum weight: (num_heads=8, head_dim=256, width=1024)
  name: “attn_vec_einsum_1”  (后缀 _1，新增 expert 独立权重)

  einsum “BTNH,NHD->BTD”:
    encoded_suffix(B,51,8,256) × weight(8,256,1024)
    → out[1](B, 51, 1024)
  start = 867

推理时（xs[0]=None）:
  i=0: x is None → out.append(None), start 不变
  i=1: end = 0 + 51 = 51
       encoded[:, 0:51] ← 此时 encoded 只有 suffix 部分 (B,51,8,256)
       → out[1](B, 51, 1024)
```

---

#### ⑫ 返回值（1 行）

```python
return out, (k, v)
```

```
out = [
    out[0]: (B, 816, 2048)  或  None（推理时）,   ← prefix attention 输出
    out[1]: (B, 51, 1024),                        ← suffix attention 输出
]

(k, v) = KV cache for next decoding step:
    k: (B, S_total, 1, 256)
    v: (B, S_total, 1, 256)
    训练时 S_total=867，推理时 S_total=S_prefix+51
```

---

### 12.3 推理 vs 训练的完整形状对比

```
                        训练（full forward）           推理（kv_cache 模式）
────────────────────────────────────────────────────────────────────────────
xs                      [prefix(B,816,2048),          [None,
                         suffix(B,51,1024)]             suffix(B,51,1024)]

qkvs 中参与的 expert    expert 0 + expert 1            仅 expert 1

Q（拼接后）             (B, 867, 8, 256)               (B, 51, 8, 256)
K（拼接后）             (B, 867, 1, 256)               (B, 51, 1, 256)
V（拼接后）             (B, 867, 1, 256)               (B, 51, 1, 256)

kv_cache 处理           跳过（cache=None）             K = cat([cache_k, K])
                                                       V = cat([cache_v, V])
K_full（用于 logits）   (B, 867, 1, 256)               (B, S_prefix+51, 1, 256)
V_full                  (B, 867, 1, 256)               (B, S_prefix+51, 1, 256)

logits                  (B,1,8,867,867)                (B,1,8,51,S_prefix+51)
                        Q_len × K_len                  Q_len(51) × K_len(S_prefix+51)

encoded（rearrange后）  (B, 867, 8, 256)               (B, 51, 8, 256)

out[0]                  (B, 816, 2048)                 None
out[1]                  (B, 51, 1024)                  (B, 51, 1024)

返回 kv_cache           (K(B,867,1,256),               (K(B,S_prefix+51,1,256),
                         V(B,867,1,256))                V(B,S_prefix+51,1,256))
────────────────────────────────────────────────────────────────────────────
```

---

### 12.4 命名规则与权重共享机制

```
_name(base, i) 的行为:
  i=0 → 直接返回 base          （与 PaliGemma 预训练权重键名完全一致）
  i=1 → 返回 base + “_1”       （action expert 的专属权重）
  i=2 → 返回 base + “_2”       （若有第三个 expert）

本模块涉及的权重名:
  ┌─────────────────────────────────┬───────────────────────────────┐
  │ Expert 0 (PaliGemma)            │ Expert 1 (Action Expert)      │
  ├─────────────────────────────────┼───────────────────────────────┤
  │ q_einsum         (8,2048,256)   │ q_einsum_1     (8,1024,256)   │
  │ kv_einsum        (2,1,2048,256) │ kv_einsum_1    (2,1,1024,256) │
  │ attn_vec_einsum  (8,256,2048)   │ attn_vec_einsum_1(8,256,1024) │
  └─────────────────────────────────┴───────────────────────────────┘

  注意：Attention 模块本身没有”共享权重”——
  每个 expert 的 Q/K/V/O 投影矩阵都是独立的。
  “共享”体现在：两个 expert 的 token 被拼接后在同一个 softmax 里做注意力，
  使得 suffix token 可以 attend 到 prefix token（跨 expert 信息交互）。
```

