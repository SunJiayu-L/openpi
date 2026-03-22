# π0 与 π0.5 模型架构深度解析

## 一、总体架构概览

π0 和 π0.5 是基于**流匹配（Flow Matching）**的 **Vision-Language-Action (VLA)** 模型，用于机器人控制任务。两者共享相同的三组件骨架，但在时步注入和状态输入方式上有关键差异。

### 1.1 三组件范式

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                         Pi0 / Pi0.5 Model                                     │
│                                                                               │
│  ┌───────────────┐    ┌───────────────────────────────────────────────┐       │
│  │ SigLIP ViT    │    │   PaliGemmaWithExpert (双专家 Transformer)     │       │
│  │ (Vision Enc.) │──> │                                               │ ──>   │
│  └───────────────┘    │  [PaliGemma 2B] + [Action Expert 300M]        │  动作  │
│                       │   共享 Attention 层，分离 FFN/Norm 权重         │       │
│  ┌───────────────┐    └───────────────────────────────────────────────┘       │
│  │ 语言 Tokenizer│──>  Prefix 序列（图像 + 文本）                              │
│  └───────────────┘                                                            │
│                                                                               │
│  ┌───────────────┐                                                            │
│  │ 噪声动作 x_t  │──>  Suffix 序列（状态 + 动作 + 时步）                       │
│  └───────────────┘                                                            │
└───────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 π0 vs π0.5 核心差异

| 特性 | π0 | π0.5 |
|------|-----|------|
| **状态输入方式** | 连续向量，经 `state_proj` 投影为 suffix token | 离散化为语言 token，编码进 prefix |
| **时步注入方式** | 与动作 token 拼接后经 MLP 融合 | 经 MLP 处理后输入 **adaRMSNorm** |
| **Action Expert Norm** | 普通 RMSNorm | **自适应 adaRMSNorm**（接收时步条件） |
| **max_token_len** | 48 | 200 |
| **Action Horizon** | 50 | 50（可按机器人配置） |

### 1.3 源码文件对照表

| 组件 | 源码文件 | 关键行号 |
|------|---------|---------|
| **π0/π0.5 主模型** | `src/openpi/models/pi0.py` | L66-279 |
| **模型配置** | `src/openpi/models/pi0_config.py` | L18-108 |
| **Gemma（双专家 Transformer）** | `src/openpi/models/gemma.py` | 全文 |
| **SigLIP 视觉编码器** | `src/openpi/models/siglip.py` | 全文 |
| **基础数据结构** | `src/openpi/models/model.py` | 全文 |
| **PyTorch 实现** | `src/openpi/models_pytorch/pi0_pytorch.py` | 全文 |

---

## 二、Vision Encoder：SigLIP ViT（So400m/14）

**这是一个纯 Transformer 的 ViT，与 InternVL 的 ViT 结构相似，但用 LayerNorm（不是 RMSNorm），且激活函数为 GELU。**

源码位于 `src/openpi/models/siglip.py`

### 2.1 SigLIP 整体结构

```
输入图像 (B, 224, 224, 3)
        |
        v
  ┌─────────────────────────────────────────────────────┐
  │  Patch Embedding (Stem)                             │
  │  Conv2d(3→1152, kernel=14, stride=14, no bias)      │
  │  → (B, 16, 16, 1152) → reshape → (B, 256, 1152)    │
  └─────────────────────────────────────────────────────┘
        |
        v
  ┌─────────────────────────────────────────────────────┐
  │  + 2D Sincos 位置编码                                │
  │  (B, 256, 1152)                                     │
  │  sin/cos of x-position and y-position               │
  │  不可学习，固定公式生成                               │
  └─────────────────────────────────────────────────────┘
        |
        v
  ┌─────────────────────────────────────────────────────┐
  │  Transformer Encoder × 27 层                        │
  │  (详见 2.2)                                          │
  └─────────────────────────────────────────────────────┘
        |
        v
  ┌─────────────────────────────────────────────────────┐
  │  Final LayerNorm(1152)                              │
  └─────────────────────────────────────────────────────┘
        |
        v
  输出 (B, 256, 1152)    pool_type="none"，保留所有 patch
        |
        v
  ┌─────────────────────────────────────────────────────┐
  │  Head 线性投影                                       │
  │  nn.Dense(num_classes=2048)                          │
  │  num_classes = paligemma_config.width = 2048         │
  │  (B, 256, 1152) → (B, 256, 2048)                    │
  └─────────────────────────────────────────────────────┘
        |
        v
  最终输出 (B, 256, 2048)    ← 匹配 PaliGemma 的隐藏维度
```

> **注意**：SigLIP 内部隐藏维度是 1152，但 `num_classes=2048` 的 Head 投影将输出对齐到 PaliGemma 的 2048 维空间。后续文档中 SigLIP 的"输出"均指投影后的 (B, 256, 2048)。

**与 InternVL ViT 的关键对比**：
- InternVL 用 **RMSNorm + LayerScale + DropPath**；SigLIP 用 **LayerNorm**，无 LayerScale，无 DropPath
- InternVL 用可学习绝对位置编码；SigLIP 用**固定 2D Sincos 位置编码**
- InternVL 有 class token；SigLIP **无 class token**，输出全部 256 个 patch token
- InternVL 48 层；SigLIP **27 层**

### 2.2 单个 Encoder Block（27层）

```
输入 hidden_states (B, 256, 1152)
        |
        |--------------------------------------------┐
        |                                            | (残差)
        v                                            |
  ┌───────────────────┐                              |
  │  LayerNorm(1152)   │  标准 LayerNorm（有 weight   |
  │                    │  和 bias），在 bfloat16 下    |
  └────────┬──────────┘                              |
           v                                         |
  ┌───────────────────┐                              |
  │  MultiHeadDotProduct                             |
  │  Attention         │  16 heads, head_dim=72      |
  │  (详见 2.3)        │                              |
  └────────┬──────────┘                              |
           v                                         |
        + <------------------------------------------┘
        |
        |--------------------------------------------┐
        |                                            | (残差)
        v                                            |
  ┌───────────────────┐                              |
  │  LayerNorm(1152)   │                             |
  └────────┬──────────┘                              |
           v                                         |
  ┌───────────────────┐                              |
  │  MLP               │  1152 → 4304 → 1152         |
  │  (详见 2.4)        │  GELU 激活                   |
  └────────┬──────────┘                              |
           v                                         |
        + <------------------------------------------┘
        |
        v
   输出 (B, 256, 1152)
```

核心公式（Post-Norm → 但实际是 Pre-Norm，在 Attention/MLP 之前做 Norm）：
```
x = x + Attention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

### 2.3 Multi-Head Attention（SigLIP 版）

```
输入 x (B, 256, 1152)
        |
        v
  ┌──────────────────────────┐
  │  Q = Linear(1152, 1152)  │  → reshape → (B, 256, 16, 72)
  │  K = Linear(1152, 1152)  │  → reshape → (B, 256, 16, 72)
  │  V = Linear(1152, 1152)  │  → reshape → (B, 256, 16, 72)
  └──────────────────────────┘
        |
        v
  ┌──────────────────────────┐
  │  attn = (Q @ K^T) / √72  │  scale = 1/√72 ≈ 0.118
  │  attn = softmax(attn)    │  全 patch 间双向注意力（无因果掩码）
  │  out = attn @ V          │
  └──────────────────────────┘
        |  reshape → (B, 256, 1152)
        v
  ┌──────────────────────────┐
  │  out_proj = Linear(1152, 1152) │
  └──────────────────────────┘
        v
   输出 (B, 256, 1152)
```

**关键特点**：标准 MHA（不是 GQA），16个头均有独立的 K/V，无 RoPE，无 QK Norm。

### 2.4 MLP（SigLIP 版，标准 2 层 GELU）

```
输入 (B, 256, 1152)
        |
        v
  Linear(1152 → 4304)    扩展 ~3.75 倍
        |
        v
  GELU()
        |
        v
  Linear(4304 → 1152)    压缩回来
        |
        v
  输出 (B, 256, 1152)
```

### 2.5 SigLIP 默认配置（So400m/14）

| 配置项 | 值 | 说明 |
|--------|-----|------|
| `variant` | `So400m/14` | 模型规格 |
| `patch_size` | 14 | patch 大小 |
| `image_size` | 224 | 输入分辨率 |
| `num_patches` | 256 | (224/14)² |
| `width` | 1152 | 隐藏维度 |
| `depth` | 27 | Transformer 层数 |
| `mlp_dim` | 4304 | MLP 中间维度 |
| `num_heads` | 16 | 注意力头数 |
| `head_dim` | 72 | 1152/16 |
| `pool_type` | `none` | 输出所有 patch token |
| `pos_embedding` | 2D sincos | 固定公式，不可训练 |
| `norm_type` | LayerNorm | 非 RMSNorm |
| `activation` | GELU | MLP 激活函数 |
| `head_proj` | Linear(1152→2048) | 输出投影到 PaliGemma 维度 |

---

## 三、输入处理管线：从原始输入到 Prefix + Suffix

**这是连接 Vision Encoder 和双专家 Transformer 的关键桥梁。** 所有模态（图像、语言文本、机器人状态、噪声动作、时步）都需要经过各自的处理流水线，最终组装成两个序列：

- **Prefix 序列**（图像 + 语言 token）→ 由 PaliGemma 专家处理
- **Suffix 序列**（状态 + 动作 + 时步 token）→ 由 Action Expert 处理

源码位于 `src/openpi/models/pi0.py:L106-186`

### 3.1 语言文本处理

源码位于 `pi0.py:L128-133`

语言指令（如 "pick up the red cube"）在进入模型前已经被 tokenizer 转化为 token ID 序列：

```
原始文本: "pick up the red cube"
        |
        v
  Tokenizer (外部预处理，词表大小 257152)
        |
        v
  tokenized_prompt (B, 48)        ← π0: 最多 48 个 token ID
                                     π0.5: 最多 200 个 token ID
        |                            (π0.5 更长因为包含了离散化的机器人状态)
        v
  ┌──────────────────────────────────────────────────────┐
  │  PaliGemma.llm.embed(token_ids)                      │
  │                                                      │
  │  embedding_table: (257152, 2048)                     │
  │  查表: 每个 token ID → 一个 2048 维向量               │
  │                                                      │
  │  注：Gemma 的 embed 方法会将结果乘以 √2048 ≈ 45.25   │
  │  即 output = embedding_table[token_ids] * √width     │
  └──────────────────────────────────────────────────────┘
        |
        v
  tokenized_inputs (B, 48, 2048)   ← 每个 token 对应一个 2048 维嵌入向量
```

**词表大小 257152** 来自 PaliGemma 的词表（Gemma 基础词表 256000 + 1152 个图像特殊 token）。

**关键点**：语言嵌入和图像投影后的输出都在同一个 **2048 维空间**，这使得它们可以直接拼接。

### 3.2 Prefix 序列组装（图像 + 语言融合）

源码位于 `pi0.py:L109-137`

Prefix 是模型的"理解输入"，包含所有视觉和语言信息：

```
════════ 图像处理（3路摄像头）════════

  base_0_rgb (B,224,224,3)
        |
        v
  SigLIP ViT → Head(1152→2048) → image_tokens_0 (B, 256, 2048)
  image_mask_0: (B,) 该摄像头是否有效（不同机器人摄像头数量不同）
  ar_mask: [False] × 256

  left_wrist_0_rgb (B,224,224,3)
        |
        v
  SigLIP ViT → Head(1152→2048) → image_tokens_1 (B, 256, 2048)
  ar_mask: [False] × 256

  right_wrist_0_rgb (B,224,224,3)
        |
        v
  SigLIP ViT → Head(1152→2048) → image_tokens_2 (B, 256, 2048)
  ar_mask: [False] × 256

════════ 语言处理 ════════

  tokenized_prompt (B, 48)
        |
        v
  PaliGemma.llm.embed → tokenized_inputs (B, 48, 2048)
  ar_mask: [False] × 48

════════ 拼接 ════════

  prefix_tokens = concat([img_0, img_1, img_2, lang], axis=1)
               → (B, 256+256+256+48, 2048)
               = (B, 816, 2048)                ← 所有 token 都是 2048 维！

  prefix_mask = concat([img_mask_0, img_mask_1, img_mask_2, lang_mask])
              → (B, 816)
              标记哪些 token 有效（缺少的摄像头对应 256 个 False）

  prefix_ar_mask = [False] × 816               ← 全部双向注意力
```

**为什么 ar_mask 全是 False？**
- `ar_mask=False` 表示"与前一个 token 在同一个 attention block 中"
- 全 False 意味着 prefix 内所有 token（图像 + 语言）互相可见，没有因果约束
- 图像 token 可以 attend 语言 token，反之亦然 → **隐式的视觉-语言融合发生在 Transformer 的注意力层中**

### 3.3 机器人状态处理

π0 和 π0.5 处理机器人状态（关节角度、末端位置等）的方式完全不同：

**π0**：状态作为连续向量，投影为 suffix 的第一个 token（源码 `pi0.py:L151-157`）

```
obs.state (B, 32)              ← 32维：关节角度/末端执行器位置/夹爪状态等
        |
        v
  state_proj                   Linear(32 → 1024)   ← 投影到 Action Expert 维度
        |
        v
  state_token (B, 1024)
        |
        v
  unsqueeze → (B, 1, 1024)    ← 作为 suffix 序列的第一个 token
  ar_mask: [True]              ← True 标记新 attention block 的开始
                                  → prefix token 无法 attend 到这个 token
```

**π0.5**：状态在数据预处理阶段已被序列化为文本字符串

```
原始状态: [0.1, -0.5, 0.3, ...]
        |
        v  (外部预处理，非模型内部)
文本化: "state: joint1=0.1 joint2=-0.5 joint3=0.3 ..."
        |
        v
与语言指令拼接: "pick up red cube. state: joint1=0.1 ..."
        |
        v
Tokenizer → tokenized_prompt (B, 200)   ← 这就是 max_token_len=200 的原因
        |
        v
已包含在 prefix 中，无需 state_proj
```

### 3.4 噪声动作处理

源码位于 `pi0.py:L159`

训练时输入的是流匹配加噪后的动作，推理时输入纯高斯噪声：

```
训练: x_t = t·ε + (1-t)·actions   ← 噪声与真实动作的插值
推理: x_t = ε ~ N(0, I)            ← 从纯噪声开始

x_t (B, 50, 32)               ← 50个时间步 × 32维动作空间
        |
        v
  action_in_proj               Linear(32 → 1024)   ← 投影到 Action Expert 维度
        |
        v
  action_tokens (B, 50, 1024)  ← 50 个 token，每个 1024 维
```

### 3.5 时步编码与融合

时步 t ∈ (0, 1) 表示去噪进度。**π0 和 π0.5 对时步的融合方式是两者最核心的差异之一**。

#### 3.5.1 时步位置编码（共用）

源码位于 `pi0.py:L48-63, L161`

```
t (B,)                         ← 标量时步，如 t=0.7 表示 70% 噪声
        |
        v
  posemb_sincos(t, dim=1024, min_period=4e-3, max_period=4.0)
        |
        v
  ┌───────────────────────────────────────────────────────────────┐
  │  fraction = linspace(0, 1, 512)     ← 512 个频率采样点         │
  │  period = 0.004 × (1000)^fraction   ← 对数均匀：[0.004, 4.0]  │
  │  sinusoid = t / period × 2π         ← (B, 512)               │
  │  output = [sin(sinusoid), cos(sinusoid)]  ← (B, 1024)        │
  └───────────────────────────────────────────────────────────────┘
        |
        v
  time_emb (B, 1024)           ← 时步嵌入向量
```

#### 3.5.2 π0：拼接 + MLP 前融合

源码位于 `pi0.py:L170-177`

π0 在进入 Transformer **之前**就将时步信息融入动作 token：

```
action_tokens (B, 50, 1024)       time_emb (B, 1024)
        |                                |
        |                                v
        |                         repeat → (B, 50, 1024)  ← 每步复制相同时步
        |                                |
        └──── concat(axis=-1) ───────────┘
                    |
                    v
              (B, 50, 2048)            ← 动作 + 时步 拼接
                    |
                    v
              action_time_mlp_in       Linear(2048 → 1024)
                    |
                    v
              swish(x)                 ← SiLU 激活
                    |
                    v
              action_time_mlp_out      Linear(1024 → 1024)
                    |
                    v
              action_expert_tokens (B, 50, 1024)  ← 时步已融入
              adarms_cond = None                  ← π0 不使用 adaRMS
```

#### 3.5.3 π0.5：MLP → adaRMSNorm 逐层注入

源码位于 `pi0.py:L162-169`

π0.5 不在输入阶段融合时步，而是将时步嵌入作为**条件信号**传给 Transformer 每一层的 adaRMSNorm：

```
time_emb (B, 1024)             action_tokens (B, 50, 1024)
        |                                |
        v                                v
  time_mlp_in  Linear(1024→1024)   action_expert_tokens = action_tokens
        |                          (不做时步融合，直接传入)
        v
  swish(x)
        |
        v
  time_mlp_out Linear(1024→1024)
        |
        v
  swish(x)
        |
        v
  adarms_cond (B, 1024)     ← 传给 Transformer 的每一层
                               在 adaRMSNorm 中生成 scale/shift/gate
                               → 时步信息在 Transformer 内部逐层注入
```

**π0 vs π0.5 时步融合的本质区别**：
| | π0 | π0.5 |
|--|-----|------|
| **融合位置** | Transformer 输入之前 | Transformer 每一层 Norm |
| **融合方式** | 拼接 → MLP 压缩 | adaRMSNorm(scale, shift, gate) |
| **影响深度** | 一次性融入，后续不再显式注入 | 每层都重新注入，调制更精细 |
| **类比** | FiLM 的前置版 | DiT 的 adaLN-Zero |

### 3.6 Suffix 序列组装

源码位于 `pi0.py:L179-186`

```
π0 的 Suffix:
══════════════════════════════════════════════════════════
  [state_token]  +  [action_expert_tokens]
  (B, 1, 1024)     (B, 50, 1024)
       ↓                  ↓
       └── concat ────────┘
              ↓
  suffix_tokens (B, 51, 1024)

  ar_mask: [True] + [True] + [False] × 49
            ↑        ↑          ↑
         state     action_0   action_1~49
      (新block开始)(新block开始)(block内双向)

  注意：action 的第一个 token 也有 ar_mask=True（代码 L182）
  这意味着 suffix 内部有两个 block:
    block 1: [state_token]          ← 单独一个 token
    block 2: [action_tokens × 50]   ← 50 个 token 互相双向 attend


π0.5 的 Suffix:
══════════════════════════════════════════════════════════
  [action_tokens]              ← 没有 state_token
  (B, 50, 1024)

  ar_mask: [True] + [False] × 49
            ↑          ↑
         action_0   action_1~49
      (新block开始)(block内双向)

  adarms_cond = time_emb (B, 1024)  ← 传给 Transformer
```

### 3.7 完整输入处理流水线图

```
┌─────────────────────────── 原始输入 ───────────────────────────────┐
│                                                                    │
│  图像×3 (224,224,3)   语言 token IDs (48/200)   状态 (32)  t∈(0,1) │
│  噪声动作 x_t (50,32)                                              │
└────────────────────────────────────────────────────────────────────┘
         │                    │                │         │        │
         ▼                    ▼                ▼         ▼        ▼
  ┌──────────────┐  ┌─────────────────┐  ┌─────────┐  ┌────┐  ┌────────┐
  │ SigLIP ViT   │  │ PaliGemma.embed │  │state_proj│  │sincos│ │act_proj│
  │ +Head(→2048) │  │ table(257152,   │  │Lin(32→  │  │pos  │ │Lin(32→ │
  │              │  │       2048)     │  │   1024) │  │emb  │ │  1024) │
  │ (B,256,2048) │  │ ×√2048         │  │(B,1,1024│  │(B,  │ │(B,50,  │
  │   ×3 路      │  │ (B,48,2048)    │  │) π0 only│  │1024)│ │ 1024)  │
  └──────┬───────┘  └───────┬────────┘  └────┬────┘  └──┬─┘  └───┬────┘
         │                  │                │         │        │
         │    ┌─────────────┘                │         │        │
         │    │                              │         │        │
         ▼    ▼                              ▼         ▼        ▼
  ┌──────────────────────┐          ┌──────────────────────────────┐
  │  Prefix 序列          │          │  Suffix 序列                  │
  │  concat(img×3, lang)  │          │  π0:  concat(state, MLP(     │
  │  (B, 816, 2048)       │          │       concat(act,time)))     │
  │  ar_mask: [F]×816     │          │       (B, 51, 1024)          │
  │                       │          │  π0.5: act_tokens             │
  │  全部双向注意力         │          │       (B, 50, 1024)          │
  │  图像↔语言 互相可见     │          │       + adarms_cond           │
  └──────────┬────────────┘          └──────────────┬───────────────┘
             │                                      │
             └────────── 送入双专家 Transformer ──────┘
                    (详见第四章)
```

---

## 四、双专家 Transformer：PaliGemma + Action Expert

这是整个系统最核心的设计。**两个 Gemma 模型共享同一套 Attention 权重，但有各自独立的 FFN、Norm 和 Projection 权重**。

源码位于 `src/openpi/models/gemma.py`

### 4.1 设计思想：混合专家注意力

```
Prefix 序列 (图像 + 文本 tokens)        Suffix 序列 (动作 tokens)
  │                                         │
  │  使用 PaliGemma 的 FFN/Norm 权重          │  使用 Action Expert 的 FFN/Norm 权重
  │                                         │
  └──────────────┬──────────────────────────┘
                 │  两个序列拼接，
                 │  共享同一套 Attention Q/K/V 权重
                 │  → 统一做注意力计算
                 │
                 ▼
              输出分开
       Prefix 输出 (忽略)      Suffix 输出 → 动作预测
```

**关键点**：共享 Attention 意味着动作 token 可以直接 attend 到图像和语言 token，实现视觉-语言-动作的统一建模。

### 4.2 双专家 Transformer 整体结构

```
PaliGemmaWithExpert (Module in gemma.py)
├── embedder: Embedder(vocab_size=257152, embed_dim=2048)   ← 只有 PaliGemma 用
├── layers: [Block × 18]                                    ← 18 层，每层双专家
│   └── 每层 Block:
│       ├── pre_attention_norm (RMSNorm)      ← 两个专家各一个
│       ├── attn (Attention)                  ← 两个专家共享
│       ├── pre_ffw_norm (RMSNorm)            ← 两个专家各一个
│       └── mlp (FeedForward)                 ← 两个专家各一个
└── final_norms: [RMSNorm × 2]                ← 两个专家各一个最终 Norm
```

### 4.3 单个 Block 前向传播

源码位于 `gemma.py:283-333`

```
输入 xs = [prefix_tokens (B, S_pre, 2048), suffix_tokens (B, S_suf, 1024)]
         adarms_cond = [None, time_emb (B, 1024)]  ← π0.5 时 suffix 有条件
        |
        |  === Attention 子层 ===
        |
        v
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Pre-Attention Norm（各专家独立）                                     │
  │                                                                     │
  │  prefix: pre_attention_norm_0(prefix_tokens, cond=None)             │
  │          → RMSNorm → (B, S_pre, 2048)                               │
  │                                                                     │
  │  suffix: pre_attention_norm_1(suffix_tokens, cond=time_emb)         │
  │          → adaRMSNorm (π0.5) 或 RMSNorm (π0)                        │
  │          → (B, S_suf, 1024)，同时输出 gate                           │
  └─────────────────────────────────────────────────────────────────────┘
        |
        v
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Shared Attention（两专家共享权重，拼接后统一计算）                     │
  │                                                                     │
  │  Q: q_einsum_0(prefix) ── 拼接 ──> (B, S_total, 8, 256)            │
  │     q_einsum_1(suffix) ──┘                                          │
  │                                                                     │
  │  K: kv_einsum_0(prefix) ── 拼接 ──> (B, S_total, 1, 256)           │
  │     kv_einsum_1(suffix) ──┘                                         │
  │                                                                     │
  │  V: 同上                                                             │
  │                                                                     │
  │  RoPE 位置编码 → Q, K 旋转                                           │
  │  Scale: head_dim^(-0.5) = 256^(-0.5) = 0.0625                      │
  │  掩码（make_attn_mask）控制注意力可见性                               │
  │  softmax → (B, 1, 8, S_total, S_total) [GQA: 8Q 共享 1 KV]         │
  │  → 输出分割回各专家                                                   │
  │                                                                     │
  │  output_proj_0(prefix_out) → (B, S_pre, 2048)                      │
  │  output_proj_1(suffix_out) → (B, S_suf, 1024)                      │
  └─────────────────────────────────────────────────────────────────────┘
        |
        |  gated_residual: x + y * gate (π0.5 adaRMS) 或 x + y (π0)
        v
        x (B, S_pre, 2048) + x (B, S_suf, 1024)
        |
        |  === FFN 子层 ===
        v
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Pre-FFW Norm（各专家独立，与 Attention 同样的 RMS/adaRMS 机制）       │
  └─────────────────────────────────────────────────────────────────────┘
        |
        v
  ┌─────────────────────────────────────────────────────────────────────┐
  │  FeedForward（各专家独立权重）                                        │
  │                                                                     │
  │  PaliGemma FFN: (B, S_pre, 2048) → 16384 → 2048  (GeGLU)          │
  │  Action Expert FFN: (B, S_suf, 1024) → 4096 → 1024  (GeGLU)       │
  │  (详见 4.5)                                                          │
  └─────────────────────────────────────────────────────────────────────┘
        |
        |  gated_residual
        v
   输出 xs = [prefix_out (B, S_pre, 2048), suffix_out (B, S_suf, 1024)]
```

### 4.4 Attention（GQA + RoPE）

源码位于 `gemma.py:157-249`

```
Q: q_einsum "BTD,NDH->BTNH"
   形状: (B, T, num_heads=8, head_dim=256)

K: kv_einsum "BSD,2KDH->2BSKH" 取 k
   形状: (B, S, num_kv_heads=1, head_dim=256)

V: kv_einsum 取 v
   形状: (B, S, num_kv_heads=1, head_dim=256)
```

**GQA 配置**（两个专家相同）：
```
num_heads = 8        ← Q 头数
num_kv_heads = 1     ← K/V 只有 1 个头
                       8 个 Q head 全部 attend 同一个 KV head
head_dim = 256       ← 每个头的维度
```

**RoPE 旋转位置编码**（源码 `gemma.py:424-440`）：
```python
freq_exponents = (2.0 / head_dim) * arange(head_dim/2)  # [0, 2/256, 4/256, ...]
timescale = 10000 ** freq_exponents
radians = positions / timescale           # (B, L, head_dim/2)
x1, x2 = split(x, 2, axis=-1)
result = concat([x1*cos - x2*sin, x2*cos + x1*sin])
```

**注意力掩码 `make_attn_mask`** 控制谁可以 attend 谁（源码 `pi0.py:19-44`）：
```
Prefix tokens (图像+文本): ar_mask=False → 双向互相 attend
Suffix 第一个 token:        ar_mask=True  → 标志新 block 开始
Suffix 后续 tokens:         ar_mask=False → 在 suffix block 内双向 attend

最终效果:
- 图像/文本 token: 互相 attend（双向）
- 动作 token: attend 所有图像/文本 + 同 block 内的其他动作 token
- 图像/文本 token: 不 attend 动作 token（因果隔离）
```

### 4.5 FeedForward（GeGLU）

源码位于 `gemma.py:252-280`

```python
# GeGLU: 与 SwiGLU 相似，但用 GELU 代替 SiLU
w_gating: (2, features, hidden_dim)  # 一个矩阵存 gate 和 up

ff_gate = x @ w_gating[0]           # (B, T, hidden_dim)
gate_value = GELU(ff_gate)           # 门控

ff1 = x @ w_gating[1]               # (B, T, hidden_dim)
activations = gate_value * ff1       # 逐元素相乘

w_linear: (hidden_dim, features)
output = activations @ w_linear      # (B, T, features)
```

**PaliGemma FFN 参数**：

| 参数 | 形状 | 说明 |
|------|------|------|
| `gating_einsum` | (2, 2048, 16384) | gate + up 投影 |
| `linear` | (16384, 2048) | down 投影 |

**Action Expert FFN 参数**：

| 参数 | 形状 | 说明 |
|------|------|------|
| `gating_einsum_1` | (2, 1024, 4096) | gate + up 投影 |
| `linear_1` | (4096, 1024) | down 投影 |

### 4.6 Normalization：RMSNorm 与 adaRMSNorm

源码位于 `gemma.py:112-131`

**普通 RMSNorm**（PaliGemma 专家使用，π0 Action Expert 也使用）：
```python
var = mean(x²)                         # float32
normed = x / sqrt(var + 1e-6)
scale = param("scale", zeros, (dim,))  # 初始化为 0
return normed * (1 + scale)            # 注意: 初始等效于 normed * 1
```

**adaRMSNorm**（π0.5 的 Action Expert 使用）：
```python
# cond 是时步 embedding (B, 1024)
modulation = Dense(dim * 3, kernel_init=zeros)(cond)  # 初始化为 0
scale, shift, gate = split(modulation, 3)              # 各 (B, 1, dim)

normed = RMSNorm(x)
output = normed * (1 + scale) + shift    # 时步条件调制
gate                                     # 用于残差连接: x + y * gate
```

**adaRMSNorm 与普通 RMSNorm 的区别**：

| | 普通 RMSNorm | adaRMSNorm |
|--|-------------|-----------|
| scale | 固定可学习参数 | 由时步 embedding 动态生成 |
| shift | 无 | 由时步 embedding 动态生成 |
| gate | 无 | 由时步 embedding 动态生成，用于门控残差 |
| 残差 | `x + y` | `x + y * gate` |

### 4.7 Gemma 配置汇总

源码位于 `gemma.py:58-108`

| 配置项 | PaliGemma (gemma_2b) | Action Expert (gemma_300m) |
|--------|---------------------|--------------------------|
| `width` | 2048 | 1024 |
| `depth` | 18 | 18（共享 scan 实现） |
| `mlp_dim` | 16384 | 4096 |
| `num_heads` | 8 | 8 |
| `num_kv_heads` | 1 | 1 |
| `head_dim` | 256 | 256 |
| `vocab_size` | 257152 | 不用（无独立 embed） |
| `activation` | GELU (GeGLU) | GELU (GeGLU) |
| `norm` | RMSNorm | RMSNorm (π0) / adaRMSNorm (π0.5) |
| `pos_emb` | RoPE, max_wavelength=10000 | 共享 RoPE |

---

## 五、π0 的完整前向传播

### 5.1 训练时 Forward Pass

源码位于 `pi0.py:189-214`

```
输入:
  obs.images = {"base_0_rgb": (B,224,224,3), "left_wrist_0_rgb": ..., "right_wrist_0_rgb": ...}
  obs.state  = (B, 32)              ← 机器人关节状态
  obs.tokenized_prompt = (B, 48)    ← 语言指令 token
  actions    = (B, 50, 32)          ← 真实动作序列 (训练目标)

════════ STEP 1: 流匹配噪声采样 ════════
  ε ~ N(0, I)   shape=(B, 50, 32)          ← 高斯噪声
  t ~ Beta(1.5, 1.0) * 0.999 + 0.001      ← t ∈ (0.001, 1.0)
  x_t = t * ε + (1-t) * actions            ← 在噪声和真实动作之间插值
  u_t = ε - actions                         ← 目标速度场

════════ STEP 2: embed_prefix() ════════
  source码: pi0.py:106-137

  for each image in {base, left_wrist, right_wrist}:
    image_tokens, _ = SigLIP(image)       → (B, 256, 2048)  ← Head 投影后
    [image tokens with full mutual attention]

  lang_emb = Gemma.embed(tokenized_prompt) → (B, 48, 2048)
    [language tokens with full mutual attention to images]

  prefix_tokens  = concat → (B, 256*3+48, 2048)   ← 816 tokens，全部 2048 维
  prefix_mask    = valid image/language mask
  prefix_ar_mask = [False * 816]                         ← 全部双向

════════ STEP 3: embed_suffix() (π0 版) ════════
  source码: pi0.py:139-186

  state_token = state_proj(obs.state)     → (B, 1024) → (B, 1, 1024)
  action_tokens = action_in_proj(x_t)    → (B, 50, 1024)

  time_emb = posemb_sincos(t, dim=1024, min_period=4e-3, max_period=4.0)
           = [sin(t/period_i), cos(t/period_i)]  → (B, 1024)

  time_tokens = repeat(time_emb, s=50)   → (B, 50, 1024)
  action_time = concat([action_tokens, time_tokens]) → (B, 50, 2048)
  action_time = action_time_mlp_in(action_time)      → (B, 50, 1024)
  action_time = swish(action_time)
  action_time = action_time_mlp_out(action_time)     → (B, 50, 1024)
              [时步信息融入每个动作 token]

  suffix_tokens  = concat([state_token, action_time]) → (B, 51, 1024)
  suffix_ar_mask = [True, False*50]   ← state token 标记 suffix 起点

════════ STEP 4: 统一 Transformer 前向传播 ════════
  full_sequence = [prefix_tokens (B,816,2048), suffix_tokens (B,51,1024)]
  attn_mask = make_attn_mask(...)   控制注意力可见性:
    ┌─────────────────────────────────────────────────────┐
    │         │  prefix(816) │  suffix(51)               │
    ├─────────┼──────────────┼───────────────────────────┤
    │ prefix  │  ✓ (双向)    │  ✗ (看不到 suffix)        │
    │ suffix  │  ✓ (可以看)  │  ✓ (suffix 内部双向)       │
    └─────────┴──────────────┴───────────────────────────┘

  positions = cumsum(input_mask) - 1  ← 位置 ID

  (prefix_out, suffix_out), _ = PaliGemmaWithExpert(
      [prefix_tokens, suffix_tokens],
      positions=positions,
      mask=attn_mask,
      adarms_cond=[None, None]   ← π0 不用 adaRMS
  )

════════ STEP 5: 输出与损失 ════════
  v_t = action_out_proj(suffix_out[:, -50:])  → (B, 50, 32)
        [取 suffix 后 50 个 token，投影到动作空间]

  Loss = mean((v_t - u_t)²)
       = mean((predicted_velocity - target_velocity)²)
       = MSE Loss
```

### 5.2 推理时采样（ODE Euler 积分）

源码位于 `pi0.py:216-279`

```
════════ PREFILL（KV Cache 预填充）════════
  prefix_tokens, prefix_mask, _ = embed_prefix(obs)
  _, kv_cache = Transformer([prefix_tokens, None], ...)
  [将图像和文本的 KV 缓存起来，后续迭代复用]

════════ ODE 迭代去噪（默认 10 步）════════
  x_t = ε ~ N(0, I)   ← 从纯噪声开始
  time = 1.0           ← t=1 为纯噪声，t=0 为干净动作
  dt = -1.0 / 10 = -0.1

  while time >= 0.05:  (robust to floating-point: time >= -dt/2)

    suffix_tokens = embed_suffix(obs, x_t, time)
    [不需要重新计算 prefix，直接用 kv_cache]

    full_attn_mask = [prefix_can_attend_to, suffix_self_attn]
    positions = prefix_len + cumsum(suffix_mask) - 1

    (None, suffix_out), _ = Transformer(
        [None, suffix_tokens],   ← None 表示 prefix 从 kv_cache 读取
        kv_cache=kv_cache,
        ...
    )

    v_t = action_out_proj(suffix_out[:, -50:])   ← 预测速度场

    x_t = x_t + dt * v_t   ← Euler 积分: 从噪声向干净动作移动

    time += dt    (time: 1.0 → 0.9 → 0.8 → ... → 0.1 → 0.0)

  return x_t     ← 最终预测的动作序列 (B, 50, 32)
```

---

## 六、π0 vs π0.5 的差异详解

### 6.1 状态输入方式

**π0**：state 作为连续 token 进入 suffix（源码 `pi0.py:151-157`）
```python
# π0 embed_suffix
state_token = self.state_proj(obs.state)[:, None, :]  # Linear(32→1024) → (B,1,1024)
tokens.append(state_token)
ar_mask += [True]  # state token 标记 suffix 起点
```

**π0.5**：state 已离散化为语言 token，直接在 prefix 的 tokenized_prompt 中（源码 `pi0.py:151`）
```python
# π0.5 embed_suffix — 没有 state_proj，没有 state_token
# state 已经被 tokenizer 转化为 text token 加入 prefix
# suffix 只包含纯动作 token
```

### 6.2 时步注入方式

**π0**：时步与动作 token 拼接，通过 MLP 融合（源码 `pi0.py:161-177`）
```python
# time_emb: (B, 1024)
time_tokens = repeat(time_emb, s=action_horizon)      # (B, 50, 1024)
action_time = concat([action_tokens, time_tokens])    # (B, 50, 2048)
action_time = action_time_mlp_in(action_time)         # Linear(2048→1024)
action_time = swish(action_time)
action_time = action_time_mlp_out(action_time)        # Linear(1024→1024)
# 时步信息被"压缩"进 action token 的表示中
```

**π0.5**：时步通过 MLP 处理后作为 adaRMSNorm 的条件输入（源码 `pi0.py:162-169`）
```python
# time_emb: (B, 1024)
time_emb = self.time_mlp_in(time_emb)    # Linear(1024→1024)
time_emb = swish(time_emb)
time_emb = self.time_mlp_out(time_emb)   # Linear(1024→1024)
time_emb = swish(time_emb)
adarms_cond = time_emb                   # 传给 Transformer 的 adaRMS

# 在每层 Block 中：
# adaRMSNorm(action_expert_tokens, cond=adarms_cond)
# → 时步信息通过 scale/shift/gate 影响每一层的 Norm
# → 比 π0 的 MLP 融合更深层次地调制整个 Transformer
```

### 6.3 π0 vs π0.5 Suffix 结构对比

```
π0 的 Suffix:
  [state_token (1)] + [action_time_tokens (50)]
   ↑ state 作为独立 token    ↑ 动作 + 时步 MLP 融合

π0.5 的 Suffix:
  [action_tokens (50)]
   ↑ 只有动作 token，时步通过 adaRMS 注入，state 在 prefix 里
```

### 6.4 Projection 层差异

| 层 | π0 | π0.5 |
|----|-----|------|
| `action_in_proj` | Linear(32 → 1024) | Linear(32 → 1024)（相同） |
| `state_proj` | Linear(32 → 1024) | **无** |
| `action_time_mlp_in` | Linear(2048 → 1024) | **无** |
| `action_time_mlp_out` | Linear(1024 → 1024) | **无** |
| `time_mlp_in` | **无** | Linear(1024 → 1024) |
| `time_mlp_out` | **无** | Linear(1024 → 1024) |
| `action_out_proj` | Linear(1024 → 32) | Linear(1024 → 32)（相同） |

---

## 七、流匹配（Flow Matching）训练目标

源码位于 `pi0.py:189-214`

### 7.1 数学原理

π0 使用**随机插值流匹配（Stochastic Interpolant）**：

```
流的定义:
  x_t = (1-t) * x_0 + t * ε
  其中 x_0 是真实动作，ε ~ N(0,I) 是噪声，t ∈ [0,1]

目标速度场 (条件流):
  u_t = ε - x_0   (从真实动作到噪声的方向)

训练目标:
  L = E[||v_θ(x_t, t, c) - u_t||²]
  其中 c 是观测条件（图像+文本+状态），v_θ 是神经网络预测的速度

采样（ODE 求解）:
  dx/dt = v_θ(x, t, c)
  从 x_1 = ε 出发，用 Euler 方法积分到 t=0，得到 x_0
```

### 7.2 时步采样分布

```python
# pi0.py:197
t ~ Beta(alpha=1.5, beta=1.0) * 0.999 + 0.001
```

Beta(1.5, 1.0) 分布的均值 = 1.5/(1.5+1.0) = 0.6，偏向较大的 t 值（更噪声端），让模型更多学习去噪的困难部分。

### 7.3 时步位置编码

源码位于 `pi0.py:48-63`

```python
def posemb_sincos(pos, embedding_dim=1024, min_period=4e-3, max_period=4.0):
    fraction = linspace(0, 1, embedding_dim/2)         # [0, 1/512, 2/512, ...]
    period = min_period * (max_period / min_period) ** fraction
    # period 范围: [4e-3, 4.0]，对数均匀分布

    sinusoid_input = pos[:, None] / period[None, :] * 2π
    return concat([sin(sinusoid_input), cos(sinusoid_input)])  # (B, 1024)
```

t ∈ [0,1]，period 范围 [0.004, 4.0] → 不同频率分量覆盖 t 的细节和粗粒度变化。

---

## 八、完整数据流总结

```
输入阶段
════════════════════════════════════════════════════════════════════════════
  图像 (B,224,224,3) x3    文本 (B,48) tokens    状态 (B,32)    t ∈ (0,1)
       │                        │                     │               │
       ▼                        ▼                     ▼               ▼
  SigLIP ViT             PaliGemma.embed          state_proj      posemb_sincos
  (B,256,1152)×3          (B,48,2048)              (B,1,1024)     (B,1024)
       │                        │                     │               │
       └──────── Prefix ─────────┘                    └─── Suffix ────┘
                (B, 816, 2048)                          π0: (B, 51, 1024)
                                                        π0.5: (B, 50, 1024)

双专家 Transformer
════════════════════════════════════════════════════════════════════════════
  [prefix (B,816,2048)]  +  [suffix (B,51,1024)]
          │                          │
          └──────── 18 × Block ───────┘
                 共享 Attention (GQA 8Q/1KV, head_dim=256, RoPE)
                 各自 FFN (GeGLU)
                 各自 Norm (RMSNorm / adaRMSNorm)
          │                          │
          ▼                          ▼
  prefix_out (忽略)         suffix_out (B,51,1024)

输出阶段
════════════════════════════════════════════════════════════════════════════
  suffix_out[:, -50:]    → (B, 50, 1024)
       │
       ▼
  action_out_proj         Linear(1024→32)
       │
       ▼
  v_t = (B, 50, 32)       预测的速度场

训练: Loss = MSE(v_t, u_t)
推理: x_t += dt * v_t  (Euler 积分，10步)，最终 x_0 = 预测动作
```

---

## 九、配置与规格汇总

### 9.1 π0Config 默认值

源码位于 `pi0_config.py:18-47`

| 配置项 | π0 | π0.5 | 说明 |
|--------|-----|------|------|
| `dtype` | `bfloat16` | `bfloat16` | 模型精度 |
| `paligemma_variant` | `gemma_2b` | `gemma_2b` | LLM backbone |
| `action_expert_variant` | `gemma_300m` | `gemma_300m` | Action Expert |
| `action_dim` | 32 | 32 | 动作维度（关节数） |
| `action_horizon` | 50 | 50 | 预测未来步数 |
| `max_token_len` | 48 | 200 | 语言 token 最大长度 |
| `pi05` | False | True | 是否启用 π0.5 特性 |
| `discrete_state_input` | False | True | 状态是否离散化 |

### 9.2 全模型组件规格总览

| 组件 | 参数量 | 维度 | 层数 |
|------|--------|------|------|
| **SigLIP ViT** | ~400M | 1152 | 27 |
| **PaliGemma (Gemma 2B)** | ~2B | 2048 | 18 |
| **Action Expert (Gemma 300M)** | ~311M | 1024 | 18 |
| **action_in_proj** | 32×1024 = 33K | - | - |
| **action_out_proj** | 1024×32 = 33K | - | - |
| **state_proj (π0 only)** | 32×1024 = 33K | - | - |
| **π0 time MLP** | 2048×1024 + 1024×1024 ≈ 3M | - | - |
| **π0.5 time MLP** | 1024×1024 × 2 ≈ 2M | - | - |

### 9.3 Vision Encoder vs Gemma 架构对比

| 特性 | SigLIP ViT | PaliGemma | Action Expert |
|------|-----------|-----------|---------------|
| **架构类型** | 双向 Encoder | 因果 Decoder | 因果 Decoder |
| **层数** | 27 | 18 | 18 |
| **隐藏维度** | 1152 | 2048 | 1024 |
| **注意力头** | 16 (MHA) | 8Q/1KV (GQA) | 8Q/1KV (GQA) |
| **head_dim** | 72 | 256 | 256 |
| **MLP 类型** | 标准 2层 GELU | GeGLU | GeGLU |
| **MLP 中间维度** | 4304 (~3.75x) | 16384 (8x) | 4096 (4x) |
| **Norm 类型** | LayerNorm | RMSNorm | RMSNorm/adaRMSNorm |
| **位置编码** | 2D Sincos (固定) | RoPE | RoPE（共享） |
| **LayerScale** | 无 | 无 | 无 |
| **KV Cache** | 无 | 有（推理时） | 有（推理时） |
| **QK Norm** | 无 | 无 | 无 |

---

## 十、从模型输出到机器人动作：完整后处理流水线

模型输出的 `(B, 50, 32)` 张量是**归一化后的动作序列**，还不能直接发送给机器人。需要经过三个阶段的后处理才能变成真实关节指令。

源码位于 `src/openpi/policies/policy.py`, `src/openpi/policies/*_policy.py`, `src/openpi/transforms.py`

### 10.1 全链路概览

```
模型 sample_actions()
        │
        ▼
  raw_actions (B, 50, 32)     ← 归一化空间，值域约 [-3, 3]（z-score）
        │
  ═══ STEP 1: Unnormalize ════════════════════════════════
        ▼
  Unnormalize(norm_stats)
        │  actions = raw * (std + 1e-6) + mean
        │  还原为机器人物理单位（弧度/米）
        ▼
  actions (B, 50, 32)         ← 物理单位，但仍是 32 维（模型统一维度）
        │
  ═══ STEP 2: OutputTransform（机器人特定）═════════════
        ▼
  各机器人的 XXXOutputs.__call__(data)
        │  裁剪维度：取 [:, :N] 丢弃 padding（32维→实际维度）
        │  坐标变换：夹爪角度转换、关节方向翻转等
        ▼
  actions (B, 50, N)          ← 机器人真实维度（如 ALOHA:14, LIBERO:7）
        │
  ═══ STEP 3: Action Chunking 执行 ══════════════════════
        ▼
  每次推理得到未来 50 步动作
  执行前 K 步（如 K=1~50），然后重新推理
```

### 10.2 STEP 1：反归一化（Unnormalize）

源码位于 `transforms.py:149-181`

训练时动作被归一化为零均值单位方差（z-score），推理输出需要还原：

```python
# Z-score 反归一化（默认）
actions = raw_actions * (std + 1e-6) + mean

# 分位数反归一化（use_quantile_norm=True 时）
# 训练时: normalized = (x - q01) / (q99 - q01) * 2 - 1   → [-1, 1]
# 推理时: actions = (raw + 1) / 2 * (q99 - q01) + q01
```

`norm_stats` 来自训练数据集的统计信息，保存在 checkpoint 目录的 `assets/` 下，由 `checkpoints.load_norm_stats()` 加载。

```
NormStats:
  mean  (32,)   ← 各动作维度的均值
  std   (32,)   ← 各动作维度的标准差
  q01   (32,)   ← 可选：1st 百分位（分位数归一化用）
  q99   (32,)   ← 可选：99th 百分位
```

### 10.3 STEP 2：机器人特定输出变换

不同机器人对动作的定义（维度、单位、坐标系）各不相同，每个 policy 有对应的 `XXXOutputs` 变换。

#### LIBERO（7维）

源码位于 `policies/libero_policy.py:86-100`

```python
class LiberoOutputs(DataTransformFn):
    def __call__(self, data):
        # 模型输出 32 维，实际只用前 7 维（末端执行器 delta 位姿 × 6 + 夹爪）
        # 后 25 维是 padding，直接丢弃
        return {"actions": np.asarray(data["actions"][:, :7])}

# 最终输出: (50, 7)
# 含义:  [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper_open]
```

#### ALOHA（14维，双臂）

源码位于 `policies/aloha_policy.py:90-101`

```python
class AlohaOutputs(DataTransformFn):
    def __call__(self, data):
        actions = np.asarray(data["actions"][:, :14])
        return {"actions": _encode_actions(actions, adapt_to_pi=self.adapt_to_pi)}

def _encode_actions(actions, adapt_to_pi):
    if adapt_to_pi:
        # 1. 关节方向翻转（pi0 训练坐标与 ALOHA 物理坐标轴方向不同）
        actions = joint_flip_mask * actions    # [1,-1,-1,1,1,1,1, 1,-1,-1,1,1,1,1]
        # 2. 夹爪角度换算（pi0 内部角度 → ALOHA 归一化位置）
        actions[:, [6, 13]] = gripper_from_angular(actions[:, [6, 13]])
    return actions

# 最终输出: (50, 14)
# 含义:  [左臂6关节角度, 左夹爪, 右臂6关节角度, 右夹爪]  (弧度/归一化)
```

#### DROID（8维，单臂）

```
actions[:, :8]:  [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, 终端执行旋转, 夹爪]
```

### 10.4 STEP 3：Action Chunking 与执行

模型每次推理输出 **50 步**的动作序列（action horizon = 50），但机器人每个控制周期只执行 **1 步或若干步**。

```
推理节奏（典型）:

  t=0: 推理 → 得到 actions[0:50]
       执行 actions[0]（或 actions[0:k]）
       执行 actions[1]
       ...
       执行 actions[k-1]

  t=k: 重新推理 → 得到新的 actions[0:50]
       执行 actions[0]
       ...

  k 通常设为 1（每步重新推理，最高频率更新）
  或更大值（降低推理频率，牺牲反应速度换吞吐量）
```

**为什么预测 50 步而不是 1 步？**
- 单步预测难以保证动作序列的时序连贯性（如抓取动作需要平滑轨迹）
- 50 步的 chunk 让模型能规划连续动作段，而不是孤立的单帧决策
- 通过 ODE 积分，每个 `action_token` 对应一个时间步的动作，50 个 token 覆盖未来 ~0.5-2 秒的轨迹

### 10.5 完整推理流水线图

```
  机器人传感器观测
  ┌─────────────────────────────────────────────────────────────┐
  │  相机图像 (3路)   关节状态   语言指令                          │
  └──────────────────────────┬──────────────────────────────────┘
                             │
                    input_transforms
                    ┌────────▼────────┐
                    │ XXXInputs       │  键名重映射、图像 uint8→[-1,1]
                    │ Normalize       │  (x - mean) / std
                    │ TokenizePrompt  │  文字→token IDs
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Policy.infer() │
                    │   ↓             │
                    │  embed_prefix   │  SigLIP + 语言嵌入 → (B,816,2048)
                    │  embed_suffix   │  状态 + 噪声动作 → (B,51,1024)
                    │  ODE ×10 步     │  Euler 积分去噪
                    │  action_out_proj│  Linear(1024→32)
                    └────────┬────────┘
                             │
                    output_transforms
                    ┌────────▼────────┐
                    │ Unnormalize     │  raw * std + mean
                    │ XXXOutputs      │  裁剪维度 + 坐标变换
                    └────────┬────────┘
                             │
                             ▼
                  actions (50, N)   ← 可直接发送给机器人
                  单位：弧度 / 归一化位置
                             │
                    Action Chunking
                    执行前 k 步，然后重新推理
```

### 10.6 动作维度 padding 的原因

模型统一使用 `action_dim=32`，但不同机器人实际维度各异：

| 机器人 | 实际维度 | padding 后 | 说明 |
|--------|---------|-----------|------|
| LIBERO | 7 | 32 | 末端执行器 6DoF + 夹爪 |
| ALOHA  | 14 | 32 | 双臂各 6关节 + 夹爪 |
| DROID  | 8 | 32 | 单臂 7DoF + 夹爪 |

- **训练时**：`InputTransform` 将实际动作 padding 到 32 维（补 0）再输入模型
- **推理时**：`OutputTransform` 取前 N 维，丢弃 padding 维度
- 统一 32 维使得同一个模型权重可以适配多种机器人（微调时只需更换 Input/Output Transform）
