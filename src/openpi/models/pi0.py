import logging

# 张量重排工具，用于把 mask 或 embedding 扩展到目标形状。
import einops
# Flax NNX：当前文件使用的主要模块/层定义。
import flax.nnx as nnx
# 旧版 Flax/JAX 模块桥接到 NNX 的适配器。
import flax.nnx.bridge as nnx_bridge
# JAX 核心库（随机数、控制流、精度控制等）。
import jax
# JAX NumPy 接口。
import jax.numpy as jnp
# 显式标记子类重写父类方法。
from typing_extensions import override

# 通用模型基类与数据结构（Observation/Actions/BaseModel）。
from openpi.models import model as _model
# Pi0/Pi0.5 配置定义。
from openpi.models import pi0_config
# Gemma（语言主干 + action expert）。
import openpi.models.gemma as _gemma
# SigLIP 视觉编码器。
import openpi.models.siglip as _siglip
# 项目内的数组类型标注与运行时 typecheck。
from openpi.shared import array_typing as at

# 统一 logger，便于在 openpi 命名空间下输出日志。
logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    # 把一维/可广播的自回归规则扩展到 batch 维。
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    # 通过累计和把“段边界”编码为可比较的段编号。
    cumsum = jnp.cumsum(mask_ar, axis=1)
    # 当前 query token 只能看见段编号不超过自己的 key token（因果或分块因果）。
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    # padding token 既不能作为 query 也不能作为 key。
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    # 同时满足“注意力拓扑”和“有效 token”。
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    # sin/cos 成对拼接，维度必须是偶数。
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    # 在 [0,1] 上均匀采样频率索引，再映射到 [min_period, max_period] 的几何级数周期。
    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    # 外积得到每个样本位置与每个频率的相位输入。
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    # 拼接 sin 与 cos 得到最终时间嵌入。
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


# ==============================
# Pi0 / Pi0.5 主模型（VLA + Flow Matching）
# 可分为 6 部分：
# 1) 工具函数：mask 构造与时间 sincos 嵌入
# 2) 初始化：视觉/语言主干与动作专家分支构建
# 3) Prefix 编码：图像 + 文本 token 化
# 4) Suffix 编码：状态/动作/时步 token 化（pi0 与 pi0.5 分叉）
# 5) 训练前向：流匹配目标构造与损失
# 6) 推理采样：带 KV cache 的 ODE 离散积分
# ==============================
class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        # 初始化基类中动作相关的基础参数（维度、horizon、最大 token 长度）。
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        # 记录是否启用 pi0.5 分支（影响状态输入和时间注入方式）。
        self.pi05 = config.pi05
        #! 获取 PaliGemma 主干配置。
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        #! 获取 action expert 配置。
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        # 构建双专家 Gemma，并桥接到 NNX。
        #! 交给gemma来融合。
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                # pi0.5 时在 action expert 里启用 adaRMSNorm 条件化。
                adarms=config.pi05,
            )
        )
        # 惰性初始化参数；pi0.5 只在第二个专家（action expert）启用 adaRMS 条件。
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        # 构建 SigLIP 图像编码器，输出宽度对齐到 PaliGemma 隐藏维。
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        # 用 fake observation 里的一张图进行惰性初始化。
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        # 统一封装视觉与语言主干。
        ###! 总模型 PaliGmma（llm+img）
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        # 将动作（或噪声动作）投影到 action expert 隐藏空间。
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            # pi0.5：时间嵌入走独立 MLP，作为 adaRMS 条件，不与动作 token 直接拼接。
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            # pi0：状态是连续向量，经线性层投影为一个 suffix token。
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            # pi0：动作 token 与时间 token 拼接后经 MLP 融合。
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        # 从 action expert 隐藏空间回投影到动作维度，得到速度场 v_t。
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        # 默认推理模式；外部 train()/eval() 会自动切换此标志。
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        # 每个 token 是否是有效输入（非 padding）。
        input_mask = []
        # 自回归分块规则（False=与前一个 token 共享同一可见域；True=开启新块）。
        ar_mask = []
        # prefix token 序列（图像 token + 文本 token）。
        tokens = []
        # embed images
        for name in obs.images:
            # 视觉编码：每路相机图像 -> patch token 序列。
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            # 收集图像 token。
            tokens.append(image_tokens)
            # 把每张图像的有效标志扩展到对应的 patch token 长度。
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            #!  图像 prefix 内采用全互看（同一块）。
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            # 文本 token 走 LLM embedding 层。
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            # 收集文本 token。
            tokens.append(tokenized_inputs)
            # 文本有效位。
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            # 图像与语言同属 prefix，全互看（不做因果隔离）。
            ar_mask += [False] * tokenized_inputs.shape[1]
        # 沿序列维拼接所有 prefix token。
        tokens = jnp.concatenate(tokens, axis=1)
        # 对齐后的 prefix 有效 mask。
        input_mask = jnp.concatenate(input_mask, axis=1)
        # Python list -> JAX array，便于后续 mask 计算。
        ar_mask = jnp.array(ar_mask)
        # 返回 prefix 表示、有效位、自回归规则。
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        # suffix 有效位（state/action token）。
        input_mask = []
        # suffix 自回归规则。
        ar_mask = []
        # suffix token 本体。
        tokens = []
        if not self.pi05:
            # add a single state token
            # pi0：把连续状态投影为 1 个 state token，放在 action token 之前。
            state_token = self.state_proj(obs.state)[:, None, :]
            # 追加 state token。
            tokens.append(state_token)
            # state token 全为有效位。
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            # state token 作为新块起点，隔离 prefix -> suffix 的反向依赖。
            ar_mask += [True]

        # 把 noisy action 序列投影到 expert 隐空间。
        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        # 将连续时间 t 编码成 sin/cos 向量（维度与 action token 隐层一致）。
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            # pi0.5：时间条件经两层 MLP + swish，作为 adaRMS 条件输入。
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            # pi0.5：动作 token 不与时间向量显式拼接。
            action_expert_tokens = action_tokens
            # 仅通过 adarms_cond 注入时间信息。
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            # pi0：把每个样本的时间向量复制到整个 action horizon。
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            # pi0：动作与时间在特征维拼接。
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            # 通过 MLP 融合动作与时间。
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            # 融合后的 token 送入 action expert。
            action_expert_tokens = action_time_tokens
            # pi0：不用 adaRMS 条件。
            adarms_cond = None
        # 追加 action token 序列。 
        tokens.append(action_expert_tokens)
        # action token 全有效。
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        #! action 序列为因果块：第一个 token 开新块，后续 token 递增因果可见域。
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        # 拼接完整 suffix token。
        tokens = jnp.concatenate(tokens, axis=1)
        # 拼接 suffix 有效位。
        input_mask = jnp.concatenate(input_mask, axis=1)
        # list -> array。
        ar_mask = jnp.array(ar_mask)
        # 返回 suffix 表示、mask，以及 pi0.5 的 adarms 条件向量。
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        # 拆分随机源：预处理增强、噪声采样、时间采样。
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        # 对 observation 做训练/推理一致的标准预处理。
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        # 提取 batch 维，不含 (action_horizon, action_dim) 两个末尾维度。
        batch_shape = actions.shape[:-2]
        # 流匹配中的噪声端样本 z ~ N(0, I)。
        noise = jax.random.normal(noise_rng, actions.shape)
        # 从 Beta(1.5, 1) 采样 t，并限制在 (0.001, 1.0) 内避免端点数值问题。
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        # 扩展 t 到动作张量形状，便于广播计算。
        time_expanded = time[..., None, None]

        ###! 构造flow matching 目标
        # 线性插值构造 x_t = t * noise + (1 - t) * actions。
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        # 目标速度场 u_t = noise - actions。
        u_t = noise - actions


        #! 前向传播
        # one big forward pass of prefix + suffix at once
        # 编码 prefix（图像+文本）。
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        # 编码 suffix（状态/动作/时间）。
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        # 拼接完整序列有效位。
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        # 把 prefix 和 suffix 的自回归分块规则拼成一条完整序列规则。
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        #  根据“有效位 + 分块规则”生成最终注意力掩码（[B, T, T]）
        attn_mask = make_attn_mask(input_mask, ar_mask)
        # 生成每个 token 的位置 id（给 RoPE/位置编码用）。
        positions = jnp.cumsum(input_mask, axis=1) - 1
        ###! 模型推理
        # 单次前向同时计算 prefix/suffix，adaRMS 条件只喂给 action expert 路径。
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        
        # 只取 suffix 末尾 action_horizon 段并回投影得到预测速度 v_t。
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        #! 计算损失
        # Flow Matching 的逐动作维 MSE，最后在动作维上求均值。
        return jnp.mean(jnp.square(v_t - u_t), axis=-1)



    ###############################
    #### 采样动作
    ####从一段高斯噪声动作 x_1 出发，反复调用模型预测当前速度场 v_t，
    ####再用 Euler 法把噪声一步步积分到干净动作 x_0。
    ###############################
    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        # 推理时固定 train=False，做确定性预处理。
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        # 反向积分步长（从 t=1 积到 t=0）。
        dt = -1.0 / num_steps
        # 当前 batch 大小。
        batch_size = observation.state.shape[0]
        if noise is None:
            # 若用户未提供初始噪声，则采样标准高斯作为 x_1。
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        # 先跑一次 prefix，把其 K/V 写入缓存，后续每步只增量计算 suffix。
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            # carry = (当前轨迹点 x_t, 当前时间 t)。
            x_t, time = carry
            # 把标量 t 扩展到 batch，编码当前 suffix。
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            # 合并后，suffix query 既可看 prefix key/value，也可看 suffix 内因果可见部分。
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            # 形状安全检查，防止拼接维度错位。
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            # suffix 的位置索引从 prefix 长度之后继续累加。
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            # 仅输入 suffix（prefix 用缓存），得到当前时刻速度估计。
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            # 当前调用未重新计算 prefix 分支，输出应为 None。
            assert prefix_out is None
            # 取 action 区段并映射为 v_t。
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            # Euler 步进：x_{t+dt} = x_t + dt * v_t，同时更新时间。
            return x_t + dt * v_t, time + dt

        def cond(carry):
            # while_loop 条件函数签名要求接收完整 carry。
            x_t, time = carry
            # robust to floating-point error
            # 给结束条件留出半个步长容差，避免浮点误差导致多迭代/少迭代。
            return time >= -dt / 2

        # 从噪声端 x_1 开始积分到 x_0，得到最终动作序列。
        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        # 返回预测动作。
        return x_0
