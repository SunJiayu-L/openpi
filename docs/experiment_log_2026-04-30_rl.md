# 实验日志 — 2026-04-30 RL 训练

> 创建：2026-04-30 CST  
> 项目路径：`/storage/yukaichengLab/lishiwen/jiayusun/openpi`  
> 框架路径：`/storage/yukaichengLab/lishiwen/jiayusun/RLinf-openpi`

---

## 一、背景与动机

SFT 方案下 task8（"put both moka pots on the stove"）成功率天花板约 70%，无法通过增加训练步数或数据增强突破。根本限制：
- 仅有 29 个 demo，50 种初始状态
- 双物体串行抓取导致结构性上限
- 历史最佳：`ft_from_wudi_4task_1k/9000` task8 ≈ 67%，`base_25k` task8 = 70%

**决策**：转向 RL 在线训练，使用 GRPO 算法。

---

## 二、框架选择

使用 **RLinf-openpi**（`/storage/yukaichengLab/lishiwen/jiayusun/RLinf-openpi`）：
- 内置 JAX → PyTorch safetensors 转换器
- 已有 libero_10 GRPO 参考 config
- 已有可用 sbatch 模板
- 针对 LIBERO + MuJoCo 的已知问题有文档记录（TROUBLESHOOTING.md）

---

## 三、起点 Checkpoint

**选择**：`pi05_libero/my_experiment/25000`（base_25k）

| 指标 | 值 |
|---|---|
| avg(4 suites) | **97.5%** ⭐ 历史最优 |
| task8 | 70% |
| 格式 | JAX orbax |

选择理由：base_25k 综合成绩最优，task8 也是 SFT 方案中最高的基底。

---

## 四、JAX → PyTorch 转换（Job 34751）

**脚本**：`rlinf/utils/ckpt_convertor/convert_openpi_jax_to_python.py`

```bash
python rlinf/utils/ckpt_convertor/convert_openpi_jax_to_python.py \
    --checkpoint_dir .../checkpoints/pi05_libero/my_experiment/25000 \
    --config_name pi05_libero \
    --output_path .../checkpoints/pt_converted/base_25k \
    --precision bfloat16
```

**结果**：
- 输出：`pt_converted/base_25k/model.safetensors`（6.8G，bfloat16）
- 同时复制：`pt_converted/base_25k/assets/physical-intelligence/libero/norm_stats.json`

**坑**：登录节点 OOM（exit 137），需放在计算节点（128G RAM）上跑。

**norm_stats 路径修正**：RLinf 期望 norm stats 位于 `checkpoint_dir/physical-intelligence/libero/norm_stats.json`，而转换脚本输出到 `assets/` 子目录。手动复制：
```bash
mkdir -p pt_converted/base_25k/physical-intelligence/libero/
cp pt_converted/base_25k/assets/physical-intelligence/libero/norm_stats.json \
   pt_converted/base_25k/physical-intelligence/libero/norm_stats.json
```

---

## 五、RL 训练配置

**Config 文件**：`RLinf-openpi/examples/embodiment/config/libero_10_grpo_pi05_base25k.yaml`  
**SBATCH 脚本**：`RLinf-openpi/run_libero10_grpo_base25k_4gpu.sbatch`

### 5.1 最终有效配置

```yaml
cluster:
  component_placement:
    env: 0          # GPU 0：MuJoCo EGL 渲染（DummyVectorEnv，顺序执行）
    rollout: 1      # GPU 1：模型推理
    actor: 2-3      # GPU 2-3：FSDP 训练，actor_world_size=2

algorithm:
  group_size: 8
  rollout_epoch: 16
  kl_beta: 0.0      # GRPO 不用 KL 惩罚
  filter_rewards: True
  rewards_lower_bound: 0.1
  rewards_upper_bound: 0.9

env:
  train:
    total_num_envs: 8          # 最小值 = group_size
    max_episode_steps: 520

actor:
  micro_batch_size: 32
  global_batch_size: 64        # 2 GPU × 32
  enable_offload: True

rollout:
  enable_offload: True
  unnorm_key: libero_10

runner:
  logger_backends: ["tensorboard"]
  val_check_interval: 20
  save_interval: 20
```

### 5.2 Logging

- **TensorBoard**（无网络依赖）：`../results/` 目录
- **WandB**：强制 `WANDB_MODE=offline`（计算节点无外网）

---

## 六、调试过程与已解决的问题

### 问题 1：paligemma tokenizer 下载失败

**错误**：`ClientConnectorDNSError: Cannot connect to host storage.googleapis.com`

**原因**：计算节点无外网，tokenizer 不在 jiayusun 的 cache 路径下。

**修复**：
```bash
cp /storage/yukaichengLab/lishiwen/.cache/openpi/big_vision/paligemma_tokenizer.model \
   /storage/yukaichengLab/lishiwen/jiayusun/.cache/openpi/big_vision/
```

### 问题 2：norm_stats 路径不匹配

**错误**：`FileNotFoundError: Norm stats file not found at: .../pt_converted/base_25k/physical-intelligence/libero/norm_stats.json`

**修复**：见 §四。

### 问题 3：pybind11 segfault + EnvGroup OOM（TROUBLESHOOTING Problem 5 & 6）

**错误**：
```
Rollout(rank=0): pybind11_object_dealloc(): Tried to deallocate unregistered instance
EnvGroup(rank=0): tried to allocate 139474767971430 bytes
```

**根因**：
1. `component_placement: all` → env/rollout/actor 混用 4 块 GPU，FSDP no_shard 撑爆显存
2. `total_num_envs: 32` 过多

**修复**：
- 分离 GPU 分配（env:0, rollout:1, actor:2-3）
- `total_num_envs: 8`，`rollout_epoch: 16`
- `enable_offload: True`

### 问题 4：total_num_envs 不能被 group_size 整除

**错误**：`AssertionError: env.train.total_num_envs // env_world_size // pipeline_stage_num must be divisible by the group_size`

**修复**：`total_num_envs` 从 4 改为 8（= group_size）。

---

## 七、训练历史

### 7.1 GRPO（已放弃）

Job 34757 使用 GRPO，速度 ~1865s/step（约 31 分钟），ETA 512 小时。太慢，已 scancel。

### 7.2 PPO 初跑（Job 34818）

切换框架为 **RLinf**（`/storage/yukaichengLab/lishiwen/jiayusun/RLinf`），算法改为 **PPO**。

| 项目 | 值 |
|---|---|
| 节点 | gnho031（4× H800） |
| 配置 | `libero_10_ppo_pi05_h800_4gpu.yaml` |
| 速度 | ~956s/step，32 envs，512 traj/step |
| 整体成功率 | 96–98%（所有任务聚合） |
| 停止原因 | SLURM 时间限制，Ray RPC 断开 |
| 保存的 checkpoint | step 20/40/60/80/100/120 |

Checkpoint 路径：
```
RLinf/logs/20260503-15:42:22-libero_10_ppo_pi05_h800_4gpu/
  libero_10_ppo_pi05_from_base25k/checkpoints/global_step_{20,40,60,80,100,120}/
    actor/model_state_dict/full_weights.pt   # 8.0GB，torch.save 格式
```

### 7.3 PPO 断点续训（Job 35259，进行中）

从 step_120 续训，`save_interval` 改为 10。

```yaml
resume_dir: ".../checkpoints/global_step_120"
save_interval: 10
```

---

## 八、RL Checkpoint 评测流程

### 8.1 格式转换（重要）

RLinf 输出 `full_weights.pt`（torch.save），openpi `serve_policy.py` 需要 `model.safetensors`。
两者 key 结构基本相同，但需过滤以下 key：

| 过滤 key | 原因 |
|---|---|
| `value_head.*`（共 8 个） | RL 专用 critic head，eval 不需要 |
| `paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight` | tied weight 被重复保存，PI0Pytorch 中不存在此 key |

转换脚本：`openpi/convert_rl_ckpts_to_safetensors.sbatch`

norm_stats 需放两处：
- `{ckpt}/assets/physical-intelligence/libero/norm_stats.json`（openpi serve_policy 用）
- `{ckpt}/physical-intelligence/libero/norm_stats.json`（RLinf 用，备用）

### 8.2 ~/.libero/config.yaml 路径修复

原指向已删除的 `RLinf-openpi/.venv/libero/`，需修正为 `openpi/third_party/libero`：

```bash
LIBERO_ROOT="/storage/yukaichengLab/lishiwen/jiayusun/openpi/third_party/libero/libero/libero"
cat > ~/.libero/config.yaml << EOF
assets: ${LIBERO_ROOT}/assets
bddl_files: ${LIBERO_ROOT}/bddl_files
benchmark_root: ${LIBERO_ROOT}
datasets: ${LIBERO_ROOT}/../datasets
init_states: ${LIBERO_ROOT}/init_files
EOF
```

### 8.3 评测 Job（进行中）

转换后 checkpoint：`openpi/checkpoints/pt_rl/ppo_step{80,100,120}/model.safetensors`

评测 sbatch 串行提交（4 suite × 50 trials）：

| Job | Step | 脚本 |
|---|---|---|
| 35256 | step80 | `eval_ppo_step80_all_suites.sbatch` |
| 35257 | step100 | `eval_ppo_step100_all_suites.sbatch` |
| 35258 | step120 | `eval_ppo_step120_all_suites.sbatch` |

结果写入：`openpi/data/libero/videos/ppo_step{80,100,120}/{suite}/`

---

## 九、关键文件路径

| 用途 | 路径 |
|---|---|
| base_25k PT checkpoint | `openpi/checkpoints/pt_converted/base_25k/` |
| RL 训练 config | `RLinf/examples/embodiment/config/libero_10_ppo_pi05_h800_4gpu.yaml` |
| SBATCH 训练脚本 | `RLinf/run_libero10_ppo_pi05.sbatch` |
| TensorBoard 日志 | `RLinf/logs/20260503-*/` |
| RL checkpoint（PT） | `RLinf/logs/20260503-*/libero_10_ppo_pi05_from_base25k/checkpoints/` |
| RL checkpoint（safetensors，eval用）| `openpi/checkpoints/pt_rl/ppo_step{80,100,120}/` |
| 转换脚本 | `openpi/convert_rl_ckpts_to_safetensors.sbatch` |
| eval 脚本 | `openpi/eval_ppo_step{80,100,120}_all_suites.sbatch` |
| libero 路径配置 | `~/.libero/config.yaml` |
