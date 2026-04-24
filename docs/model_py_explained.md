# `model.py` 学习讲解（面向 `pi0/pi0.5`）

本文对应源码：
- [src/openpi/models/model.py](/storage/yukaichengLab/lishiwen/jiayusun/openpi/src/openpi/models/model.py)

目标：你读完后，能明确回答这几个问题：
1. 模型最开始接收的输入到底长什么样？
2. 这些输入在进入 `pi0.py` 之前做了哪些处理？
3. `BaseModelConfig` / `BaseModel` 各自负责什么？
4. checkpoint 参数是怎么恢复成可加载参数树的？

---

## 1. 文件在整体工程里的位置

`model.py` 是“模型层公共地基”。
- 它**不实现具体网络前向**（那在 `pi0.py`、`pi0_fast.py` 等）。
- 它负责统一：
  - 输入数据结构（`Observation`、`Actions`）
  - 输入预处理（`preprocess_observation`）
  - 模型抽象接口（`BaseModelConfig`、`BaseModel`）
  - 参数恢复工具（`restore_params`）

你可以把它理解成：
- `pi0.py` 是“具体算法实现”
- `model.py` 是“算法实现必须遵守的协议与公共工具”

---

## 2. 建议阅读顺序（按执行链路）

1. `Observation`（第 97 行）
2. `Observation.from_dict`（第 124 行）
3. `preprocess_observation`（第 163 行）
4. `BaseModel` 抽象接口（第 296 行）
5. `BaseModelConfig.load` / `restore_params`（第 259 / 320 行）

---

## 3. 部分一：基础类型与常量

### 3.1 `ModelType`（第 41 行）
定义支持的模型类型：
- `PI0`
- `PI0_FAST`
- `PI05`

作用：配置层通过它分发到不同实现。

### 3.2 `IMAGE_KEYS`（第 51 行）
```python
("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
```
作用：定义模型默认“必须有”的三路相机输入键。

### 3.3 `IMAGE_RESOLUTION`（第 60 行）
```python
(224, 224)
```
作用：统一模型输入图像分辨率。

---

## 4. 部分二：输入结构 `Observation` + `Actions`

### 4.1 `Observation`（第 97 行）
核心字段：
- `images`: `dict[str, float[*b, h, w, c]]`，图像值域约定 `[-1, 1]`
- `image_masks`: `dict[str, bool[*b]]`，每路图像是否有效
- `state`: `float[*b, s]`，低维机器人状态
- 可选文本字段：`tokenized_prompt`、`tokenized_prompt_mask`
- `pi0-fast` 扩展字段：`token_ar_mask`、`token_loss_mask`

### 4.2 `from_dict`（第 124 行）
把数据流水线产出的嵌套 dict 转成 `Observation`。

关键点：
1. 强约束：`tokenized_prompt` 和 `tokenized_prompt_mask` 必须同时出现或同时缺失。
2. 图像归一化：
   - numpy `uint8`：`[0,255] -> [-1,1]`
   - torch `uint8`：先转 float，再做数值归一化（该分支还做了维度变换用于 torch 路径）

### 4.3 `to_dict`（第 148 行）
把 dataclass 再映射回 dict：
- `images -> image`
- `image_masks -> image_mask`

### 4.4 `Actions`（第 159 行）
```python
Actions = at.Float[ArrayT, "*b ah ad"]
```
约定动作张量形状：
- `*b`: 批次维
- `ah`: action horizon（动作序列长度）
- `ad`: action dim（动作维）

---

## 5. 部分三：输入预处理 `preprocess_observation`

函数位置：第 163 行。

这是 `pi0.compute_loss` / `pi0.sample_actions` 一开始都会调用的入口预处理。

### 5.1 它做了什么

1. 校验必须图像键是否齐全
- 若 `IMAGE_KEYS` 中任意键缺失，直接 `ValueError`。

2. 图像分辨率对齐
- 若输入不是 `(224,224)`，调用 `image_tools.resize_with_pad`。

3. 训练增强（仅 `train=True`）
- 把图像从 `[-1,1]` 映射到 `[0,1]` 供 `augmax` 使用。
- 非 wrist 相机：`RandomCrop + Resize + Rotate`
- 所有相机：`ColorJitter`
- 用 `jax.vmap` + 每样本独立 RNG 批处理增强
- 再映回 `[-1,1]`

4. mask 补全
- 如果某路图像没有提供 mask，则默认全 `True`（不屏蔽）。

5. 返回新的 `Observation`
- 保留原 `state`/文本字段
- 更新 `images` 和 `image_masks`

### 5.2 你最需要记住的一句

`preprocess_observation` 是“模型输入最后一道闸门”：
- 形状统一
- 视角齐全
- 值域统一
- 训练增强
- mask 完整

---

## 6. 部分四：配置抽象 `BaseModelConfig`

类位置：第 238 行。

这是“如何创建/加载模型”的抽象协议。

### 6.1 必须由子类实现
- `model_type`
- `create(rng)`
- `inputs_spec(batch_size)`

### 6.2 `load`（第 259 行）流程

1. `eval_shape(self.create, key)` 先构建参数树模板（不做真实参数分配）
2. `nnx.split` 拿到 `graphdef + state`
3. 可选移除多余参数（`intersect_trees`）
4. 校验参数树结构与 shape（dtype 可放宽）
5. 把参数写回 state，再 `nnx.merge` 得到可运行模型

### 6.3 `load_pytorch`（第 272 行）
走 PyTorch 兼容路径：
- 构建 `PI0Pytorch`
- 用 safetensors 加载权重

### 6.4 `fake_obs` / `fake_act`
基于 `inputs_spec` 生成“全 1 假输入”，常用于：
- lazy init
- shape tracing
- dry-run

---

## 7. 部分五：模型抽象 `BaseModel`

类位置：第 296 行。

定义所有模型必须实现的两个核心行为：
- `compute_loss(...)`：训练
- `sample_actions(...)`：推理采样

`pi0.py` 就是在实现这两个抽象方法。

---

## 8. 部分六：参数恢复 `restore_params`

函数位置：第 320 行。

作用：从 Orbax checkpoint 读参数，并标准化成 NNX 期望的“pure dict”。

### 8.1 流程

1. 处理路径
- 本地路径转绝对路径
- `gs://` 路径保持原样

2. 默认 sharding
- 若恢复为 `jax.Array` 且没传 sharding，默认全设备复制分片

3. 读取 checkpoint
- 先拿 metadata
- 再按 `restore_type / dtype / sharding` 执行 restore

4. 清理 NNX 包装
- 如果叶子路径都以 `"value"` 结尾，去掉这层后再返回

### 8.2 为什么要去 `value`

训练时若通过某些 NNX state 保存路径，叶子会多包一层 `value`；
而模型加载通常用“pure dict”，所以这里做统一清洗。

---

## 9. 一条完整调用链（训练时）

```text
原始 batch(dict)
  -> Observation.from_dict
  -> preprocess_observation(train=True)
  -> pi0.compute_loss(...)
     -> embed_prefix / embed_suffix / transformer forward
  -> loss
```

推理时几乎一样，只是进入 `pi0.sample_actions(...)`，并且通常 `train=False` 不做增强。

---

## 10. 常见易混点

1. `Observation.images` 的约定是 `[-1,1]`，不是 `[0,1]`。增强时才临时转 `[0,1]`。
2. `image_mask` 缺失不会报错，会默认全 True。
3. `BaseModelConfig.load` 与 `restore_params` 分工不同：
   - `restore_params`：从磁盘恢复“参数树”
   - `load`：把参数树装配进“模型结构”
4. `BaseModel` 只定义接口，不做具体前向。

---

## 11. 下一步建议

如果你现在继续学 `pi0`，建议立刻对照看这两个函数：
1. `pi0.py::compute_loss`：把 `model.py` 的输入协议接到 flow matching 训练目标。
2. `pi0.py::sample_actions`：把同一输入协议接到采样/控制输出。

