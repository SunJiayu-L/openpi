"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

# =========================
# 1) 依赖导入与模块别名
# 作用：导入数值计算、进度条、CLI 解析，以及 openpi 的模型/数据/归一化模块。
# =========================
import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


# =========================
# 2) 预处理变换：移除字符串字段
# 作用：过滤 batch 中的字符串（例如文本描述），
# 因为计算 norm stats 只关心数值张量，且字符串不适合进入 JAX 数值流水线。
# =========================
class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


# =========================
# 3) Torch 数据加载器构建
# 作用：面向常规 LeRobot/Torch 数据源，创建数据集、应用变换并返回 DataLoader。
# =========================
def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    # repo_id 用于定位数据与输出目录，是计算统计量的必要字段。
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")

    # 先构建原始 torch dataset，再串联重排/预处理/去字符串三个变换。
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )

    # 当指定 max_frames 时，使用随机采样并限制 batch 数；
    # 否则遍历完整数据集（整除 batch_size 后的 batch 数）。
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


# =========================
# 4) RLDS 数据加载器构建
# 作用：面向 RLDS 数据源，构建可迭代数据集并应用同样的数据变换。
# =========================
def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: 当前长度在 DROID 场景下是硬编码逻辑。
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


# =========================
# 5) 主流程：加载配置 -> 选择数据源 -> 累积统计 -> 保存结果
# 作用：按 config_name 计算 state/actions 的归一化统计并写入 assets 目录。
# =========================
def main(config_name: str, max_frames: int | None = None):
    # 读取训练配置，并创建该配置对应的数据配置（包含 transforms、repo_id 等）。
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    # 根据 data_config 是否提供 rlds_data_dir，选择 RLDS 或 Torch 的加载路径。
    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    # 目前仅统计这两个关键字段：机器人状态 state 与动作 actions。
    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    # 逐 batch 增量更新统计量（均值/方差/分位数等由 RunningStats 内部维护）。
    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    # 导出最终统计结果并保存到 assets/{config_name}/{repo_id}/norm_stats.json。
    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


# =========================
# 6) CLI 入口
# 作用：通过命令行参数调用 main，例如：
# uv run scripts/compute_norm_stats.py --config-name pi05_libero
# =========================
if __name__ == "__main__":
    tyro.cli(main)
