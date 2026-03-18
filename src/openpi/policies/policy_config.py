import logging
import os
import pathlib
import shutil
from typing import Any

import flax.nnx as nnx
import jax.numpy as jnp
import orbax.checkpoint as ocp

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
import openpi.shared.normalize as _normalize
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms
from openpi.policies.model_merging import merging_functions


def save_merged_checkpoint(
    model: _model.BaseModel,
    save_path: pathlib.Path | str,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    asset_id: str | None = None,
) -> pathlib.Path:
    """Save merged model params and norm_stats to a checkpoint directory.

    The saved checkpoint is compatible with ``create_trained_policy`` and
    ``model.restore_params``, so the merged model can be loaded later without
    re-merging.

    Directory layout::

        save_path/
            params/          # orbax checkpoint (same format as training checkpoints)
            assets/
                {asset_id}/
                    norm_stats.json

    Returns:
        The resolved save_path.
    """
    save_path = pathlib.Path(save_path).resolve()
    save_path.mkdir(parents=True, exist_ok=True)

    # Save params.
    params_path = save_path / "params"
    params = nnx.state(model).to_pure_dict()
    logging.info("Saving merged params to %s", params_path)
    with ocp.PyTreeCheckpointer() as ckptr:
        ckptr.save(params_path, ocp.args.PyTreeSave({"params": params}))

    # Save norm_stats.
    if norm_stats is not None and asset_id is not None:
        _normalize.save(save_path / "assets" / asset_id, norm_stats)
        logging.info("Saved norm_stats to %s", save_path / "assets" / asset_id)

    logging.info("Merged checkpoint saved to %s", save_path)
    return save_path


def create_merged_policy(
    train_config: _config.TrainConfig,
    checkpoint_dirs: list[pathlib.Path | str],
    merging_fn: str,
    merging_fn_kwargs: dict[str, Any] | None = None,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    save_path: pathlib.Path | str | None = None,
) -> _policy.Policy:
    """Create a policy by merging multiple checkpoints.

    Norm stats are loaded from train_config.assets_dirs (the config-level assets
    directory) rather than from any single checkpoint, because merged models may
    combine checkpoints trained on different data subsets with different per-subset
    norm stats.  Using the config-level assets ensures the statistics cover the
    full data distribution.

    If ``save_path`` is provided, the merged params and norm_stats are saved to
    disk so the checkpoint can be loaded later via ``create_trained_policy``.
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dirs = [download.maybe_download(str(d)) for d in checkpoint_dirs]

    logging.info("Available merging functions: %s, using %s", list(merging_functions.keys()), merging_fn)
    fn = merging_functions[merging_fn]
    model = fn(train_config, checkpoint_dirs, merging_fn_kwargs)

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        # Load norm stats from the config assets dir (covers all suites) instead
        # of a single checkpoint, so the merged model normalizes correctly across
        # different task distributions.
        norm_stats = _checkpoints.load_norm_stats(train_config.assets_dirs, data_config.asset_id)

    # Save merged checkpoint to disk if requested.
    if save_path is not None:
        save_merged_checkpoint(model, save_path, norm_stats, data_config.asset_id)

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
    )


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    # Check if this is a PyTorch model by looking for model.safetensors
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)

    logging.info("Loading model...")
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    # Determine the device to use for PyTorch models
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )
