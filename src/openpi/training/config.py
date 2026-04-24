"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.polaris_config as polaris_config
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.libero_split as _libero_split
import openpi.training.libero_suite_episodes as _suite_eps
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Optional local root for the primary LeRobot dataset.
    lerobot_root: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # If provided, only these episode indices will be used from the LeRobot dataset.
    episodes: list[int] | None = None

    # Additional LeRobot datasets to concatenate with the main dataset.
    # Each entry is (repo_id, root_dir, episodes). root_dir can be None to use the default HF cache.
    # episodes can be None to use all episodes from that dataset.
    extra_lerobot_datasets: Sequence[tuple[str, str | None, Sequence[int] | None]] = ()

    # Optional sampling weights for the concatenated LeRobot datasets.
    # If provided, the length must equal: 1 + len(extra_lerobot_datasets),
    # where index 0 is the primary dataset and the rest follow extra_lerobot_datasets order.
    # Weights control dataset-level sampling ratio (implemented with replacement sampling).
    dataset_mix_weights: Sequence[float] | None = None

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.

    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (
        droid_rlds_dataset.RLDSDataset(
            name="droid",
            version="1.0.1",
            weight=1.0,
            filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        ),
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            datasets=self.datasets,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0_config.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("/storage/yukaichengLab/lishiwen/.cache/openpi/openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/storage/yukaichengLab/lishiwen/.cache/openpi/openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi05_libero_from_merged_4task_gd",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            assets=AssetsConfig(
                assets_dir="/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero/my_experiment/5000/assets",
            ),
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=128,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero_4task_merge_gd_frozen_large/0/params"
        ),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi05_libero_plus_lerobot",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="libero_plus_lerobot",
            assets=AssetsConfig(asset_id="pi05_libero_plus_lerobot"),
            base_config=DataConfig(
                prompt_from_task=True,
                lerobot_root="/storage/yukaichengLab/lishiwen/jiayusun/libero_plus_lerobot",
            ),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi05_libero_no10",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                prompt_from_task=True,
                # Train on 36 tasks (1515 episodes).
                # Excludes first task of each suite (task 0, 10, 20, 30) for held-out testing.
                episodes=_libero_split.TRAIN_EPISODES,
            ),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    #
    # Single-suite fine-tuning configs (pi0 x 4 suites + pi0.5 x 4 suites = 8 configs)
    #
    # --- pi0 single-suite ---
    TrainConfig(
        name="pi0_libero_10",
        model=pi0_config.Pi0Config(),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True, episodes=_suite_eps.LIBERO_10_EPISODES),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/storage/yukaichengLab/lishiwen/.cache/openpi/openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_goal",
        model=pi0_config.Pi0Config(),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True, episodes=_suite_eps.LIBERO_GOAL_EPISODES),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/storage/yukaichengLab/lishiwen/.cache/openpi/openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_object",
        model=pi0_config.Pi0Config(),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True, episodes=_suite_eps.LIBERO_OBJECT_EPISODES),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/storage/yukaichengLab/lishiwen/.cache/openpi/openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_spatial",
        model=pi0_config.Pi0Config(),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True, episodes=_suite_eps.LIBERO_SPATIAL_EPISODES),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/storage/yukaichengLab/lishiwen/.cache/openpi/openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
    ),
    # --- pi0.5 single-suite ---
    TrainConfig(
        name="pi05_libero_10",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True, episodes=_suite_eps.LIBERO_10_EPISODES),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=10_000, peak_lr=5e-5, decay_steps=1_000_000, decay_lr=5e-5),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi05_libero_goal",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True, episodes=_suite_eps.LIBERO_GOAL_EPISODES),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=10_000, peak_lr=5e-5, decay_steps=1_000_000, decay_lr=5e-5),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi05_libero_object",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True, episodes=_suite_eps.LIBERO_OBJECT_EPISODES),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=10_000, peak_lr=5e-5, decay_steps=1_000_000, decay_lr=5e-5),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi05_libero_spatial",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True, episodes=_suite_eps.LIBERO_SPATIAL_EPISODES),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=10_000, peak_lr=5e-5, decay_steps=1_000_000, decay_lr=5e-5),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
    ),
    # --- pi0.5 single-suite fine-tuning from pi05_libero/my_experiment/29999 ---
    TrainConfig(
        name="pi05_libero_10_from_base29999",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            assets=AssetsConfig(
                assets_dir="/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero/my_experiment/29999/assets",
            ),
            base_config=DataConfig(prompt_from_task=True, episodes=_suite_eps.LIBERO_10_EPISODES),
            extra_delta_transform=False,
        ),
        batch_size=128,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=5e-5, decay_steps=30_000, decay_lr=5e-6),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero/my_experiment/29999/params"
        ),
        num_train_steps=30_000,
    ),
    # --- pi0.5 libero-10 fine-tuning from pi05_libero/my_experiment/10000 (full norm_stats) ---
    TrainConfig(
        name="pi05_libero_10_from_pi05libero_10k",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            assets=AssetsConfig(
                assets_dir="/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero/my_experiment/10000/assets",
            ),
            base_config=DataConfig(prompt_from_task=True, episodes=_suite_eps.LIBERO_10_EPISODES),
            extra_delta_transform=False,
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=5e-5, decay_steps=30_000, decay_lr=5e-6),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero/my_experiment/10000/params"
        ),
        num_train_steps=30_000,
    ),
    # --- pi0.5 libero-object fine-tuning from pi05_libero/my_experiment/10000 (full norm_stats) ---
    TrainConfig(
        name="pi05_libero_object_from_pi05libero_10k",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            assets=AssetsConfig(
                assets_dir="/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero/my_experiment/10000/assets",
            ),
            base_config=DataConfig(prompt_from_task=True, episodes=_suite_eps.LIBERO_OBJECT_EPISODES),
            extra_delta_transform=False,
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=5e-5, decay_steps=30_000, decay_lr=5e-6),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero/my_experiment/10000/params"
        ),
        num_train_steps=30_000,
    ),
    # --- pi0.5 libero-goal fine-tuning from pi05_libero/my_experiment/10000 (full norm_stats) ---
    TrainConfig(
        name="pi05_libero_goal_from_pi05libero_10k",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            assets=AssetsConfig(
                assets_dir="/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero/my_experiment/10000/assets",
            ),
            base_config=DataConfig(prompt_from_task=True, episodes=_suite_eps.LIBERO_GOAL_EPISODES),
            extra_delta_transform=False,
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=5e-5, decay_steps=30_000, decay_lr=5e-6),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero/my_experiment/10000/params"
        ),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi05_libero_goal_from_base29999",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            assets=AssetsConfig(
                assets_dir="/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero/my_experiment/29999/assets",
            ),
            base_config=DataConfig(prompt_from_task=True, episodes=_suite_eps.LIBERO_GOAL_EPISODES),
            extra_delta_transform=False,
        ),
        batch_size=128,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=5e-5, decay_steps=30_000, decay_lr=5e-6),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero/my_experiment/29999/params"
        ),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi05_libero_object_from_base29999",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            assets=AssetsConfig(
                assets_dir="/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero/my_experiment/29999/assets",
            ),
            base_config=DataConfig(prompt_from_task=True, episodes=_suite_eps.LIBERO_OBJECT_EPISODES),
            extra_delta_transform=False,
        ),
        batch_size=128,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=5e-5, decay_steps=30_000, decay_lr=5e-6),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero/my_experiment/29999/params"
        ),
        num_train_steps=30_000,
    ),
    # --- pi0.5 libero-spatial fine-tuning from pi05_libero/my_experiment/10000 (full norm_stats) ---
    TrainConfig(
        name="pi05_libero_spatial_from_pi05libero_10k",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            assets=AssetsConfig(
                assets_dir="/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero/my_experiment/10000/assets",
            ),
            base_config=DataConfig(prompt_from_task=True, episodes=_suite_eps.LIBERO_SPATIAL_EPISODES),
            extra_delta_transform=False,
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=5e-5, decay_steps=30_000, decay_lr=5e-6),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero/my_experiment/10000/params"
        ),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi05_libero_spatial_from_base29999",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            assets=AssetsConfig(
                assets_dir="/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero/my_experiment/29999/assets",
            ),
            base_config=DataConfig(prompt_from_task=True, episodes=_suite_eps.LIBERO_SPATIAL_EPISODES),
            extra_delta_transform=False,
        ),
        batch_size=128,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=5e-5, decay_steps=30_000, decay_lr=5e-6),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero/my_experiment/29999/params"
        ),
        num_train_steps=30_000,
    ),
    # --- pi0.5 LIBERO-10 single-task retraining from custom base (10k step) ---
    # Task ids are within libero_10 suite indexing: 0, 6, 9.
    # Episode ids below are extracted from local LeRobot dataset at:
    # /storage/yukaichengLab/lishiwen/jiayusun/huggingface/lerobot
    TrainConfig(
        name="pi05_libero10_task0_retrain",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            assets=AssetsConfig(asset_id="pi05_libero10_task0_retrain"),
            base_config=DataConfig(
                prompt_from_task=True,
                episodes=[0, 18, 22, 33, 58, 85, 88, 105, 107],
            ),
            extra_delta_transform=False,
        ),
        batch_size=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero_RETRAIN_base/pi05_libero_RETRAIN_base/10000/params"
        ),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi05_libero10_task6_retrain",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            assets=AssetsConfig(asset_id="pi05_libero10_task6_retrain"),
            base_config=DataConfig(
                prompt_from_task=True,
                episodes=[10, 20, 23, 46, 51, 54, 67, 70, 73, 86, 100, 106],
            ),
            extra_delta_transform=False,
        ),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero_RETRAIN_base/pi05_libero_RETRAIN_base/10000/params"
        ),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi05_libero10_task9_retrain",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            assets=AssetsConfig(asset_id="pi05_libero10_task9_retrain"),
            base_config=DataConfig(
                prompt_from_task=True,
                episodes=[27, 28, 47, 55, 61, 64, 81, 103, 104, 109],
            ),
            extra_delta_transform=False,
        ),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero_RETRAIN_base/pi05_libero_RETRAIN_base/10000/params"
        ),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi05_libero10_task4_ep7_retrain",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                prompt_from_task=True,
                episodes=[7],
            ),
            extra_delta_transform=False,
        ),
        batch_size=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200,
            peak_lr=5e-5,
            decay_steps=5_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero/my_experiment/5000/params"
        ),
        num_train_steps=5_001,
        save_interval=500,
        log_interval=50,
        keep_period=500,
    ),
    TrainConfig(
        name="pi05_libero10_task4_all43_retrain_100",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                prompt_from_task=True,
                episodes=[
                    7, 9, 25, 29, 30, 41, 63, 74, 82, 83, 96, 98, 124, 135, 148, 160, 161, 163, 171, 174, 188,
                    195, 196, 205, 208, 221, 222, 223, 237, 246, 250, 256, 265, 266, 275, 281, 286, 289, 293, 297,
                    318, 331, 373,
                ],
            ),
            extra_delta_transform=False,
        ),
        batch_size=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10,
            peak_lr=5e-5,
            decay_steps=100,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero/my_experiment/5000/params"
        ),
        num_train_steps=101,
        save_interval=100,
        log_interval=10,
        keep_period=100,
    ),
    TrainConfig(
        name="pi05_libero10_task5_all33_from_t4step100_retrain_100",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                prompt_from_task=True,
                episodes=[
                    8, 13, 26, 39, 69, 71, 77, 79, 92, 101, 102, 118, 132, 137, 156, 181, 199, 200, 219, 234,
                    238, 241, 260, 291, 308, 317, 334, 336, 340, 350, 352, 359, 363,
                ],
            ),
            extra_delta_transform=False,
        ),
        batch_size=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10,
            peak_lr=5e-5,
            decay_steps=100,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero10_task4_all43_retrain_100/task4_all43_from_ckpt5000_100steps/100/params"
        ),
        num_train_steps=101,
        save_interval=50,
        log_interval=10,
        keep_period=50,
    ),
    TrainConfig(
        name="pi05_libero_goal_task14_all47_from_t4step100_retrain_100",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                prompt_from_task=True,
                episodes=[
                    385, 389, 396, 397, 404, 417, 419, 423, 427, 430, 432, 436, 454, 468, 470, 475, 476, 479,
                    480, 482, 512, 515, 528, 533, 543, 546, 554, 555, 556, 574, 588, 591, 592, 596, 619, 635,
                    645, 648, 656, 668, 682, 686, 697, 735, 766, 788, 795,
                ],
            ),
            extra_delta_transform=False,
        ),
        batch_size=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10,
            peak_lr=5e-5,
            decay_steps=100,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero10_task4_all43_retrain_100/task4_all43_from_ckpt5000_100steps/100/params"
        ),
        num_train_steps=101,
        save_interval=50,
        log_interval=10,
        keep_period=50,
    ),
    TrainConfig(
        name="pi05_libero_obj_t22_t23_t24_all135_from_libero10_5000_retrain_50",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                prompt_from_task=True,
                episodes=[
                    810, 856, 862, 886, 892, 894, 900, 903, 913, 919, 925, 931, 935, 936, 959, 964, 982, 996,
                    997, 1015, 1016, 1021, 1036, 1055, 1057, 1058, 1061, 1069, 1072, 1079, 1092, 1103, 1126, 1153,
                    1162, 1184, 1191, 1194, 1198, 1202, 1213, 1221, 1241, 1246, 1256,
                    811, 812, 824, 843, 846, 849, 853, 858, 867, 871, 889, 893, 896, 932, 943, 950, 951, 954,
                    968, 973, 980, 988, 1005, 1008, 1022, 1025, 1034, 1067, 1077, 1089, 1125, 1129, 1130, 1135,
                    1136, 1144, 1181, 1193, 1199, 1226, 1230, 1231, 1235, 1236, 1251, 1258,
                    813, 818, 825, 850, 852, 857, 860, 876, 877, 887, 905, 908, 934, 945, 978, 983, 984, 985,
                    1002, 1011, 1020, 1024, 1035, 1074, 1075, 1100, 1102, 1108, 1131, 1143, 1148, 1151, 1152, 1166,
                    1171, 1187, 1205, 1219, 1224, 1225, 1233, 1242, 1243, 1245,
                ],
            ),
            extra_delta_transform=False,
        ),
        batch_size=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=5,
            peak_lr=5e-5,
            decay_steps=50,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero_10/my_experiment/5000/params"
        ),
        num_train_steps=51,
        save_interval=10,
        log_interval=10,
        keep_period=10,
    ),
    TrainConfig(
        name="pi05_libero_obj_task22_ep45_from_libero10_5000_retrain_15",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                prompt_from_task=True,
                episodes=[
                    810, 856, 862, 886, 892, 894, 900, 903, 913, 919, 925, 931, 935, 936, 959, 964, 982, 996,
                    997, 1015, 1016, 1021, 1036, 1055, 1057, 1058, 1061, 1069, 1072, 1079, 1092, 1103, 1126, 1153,
                    1162, 1184, 1191, 1194, 1198, 1202, 1213, 1221, 1241, 1246, 1256,
                ],
            ),
            extra_delta_transform=False,
        ),
        batch_size=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2,
            peak_lr=5e-5,
            decay_steps=15,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero_10/my_experiment/5000/params"
        ),
        num_train_steps=16,
        save_interval=15,
        log_interval=5,
        keep_period=15,
    ),
    TrainConfig(
        name="pi05_libero_obj_task23_ep46_from_libero10_5000_retrain_15",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                prompt_from_task=True,
                episodes=[
                    811, 812, 824, 843, 846, 849, 853, 858, 867, 871, 889, 893, 896, 932, 943, 950, 951, 954,
                    968, 973, 980, 988, 1005, 1008, 1022, 1025, 1034, 1067, 1077, 1089, 1125, 1129, 1130, 1135,
                    1136, 1144, 1181, 1193, 1199, 1226, 1230, 1231, 1235, 1236, 1251, 1258,
                ],
            ),
            extra_delta_transform=False,
        ),
        batch_size=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2,
            peak_lr=5e-5,
            decay_steps=15,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero_10/my_experiment/5000/params"
        ),
        num_train_steps=16,
        save_interval=15,
        log_interval=5,
        keep_period=15,
    ),
    TrainConfig(
        name="pi05_libero_obj_task23_ep46_from_merged_t22w02_t24w08_retrain_15",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                prompt_from_task=True,
                episodes=[
                    811, 812, 824, 843, 846, 849, 853, 858, 867, 871, 889, 893, 896, 932, 943, 950, 951, 954,
                    968, 973, 980, 988, 1005, 1008, 1022, 1025, 1034, 1067, 1077, 1089, 1125, 1129, 1130, 1135,
                    1136, 1144, 1181, 1193, 1199, 1226, 1230, 1231, 1235, 1236, 1251, 1258,
                ],
            ),
            extra_delta_transform=False,
        ),
        batch_size=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2,
            peak_lr=5e-5,
            decay_steps=15,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/merged/t22w02_t24w08_vl_only/0/params"
        ),
        num_train_steps=16,
        save_interval=15,
        log_interval=5,
        keep_period=15,
    ),
    TrainConfig(
        name="pi05_libero_obj_task23_ep46_from_libero10_5000_retrain_30",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                prompt_from_task=True,
                episodes=[
                    811, 812, 824, 843, 846, 849, 853, 858, 867, 871, 889, 893, 896, 932, 943, 950, 951, 954,
                    968, 973, 980, 988, 1005, 1008, 1022, 1025, 1034, 1067, 1077, 1089, 1125, 1129, 1130, 1135,
                    1136, 1144, 1181, 1193, 1199, 1226, 1230, 1231, 1235, 1236, 1251, 1258,
                ],
            ),
            extra_delta_transform=False,
        ),
        batch_size=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2,
            peak_lr=5e-5,
            decay_steps=30,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero_10/my_experiment/5000/params"
        ),
        num_train_steps=31,
        save_interval=30,
        log_interval=5,
        keep_period=30,
    ),
    TrainConfig(
        name="pi05_libero_obj_task23_ep46_from_merged_t22w02_t24w08_retrain_30",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                prompt_from_task=True,
                episodes=[
                    811, 812, 824, 843, 846, 849, 853, 858, 867, 871, 889, 893, 896, 932, 943, 950, 951, 954,
                    968, 973, 980, 988, 1005, 1008, 1022, 1025, 1034, 1067, 1077, 1089, 1125, 1129, 1130, 1135,
                    1136, 1144, 1181, 1193, 1199, 1226, 1230, 1231, 1235, 1236, 1251, 1258,
                ],
            ),
            extra_delta_transform=False,
        ),
        batch_size=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2,
            peak_lr=5e-5,
            decay_steps=30,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/merged/t22w02_t24w08_vl_only/0/params"
        ),
        num_train_steps=31,
        save_interval=30,
        log_interval=5,
        keep_period=30,
    ),
    TrainConfig(
        name="pi05_libero_obj_task23_ep46_from_libero10_5000_retrain_50",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                prompt_from_task=True,
                episodes=[
                    811, 812, 824, 843, 846, 849, 853, 858, 867, 871, 889, 893, 896, 932, 943, 950, 951, 954,
                    968, 973, 980, 988, 1005, 1008, 1022, 1025, 1034, 1067, 1077, 1089, 1125, 1129, 1130, 1135,
                    1136, 1144, 1181, 1193, 1199, 1226, 1230, 1231, 1235, 1236, 1251, 1258,
                ],
            ),
            extra_delta_transform=False,
        ),
        batch_size=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2,
            peak_lr=5e-5,
            decay_steps=50,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero_10/my_experiment/5000/params"
        ),
        num_train_steps=51,
        save_interval=50,
        log_interval=5,
        keep_period=50,
    ),
    TrainConfig(
        name="pi05_libero_obj_task23_ep46_from_merged_t22w02_t24w08_retrain_50",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                prompt_from_task=True,
                episodes=[
                    811, 812, 824, 843, 846, 849, 853, 858, 867, 871, 889, 893, 896, 932, 943, 950, 951, 954,
                    968, 973, 980, 988, 1005, 1008, 1022, 1025, 1034, 1067, 1077, 1089, 1125, 1129, 1130, 1135,
                    1136, 1144, 1181, 1193, 1199, 1226, 1230, 1231, 1235, 1236, 1251, 1258,
                ],
            ),
            extra_delta_transform=False,
        ),
        batch_size=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2,
            peak_lr=5e-5,
            decay_steps=50,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/merged/t22w02_t24w08_vl_only/0/params"
        ),
        num_train_steps=51,
        save_interval=50,
        log_interval=5,
        keep_period=50,
    ),
    TrainConfig(
        name="pi05_libero_obj_task23_ep46_from_mvl_bact_retrain_50",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                prompt_from_task=True,
                episodes=[
                    811, 812, 824, 843, 846, 849, 853, 858, 867, 871, 889, 893, 896, 932, 943, 950, 951, 954,
                    968, 973, 980, 988, 1005, 1008, 1022, 1025, 1034, 1067, 1077, 1089, 1125, 1129, 1130, 1135,
                    1136, 1144, 1181, 1193, 1199, 1226, 1230, 1231, 1235, 1236, 1251, 1258,
                ],
            ),
            extra_delta_transform=False,
        ),
        batch_size=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2,
            peak_lr=5e-5,
            decay_steps=50,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/merged/task23_ablation_init/mvl_bact/0/params"
        ),
        num_train_steps=51,
        save_interval=50,
        log_interval=5,
        keep_period=50,
    ),
    TrainConfig(
        name="pi05_libero_obj_task23_ep46_from_bvl_mact_retrain_50",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                prompt_from_task=True,
                episodes=[
                    811, 812, 824, 843, 846, 849, 853, 858, 867, 871, 889, 893, 896, 932, 943, 950, 951, 954,
                    968, 973, 980, 988, 1005, 1008, 1022, 1025, 1034, 1067, 1077, 1089, 1125, 1129, 1130, 1135,
                    1136, 1144, 1181, 1193, 1199, 1226, 1230, 1231, 1235, 1236, 1251, 1258,
                ],
            ),
            extra_delta_transform=False,
        ),
        batch_size=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2,
            peak_lr=5e-5,
            decay_steps=50,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/merged/task23_ablation_init/bvl_mact/0/params"
        ),
        num_train_steps=51,
        save_interval=50,
        log_interval=5,
        keep_period=50,
    ),
    TrainConfig(
        name="pi05_libero_obj_task24_ep44_from_libero10_5000_retrain_15",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                prompt_from_task=True,
                episodes=[
                    813, 818, 825, 850, 852, 857, 860, 876, 877, 887, 905, 908, 934, 945, 978, 983, 984, 985,
                    1002, 1011, 1020, 1024, 1035, 1074, 1075, 1100, 1102, 1108, 1131, 1143, 1148, 1151, 1152, 1166,
                    1171, 1187, 1205, 1219, 1224, 1225, 1233, 1242, 1243, 1245,
                ],
            ),
            extra_delta_transform=False,
        ),
        batch_size=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2,
            peak_lr=5e-5,
            decay_steps=15,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero_10/my_experiment/5000/params"
        ),
        num_train_steps=16,
        save_interval=15,
        log_interval=5,
        keep_period=15,
    ),
    # --- pi0.5 50/50 co-training: (SGO+LM90) vs (single libero_10 task) ---
    # Group A (50%): SGO + LM90
    # Group B (50%): task-specific libero_10 episodes
    # Since Group A is represented as two concatenated datasets (SGO, LM90),
    # we split its 50% proportionally by episode counts: 1314 and 3917.
    # Resulting dataset_mix_weights: [0.1256, 0.3744, 0.5]
    TrainConfig(
        name="pi05_sgo_lm90_mix50_task0",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            assets=AssetsConfig(asset_id="pi05_sgo_lm90_mix50_task0"),
            base_config=DataConfig(
                prompt_from_task=True,
                episodes=(
                    _suite_eps.LIBERO_GOAL_EPISODES
                    + _suite_eps.LIBERO_OBJECT_EPISODES
                    + _suite_eps.LIBERO_SPATIAL_EPISODES
                ),
                extra_lerobot_datasets=(
                    (
                        "lerobot_lm90",
                        "/storage/yukaichengLab/lishiwen/jiayusun/huggingface/lerobot_lm90",
                        None,
                    ),
                    (
                        "physical-intelligence/libero",
                        None,
                        [0, 18, 22, 33, 58, 85, 88, 105, 107],
                    ),
                ),
                dataset_mix_weights=(0.1256, 0.3744, 0.5),
            ),
            extra_delta_transform=False,
        ),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero_RETRAIN_base/pi05_libero_RETRAIN_base/10000/params"
        ),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi05_sgo_lm90_mix50_task6",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            assets=AssetsConfig(asset_id="pi05_sgo_lm90_mix50_task6"),
            base_config=DataConfig(
                prompt_from_task=True,
                episodes=(
                    _suite_eps.LIBERO_GOAL_EPISODES
                    + _suite_eps.LIBERO_OBJECT_EPISODES
                    + _suite_eps.LIBERO_SPATIAL_EPISODES
                ),
                extra_lerobot_datasets=(
                    (
                        "lerobot_lm90",
                        "/storage/yukaichengLab/lishiwen/jiayusun/huggingface/lerobot_lm90",
                        None,
                    ),
                    (
                        "physical-intelligence/libero",
                        None,
                        [10, 20, 23, 46, 51, 54, 67, 70, 73, 86, 100, 106],
                    ),
                ),
                dataset_mix_weights=(0.1256, 0.3744, 0.5),
            ),
            extra_delta_transform=False,
        ),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero_RETRAIN_base/pi05_libero_RETRAIN_base/10000/params"
        ),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi05_sgo_lm90_mix50_task9",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            assets=AssetsConfig(asset_id="pi05_sgo_lm90_mix50_task9"),
            base_config=DataConfig(
                prompt_from_task=True,
                episodes=(
                    _suite_eps.LIBERO_GOAL_EPISODES
                    + _suite_eps.LIBERO_OBJECT_EPISODES
                    + _suite_eps.LIBERO_SPATIAL_EPISODES
                ),
                extra_lerobot_datasets=(
                    (
                        "lerobot_lm90",
                        "/storage/yukaichengLab/lishiwen/jiayusun/huggingface/lerobot_lm90",
                        None,
                    ),
                    (
                        "physical-intelligence/libero",
                        None,
                        [27, 28, 47, 55, 61, 64, 81, 103, 104, 109],
                    ),
                ),
                dataset_mix_weights=(0.1256, 0.3744, 0.5),
            ),
            extra_delta_transform=False,
        ),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero_RETRAIN_base/pi05_libero_RETRAIN_base/10000/params"
        ),
        num_train_steps=30_000,
    ),
    # --- pi0.5 spatial+goal+object (3 suites combined) ---
    TrainConfig(
        name="pi05_libero_sgo",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                prompt_from_task=True,
                # spatial (1261-1692) + goal (379-806) + object (807-1260) = 1314 episodes
                episodes=(
                    _suite_eps.LIBERO_GOAL_EPISODES
                    + _suite_eps.LIBERO_OBJECT_EPISODES
                    + _suite_eps.LIBERO_SPATIAL_EPISODES
                ),
            ),
            extra_delta_transform=False,
        ),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/storage/yukaichengLab/lishiwen/jiayusun/openpi_pt/pi05_model/pi05_base/params"
        ),
        num_train_steps=30_000,
    ),
    # --- pi0.5 spatial+goal+object + libero_lm90 (all 90 tasks) ---
    # Primary: physical-intelligence/libero SGO (1314 eps)
    # Extra:   lerobot_lm90 (3917 eps, 73 tasks, locally at /storage/.../huggingface/lerobot_lm90)
    # Combined: ~5231 eps. Unified norm stats saved under asset_id "pi05_libero_sgo_lm90".
    TrainConfig(
        name="pi05_libero_sgo_lm90",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            assets=AssetsConfig(asset_id="pi05_libero_sgo_lm90"),
            base_config=DataConfig(
                prompt_from_task=True,
                episodes=(
                    _suite_eps.LIBERO_GOAL_EPISODES
                    + _suite_eps.LIBERO_OBJECT_EPISODES
                    + _suite_eps.LIBERO_SPATIAL_EPISODES
                ),
                extra_lerobot_datasets=(
                    (
                        "lerobot_lm90",
                        "/storage/yukaichengLab/lishiwen/jiayusun/huggingface/lerobot_lm90",
                        None,  # use all 3917 episodes
                    ),
                ),
            ),
            extra_delta_transform=False,
        ),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/storage/yukaichengLab/lishiwen/jiayusun/openpi_pt/pi05_model/pi05_base/params"
        ),
        num_train_steps=30_000,
    ),
    # --- pi0.5 libero_10 + libero_spatial fine-tuning from WUDI-MLLM 5k merged checkpoint ---
    # Dataset: libero_10 + libero_spatial only (2 suites)
    # Norm stats: pi05_libero/my_experiment/10000/assets
    # Init: wudi_mllm/10_spatial_from10k_iter5k (WUDI 5000-iter merge of libero_10 + libero_spatial)
    TrainConfig(
        name="pi05_libero_10_spatial_from_wudi_mllm_5k",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            assets=AssetsConfig(
                assets_dir="/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero/my_experiment/10000/assets",
            ),
            base_config=DataConfig(
                prompt_from_task=True,
                lerobot_root="/storage/yukaichengLab/lishiwen/jiayusun/libero",
                episodes=_suite_eps.LIBERO_10_EPISODES + _suite_eps.LIBERO_SPATIAL_EPISODES,
            ),
            extra_delta_transform=False,
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=5e-5, decay_steps=30_000, decay_lr=5e-6),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/wudi_mllm/10_spatial_from10k_iter5k/params"
        ),
        num_train_steps=30_000,
    ),
    # Init: wudi_mllm/4task_from10k_iter500 (WUDI 500-iter merge of 4 suites)
    # Dataset: all 4 suites (libero_10 + spatial + goal + object)
    # Steps: 10k
    TrainConfig(
        name="pi05_libero_4task_from_wudi_4task_500",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            assets=AssetsConfig(
                assets_dir="/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero/my_experiment/10000/assets",
            ),
            base_config=DataConfig(
                prompt_from_task=True,
                lerobot_root="/storage/yukaichengLab/lishiwen/jiayusun/libero",
                episodes=_suite_eps.LIBERO_10_EPISODES + _suite_eps.LIBERO_SPATIAL_EPISODES
                    + _suite_eps.LIBERO_GOAL_EPISODES + _suite_eps.LIBERO_OBJECT_EPISODES,
            ),
            extra_delta_transform=False,
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=500, peak_lr=5e-5, decay_steps=10_000, decay_lr=5e-6),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/wudi_mllm/4task_from10k_iter500/params"
        ),
        num_train_steps=10_000,
    ),
    # Init: wudi_mllm/4task_from10k_iter1k (WUDI 1000-iter merge of 4 suites)
    # Dataset: all 4 suites (libero_10 + spatial + goal + object)
    # Steps: 10k
    TrainConfig(
        name="pi05_libero_4task_from_wudi_4task_1k",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            assets=AssetsConfig(
                assets_dir="/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero/my_experiment/10000/assets",
            ),
            base_config=DataConfig(
                prompt_from_task=True,
                lerobot_root="/storage/yukaichengLab/lishiwen/jiayusun/libero",
                episodes=_suite_eps.LIBERO_10_EPISODES + _suite_eps.LIBERO_SPATIAL_EPISODES
                    + _suite_eps.LIBERO_GOAL_EPISODES + _suite_eps.LIBERO_OBJECT_EPISODES,
            ),
            extra_delta_transform=False,
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=500, peak_lr=5e-5, decay_steps=10_000, decay_lr=5e-6),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/wudi_mllm/4task_from10k_iter1k/params"
        ),
        num_train_steps=10_000,
    ),
    # Init: wudi_mllm/3task_sog_mean_iter500 (WUDI 500-iter mean merge of spatial+object+goal)
    # Dataset: all 4 suites (libero_10 + spatial + goal + object)
    # Steps: 5k, batch_size=32, 2-GPU FSDP
    TrainConfig(
        name="pi05_libero_4task_from_3task_sog_iter500",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            assets=AssetsConfig(
                assets_dir="/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero/my_experiment/10000/assets",
            ),
            base_config=DataConfig(
                prompt_from_task=True,
                lerobot_root="/storage/yukaichengLab/lishiwen/jiayusun/libero",
                episodes=_suite_eps.LIBERO_10_EPISODES + _suite_eps.LIBERO_SPATIAL_EPISODES
                    + _suite_eps.LIBERO_GOAL_EPISODES + _suite_eps.LIBERO_OBJECT_EPISODES,
            ),
            extra_delta_transform=False,
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=500, peak_lr=5e-5, decay_steps=5_000, decay_lr=5e-6),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/wudi_mllm/3task_sog_mean_iter500/params"
        ),
        num_train_steps=5_000,
    ),
    # Init: ft_from_3task_sog_iter500/4999 (5k-step 4-task FT)
    # Dataset: libero_10 only — fix "put both moka pots on the stove" weakness
    # Steps: 1k, batch_size=32, 2-GPU FSDP
    TrainConfig(
        name="pi05_libero_10_from_4task_3sog500_5k",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            assets=AssetsConfig(
                assets_dir="/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero/my_experiment/10000/assets",
            ),
            base_config=DataConfig(
                prompt_from_task=True,
                lerobot_root="/storage/yukaichengLab/lishiwen/jiayusun/libero",
                episodes=_suite_eps.LIBERO_10_EPISODES,
            ),
            extra_delta_transform=False,
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=100, peak_lr=2e-5, decay_steps=1_000, decay_lr=2e-6),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero_4task_from_3task_sog_iter500/ft_from_3task_sog_iter500/4999/params"
        ),
        num_train_steps=1_000,
    ),
    # --- pi0.5 all-task (1693 eps) fine-tuning from WUDI-MLLM 5k merged checkpoint ---
    # Dataset: /storage/yukaichengLab/lishiwen/jiayusun/huggingface/lerobot (40 tasks, 1693 eps)
    # Norm stats: pi05_libero/my_experiment/10000/assets
    # Init: wudi_mllm/10_spatial_from10k_iter5k (WUDI 5000-iter merge of libero_10 + libero_spatial)
    TrainConfig(
        name="pi05_libero_all_from_wudi_mllm_5k",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            assets=AssetsConfig(
                assets_dir="/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/pi05_libero/my_experiment/10000/assets",
            ),
            base_config=DataConfig(
                prompt_from_task=True,
                lerobot_root="/storage/yukaichengLab/lishiwen/jiayusun/libero",
            ),
            extra_delta_transform=False,
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=5e-5, decay_steps=30_000, decay_lr=5e-6),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/storage/yukaichengLab/lishiwen/jiayusun/openpi/checkpoints/wudi_mllm/10_spatial_from10k_iter5k/params"
        ),
        num_train_steps=30_000,
    ),
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instructions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),
    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        # This config is for fine-tuning pi0-FAST-base on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="/mnt/pi-data/kevin",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05-DROID on a custom (smaller) DROID dataset.
        # Here, we use LeRobot data format (like for all other fine-tuning examples)
        # To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
        name="pi05_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # Replace with your custom DROID LeRobot dataset repo id.
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    #
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    # RoboArena & PolaRiS configs.
    *roboarena_config.get_roboarena_configs(),
    *polaris_config.get_polaris_configs(),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
