"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $HF_LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tyro

# Output root — same parent dir as the existing lerobot dataset
OUTPUT_ROOT = Path("/storage/yukaichengLab/lishiwen/jiayusun/huggingface")
REPO_NAME = "lerobot_lm90"  # saves to OUTPUT_ROOT / REPO_NAME
RAW_DATASET_NAMES = [
    "libero_lm_90",
]  # libero_lm_90: 90-task LIBERO dataset with language motions and segmentation


def resize_image(img_array: np.ndarray, size: tuple = (256, 256)) -> np.ndarray:
    """Resize a uint8 HxWxC image array to the target size."""
    img = Image.fromarray(img_array)
    img = img.resize((size[1], size[0]), Image.BILINEAR)
    return np.array(img)


def main(data_dir: str, *, push_to_hub: bool = False):
    # Clean up any existing dataset in the output directory
    output_path = OUTPUT_ROOT / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    # Schema matches the existing lerobot dataset at OUTPUT_ROOT/lerobot
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        root=OUTPUT_ROOT / REPO_NAME,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=4,
        image_writer_processes=0,
    )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # libero_lm_90 images are 224x224; resize to 256x256 for schema compatibility
    for raw_dataset_name in RAW_DATASET_NAMES:
        raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
        for episode in raw_dataset:
            for step in episode["steps"].as_numpy_iterator():
                dataset.add_frame(
                    {
                        "image": resize_image(step["observation"]["image"]),
                        "wrist_image": resize_image(step["observation"]["wrist_image"]),
                        "state": step["observation"]["state"],
                        "actions": step["action"],
                        "task": step["language_instruction"].decode(),
                    }
                )
            dataset.save_episode()

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )

    print(f"Dataset saved to: {output_path}")
    print(f"Total episodes: {dataset.num_episodes}")
    print(f"Total frames: {dataset.num_frames}")


if __name__ == "__main__":
    tyro.cli(main)
