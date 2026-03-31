"""
Dump validation batches from a dataset config to a pickle file.
Used as input for arithmetic.py / arithmetic_torch.py weight optimization.

Example:
    python scripts/dump_data.py --dataset pi05_libero --output libero_val.pkl
    python scripts/dump_data.py --dataset pi05_droid --output droid_val.pkl
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import pickle

from tqdm import tqdm

from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def main():
    parser = argparse.ArgumentParser(description="Dump validation batches for model arithmetic")
    parser.add_argument("--dataset", required=True, type=str, help="Config name (e.g. pi05_libero)")
    parser.add_argument("--output", required=True, help="Output pickle file path")
    parser.add_argument("--num_batches", type=int, default=50, help="Number of batches to dump")
    parser.add_argument("--batch_size", type=int, default=None, help="Override config batch size (use small value on CPU)")
    args = parser.parse_args()

    config = _config.get_config(args.dataset)
    if args.batch_size is not None:
        import dataclasses
        config = dataclasses.replace(config, batch_size=args.batch_size)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=None,
        shuffle=True,
    )
    samples_list = []
    for i, samples in enumerate(tqdm(data_loader, desc="Dumping batches")):
        if i >= args.num_batches:
            break
        samples_list.append(samples)

    with open(args.output, "wb") as f:
        pickle.dump(samples_list, f)

    print(f"✓ Dumped {len(samples_list)} batches to {args.output}")


if __name__ == "__main__":
    main()
