"""
Compute average loss of a checkpoint on a dataset config.

Example:
    python scripts/compute_loss_on_dataset.py \
        --config pi05_libero \
        --checkpoint checkpoints/pi05_libero_4task_merge_gd_frozen_large/0 \
        --num_batches 200 \
        --batch_size 32
"""

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import argparse
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.policies import policy_config as _policy_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Config name, e.g. pi05_libero")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint directory")
    parser.add_argument("--num_batches", type=int, default=200, help="Number of batches to evaluate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (override config)")
    args = parser.parse_args()

    config = _config.get_config(args.config)
    if args.batch_size != config.batch_size:
        config = dataclasses.replace(config, batch_size=args.batch_size)
        print(f"Overriding batch_size: {config.batch_size} -> {args.batch_size}")

    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Num batches: {args.num_batches}, batch_size: {args.batch_size}")
    print(f"Total samples: {args.num_batches * args.batch_size}")

    # Load policy/model
    print("\nLoading checkpoint...")
    policy = _policy_config.create_trained_policy(config, args.checkpoint, norm_stats=None)
    model = policy._model
    print("Checkpoint loaded.")

    # Create data loader
    data_loader = _data_loader.create_data_loader(config, sharding=None, shuffle=False)

    # Compute loss
    losses = []
    print(f"\nComputing loss over {args.num_batches} batches...")
    for i, samples in enumerate(tqdm(data_loader, total=args.num_batches, desc="Evaluating")):
        if i >= args.num_batches:
            break
        loss = model.compute_loss(jax.random.key(0), samples[0], samples[1])
        losses.append(float(jnp.mean(loss)))

    losses = np.array(losses)
    print("\n" + "=" * 50)
    print(f"Results over {len(losses)} batches ({len(losses) * args.batch_size} samples):")
    print(f"  Mean loss:   {losses.mean():.6f}")
    print(f"  Std loss:    {losses.std():.6f}")
    print(f"  Min loss:    {losses.min():.6f}")
    print(f"  Max loss:    {losses.max():.6f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
