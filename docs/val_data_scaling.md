# Scaling Validation Data for Weight Merging Optimization

## Background

Model arithmetic merges `N` fine-tuned checkpoints via weighted averaging:

```
θ_mixed = Σ wᵢ · θᵢ
```

Weights `wᵢ` are optimized by gradient descent on a held-out **validation set**, minimizing the loss of the mixed model. The quality of the learned weights directly depends on how representative the validation set is.

This document describes the validation data pipeline, issues encountered when scaling it up, and all fixes applied.

---

## Dataset

**Source:** `huggingface/lerobot` — Physical Intelligence LIBERO dataset (LeRobot v2.0 format)

| Stat | Value |
|------|-------|
| Total episodes | 1,693 |
| Total frames | 273,465 |
| FPS | 10 Hz |
| Tasks | 40 (10 per suite) |
| Image resolution | 256 × 256 × 3 (RGB + wrist) |

**Suite breakdown:**

| Suite | Episodes | Frames | Avg episode length |
|-------|---------|--------|--------------------|
| libero_spatial | 432 | 52,970 | 122 frames (~12s) |
| libero_goal | 428 | 52,042 | 121 frames (~12s) |
| libero_object | 454 | 66,984 | 147 frames (~15s) |
| libero_10 | 379 | 101,469 | 267 frames (~27s) |

Total usable training samples (each frame = one sample with `action_horizon=10` lookahead): **256,535**.

---

## How Validation Data is Dumped

The dump is **purely offline** — no simulator involved. It reads from static parquet files via the LeRobot `DataLoader`, applies the same transforms used during training, and serializes batches to a pickle file.

```python
# scripts/dump_data.py (simplified)
loader = create_data_loader(config, sharding=None, shuffle=True)
batches = [next(loader) for _ in range(NUM_BATCHES)]
pickle.dump(batches, open(output, "wb"))
```

Each batch item: `(obs_dict, actions)`
- `obs["image"]`: `(B, 256, 256, 3)` float32
- `obs["wrist_image"]`: `(B, 256, 256, 3)` float32
- `obs["state"]`: `(B, 8)` float32
- `actions`: `(B, 10, 7)` float32 (action_horizon=10, action_dim=7)
- `obs["tokenized_prompt"]`: `(B, 180)` int32

**Memory note:** Each sample is ~1.5 MB uncompressed float32, vs ~27 MB/episode in compressed parquet — ~18× inflation due to PNG → float32. A 57 GB pkl corresponds to ~3.7 GB of source parquet.

---

## How the GD Optimizer Uses Validation Data

Understanding this is critical for sizing the validation set correctly.

### GD loop (one batch per iteration)

```python
# arithmetic.py — optimize_weights_with_gradient_descent
for iteration in range(num_iterations):
    batch = data_samples_list[iteration % len(data_samples_list)]  # ONE batch
    loss, grads = jax.value_and_grad(compute_loss)(mixed_params_gpu, policy, batch)
    ...
```

Each iteration uses **one batch** selected via cyclic modulo. With 50 iterations and N batches, the optimizer cycles through `min(50, N)` unique batches.

### compute_checkpoint_losses (all batches)

```python
# Called for inverse_loss method and for initial loss reporting
for ckpt in checkpoints:
    for batch in data_samples_list:   # ALL batches
        loss = policy.compute_loss(..., batch)
```

This function iterates over **every batch** in `data_samples_list` sequentially for each checkpoint. With large pkls this becomes the GPU memory bottleneck (see Fix 3 below).

---

## Problem 1: Undersized Validation Set

### Reference implementation (kai0)

| Parameter | Value |
|-----------|-------|
| `num_batches` | 50 |
| `batch_size` | **256** (config default for pi05_libero) |
| Total samples | **12,800** |
| Dataset coverage | **4.99%** |

### Our initial port

When porting, `--batch_size 32` was passed explicitly to fit CPU memory on the login node, but `num_batches` was kept at 13 per suite (52 total):

| Parameter | Value |
|-----------|-------|
| `num_batches` | 52 (13 × 4 suites) |
| `batch_size` | **32** (manually overridden) |
| Total samples | **1,664** |
| Dataset coverage | **0.65%** |

This is **7.7× fewer samples** than kai0. The optimizer sees far less signal per iteration.

---

## Problem 2: Biased Distribution

The initial dump used the `pi05_libero` mixed config (all 4 suites combined). Due to random shuffling, the front batches happened to oversample `libero_10` (36.7% vs expected 25%). This caused GD to assign disproportionate weight to `libero_10`:

```
Biased weights:   [spatial=0.10, object=0.11, goal=0.23, libero_10=0.56]
Balanced weights: [spatial=0.19, object=0.26, goal=0.25, libero_10=0.30]
```

---

## Fix 1: Balanced Distribution

Replaced the single mixed-config dump with **4 parallel iterators + direct interleave**, guaranteeing exactly 25% per suite:

```python
# Old (biased): single mixed dataloader
loader = create_data_loader(get_config("pi05_libero"), shuffle=True)

# New (balanced): 4 suite loaders interleaved
loaders = [iter(create_data_loader(get_config(s), shuffle=True))
           for s in ['pi05_libero_10', 'pi05_libero_goal',
                     'pi05_libero_object', 'pi05_libero_spatial']]
for i in range(NUM_BATCHES):
    for loader in loaders:
        batches.append(next(loader))
# Result: [l10, goal, obj, spatial, l10, goal, obj, spatial, ...]
```

This also eliminates the need for 4 intermediate pkl files (previously dumped separately then merged) — peak disk usage equals only the output file size.

**Output:** `pkl/libero_val_balanced.pkl` — 52 batches, 1,664 samples

---

## Fix 2: Scaled-Up Dump (20×)

Increased `num_batches` from 13 to **260 per suite** (20×):

| Parameter | Before (v2) | After (v3) |
|-----------|-------------|-----------|
| Batches per suite | 13 | **260** |
| Total batches | 52 | **1,040** |
| `batch_size` | 32 | 32 |
| Total samples | 1,664 | **33,280** |
| Dataset coverage | 0.65% | **13.0%** |
| File size | 2.9 GB | **57 GB** |

**Output:** `pkl/libero_val_balanced_large.pkl` (Job 32950, gnho031, completed)

---

## Fix 3: GPU OOM During GD with Large pkl

After switching GD to the large validation pkl (`1,040` batches), two merge jobs failed with GPU OOM:

- Job `32952` (gnho034): `FAILED`, `RESOURCE_EXHAUSTED`
- Job `32954` (gnho031): `FAILED`, `RESOURCE_EXHAUSTED`

Representative error:

```text
jaxlib.xla_extension.XlaRuntimeError: RESOURCE_EXHAUSTED:
Failed to allocate request for 576.00MiB on device ordinal 0
```

### Root cause

Both failed jobs ran on gnho034 **while a competing WUDI merge job was already occupying GPUs on the same node**. `arithmetic.py` overrides `CUDA_VISIBLE_DEVICES` via `--gpu_ids`, ignoring the SLURM-allocated device set, so it conflicted with the concurrent job on the same physical GPUs. The OOM was not caused by the pkl size itself, but by reduced available VRAM on a contested node.

### Fix: Request sufficient resources, avoid node contention

The correct approach is to request enough SLURM resources so the job runs on an idle node with full VRAM available:

```bash
#SBATCH --gres=gpu:4       # 4 × H800 (80 GB each) = 320 GB VRAM
#SBATCH --mem=400G         # system RAM: 57 GB pkl + ~48 GB checkpoints + overhead
```

No code changes to `arithmetic.py` are needed. The GD loop and `compute_checkpoint_losses` both run on `jax.devices("gpu")[0]` (single GPU), which has 80 GB VRAM — more than sufficient for the pi0.5 model (~12 GB params) plus batch buffers.

**Memory breakdown:**

| Component | Size |
|-----------|------|
| `libero_val_balanced_large.pkl` loaded into RAM | ~57 GB |
| 4 checkpoints on CPU RAM | ~48 GB (×12 GB each) |
| Mixed params on GPU 0 | ~12 GB |
| Gradients + JAX buffers on GPU 0 | ~10 GB |
| **Total RAM needed** | **~110 GB** |
| **Total VRAM needed (GPU 0)** | **~22 GB** |

### Current status

- Job `32963` (gnho009): running with `gpu:4`, `--mem=400G`, full 1,040-batch pkl, no truncation

---

## Summary

| Version | Samples used by GD | Distribution | Notes |
|---------|--------------------|-------------|-------|
| v1 (small mixed) | 1,664 | biased | deprecated |
| v2 (small balanced) | 1,664 | balanced (25/25/25/25) | stable but small |
| v3 (large balanced) | 33,280 | balanced (25/25/25/25) | dump complete, job 32950 |
| **v3 on dedicated node** | **33,280** | **balanced (25/25/25/25)** | **job 32963, gpu:4, mem=400G** |
