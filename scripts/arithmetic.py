"""
Mix multiple OpenPI JAX checkpoints (Orbax/OCDBT) with weighted parameter averaging.

For PyTorch checkpoints (model.safetensors), use arithmetic_torch.py instead.

Example:
    python scripts/arithmetic.py \\
        --config pi05_libero \\
        --data-path libero_val.pkl \\
        --checkpoints /path/to/ckpt1/90000 /path/to/ckpt2/90000 \\
        --output /path/to/mixed \\
        --optimize_method inverse_loss \\
        --gpu_ids 0
"""

import argparse
import gc
import logging
import os
import time
from functools import partial
from pathlib import Path
import pickle

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from flax import nnx
import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from tqdm import tqdm

from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
import openpi.shared.normalize as _normalize
from openpi.training import config as _config

from model_arithmetic_common import (
    compute_optimal_weights,
    load_norm_stats,
    mix_norm_stats,
    mix_params,
    save_norm_stats,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("jax").setLevel(logging.ERROR)
logging.getLogger("xla").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ==============================
# Checkpoint read/write helpers
# ==============================

def resolve_ckpt_path(path: str) -> str:
    """Resolve input path to the actual JAX params directory."""
    p = Path(path).resolve()
    if (p / "_METADATA").exists():
        return str(p)
    elif (p / "_CHECKPOINT_METADATA").exists() and (p / "params" / "_METADATA").exists():
        return str(p / "params")
    elif (p.name == "params") and (p.parent / "_CHECKPOINT_METADATA").exists():
        return str(p)
    else:
        raise FileNotFoundError(f"Invalid JAX checkpoint path: {p}")


def load_jax_params(checkpoint_path: str):
    """Load JAX checkpoint params as a flat dict[str, np.ndarray]."""
    resolved = resolve_ckpt_path(checkpoint_path)
    params = _model.restore_params(resolved, restore_type=np.ndarray)
    return flax.traverse_util.flatten_dict(params, sep="/")


def save_jax_params(flat_params, output_dir):
    """Save flat params dict as Orbax/OCDBT checkpoint (step=0)."""
    nested = flax.traverse_util.unflatten_dict(flat_params, sep="/")
    output_dir = str(Path(output_dir).resolve())  # Orbax requires absolute path
    os.makedirs(output_dir, exist_ok=True)
    mngr = ocp.CheckpointManager(
        output_dir,
        item_handlers={"params": ocp.PyTreeCheckpointHandler(use_ocdbt=True)},
        options=ocp.CheckpointManagerOptions(max_to_keep=None, create=True),
    )
    mngr.save(0, {"params": {"params": nested}})
    mngr.wait_until_finished()
    print(f"✓ Saved JAX checkpoint to {output_dir}/0/")


# ==============================
# Weight optimization strategies
# ==============================

def find_norm_stats_path(ckpt_path: str) -> str | None:
    """Find norm_stats.json within a checkpoint directory (searches assets/ recursively)."""
    import glob as _glob
    # Standard openpi layout: {step}/assets/{asset_id}/norm_stats.json
    candidates = _glob.glob(os.path.join(ckpt_path, "assets", "**", "norm_stats.json"), recursive=True)
    if candidates:
        return candidates[0]
    # Fallback: norm_stats.json directly in ckpt_path or its parent
    for p in [ckpt_path, os.path.dirname(ckpt_path)]:
        candidate = os.path.join(p, "norm_stats.json")
        if os.path.exists(candidate):
            return candidate
    return None


def compute_checkpoint_losses(checkpoints, config, data_samples_list):
    """Compute mean loss per checkpoint on validation batches (for inverse_loss weights)."""
    losses = []
    for ckpt_path in checkpoints:
        # norm_stats=None: create_trained_policy auto-loads from checkpoint's assets/ dir
        policy = _policy_config.create_trained_policy(config, ckpt_path, norm_stats=None)
        ckpt_losses = []
        for data_samples in tqdm(data_samples_list, desc="Computing checkpoint losses"):
            loss = policy._model.compute_loss(jax.random.key(0), data_samples[0], data_samples[1])
            ckpt_losses.append(float(jnp.mean(loss)))
        print(f"Checkpoint losses for {ckpt_path}: {ckpt_losses}")
        losses.append(float(np.mean(ckpt_losses)))
        del policy
    print(f"Computed losses: {losses}")
    return losses


def optimize_weights_with_gradient_descent(
    checkpoints, config, data_samples_list,
    num_iterations=50, learning_rate=0.1, print_every=1
):
    """Optimize mixing weights via gradient descent on the probability simplex."""
    print("\n" + "=" * 60)
    print("Optimizing weights with gradient descent...")
    print("=" * 60)

    policy = _policy_config.create_trained_policy(config, checkpoints[0], norm_stats=None)

    params_list_cpu = []
    for ckpt_path in tqdm(checkpoints, desc="Loading checkpoints"):
        resolved = resolve_ckpt_path(ckpt_path)
        params = _model.restore_params(resolved, restore_type=np.ndarray)
        params_list_cpu.append(flax.traverse_util.flatten_dict(params, sep="/"))

    cpu_device = jax.devices("cpu")[0]
    params_list_jax_cpu = jax.device_put(params_list_cpu, cpu_device)

    n_checkpoints = len(checkpoints)
    log_weights = jnp.zeros(n_checkpoints)
    schedule = optax.cosine_decay_schedule(init_value=learning_rate, decay_steps=num_iterations, alpha=0.01)
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(log_weights)

    @partial(jax.jit, static_argnames=["policy"])
    def compute_loss_wrt_params(flat_params, policy, data_samples):
        model = policy._model
        nested_params = flax.traverse_util.unflatten_dict(flat_params, sep="/")
        nnx.update(model, nnx.State(nested_params))
        loss = model.compute_loss(jax.random.key(0), data_samples[0], data_samples[1])
        return jnp.mean(loss)

    @partial(jax.jit, backend="cpu")
    def mix_params_cpu(params_list, weights):
        def weighted_sum(*args):
            res = jnp.zeros_like(args[0])
            for p, w in zip(args, weights):
                res += p * w
            return res
        return jax.tree.map(weighted_sum, *params_list)

    @partial(jax.jit, backend="cpu")
    def project_grads_cpu(grads, params_list):
        dots = []
        for p_k in params_list:
            term_dots = jax.tree.map(lambda g, p: jnp.sum(g * p), grads, p_k)
            dots.append(jax.tree_util.tree_reduce(jnp.add, term_dots))
        return jnp.array(dots)

    best_loss = float("inf")
    best_weights = None
    gpu_device = jax.devices("gpu")[0]

    for iteration in range(num_iterations):
        current_weights = jax.nn.softmax(log_weights)
        mixed_params_cpu = mix_params_cpu(params_list_jax_cpu, current_weights)
        mixed_params_gpu = jax.device_put(mixed_params_cpu, gpu_device)

        loss_value, param_grads_gpu = jax.value_and_grad(compute_loss_wrt_params)(
            mixed_params_gpu, policy, data_samples_list[iteration % len(data_samples_list)]
        )
        param_grads_cpu = jax.device_put(param_grads_gpu, cpu_device)
        g_k = project_grads_cpu(param_grads_cpu, params_list_jax_cpu)
        g_k_np = np.array(g_k)
        weights_np = np.array(current_weights)
        g_bar = np.sum(g_k_np * weights_np)
        grad_log_weights = weights_np * (g_k_np - g_bar)

        updates, opt_state = optimizer.update(jnp.array(grad_log_weights), opt_state)
        log_weights = optax.apply_updates(log_weights, updates)
        loss_val_float = float(loss_value)

        if loss_val_float < best_loss:
            best_loss = loss_val_float
            best_weights = weights_np.copy()
        if (iteration + 1) % print_every == 0 or iteration == 0:
            print(f"Iter {iteration + 1}/{num_iterations}: loss={loss_val_float:.6f}, weights={weights_np}")
        if _WANDB_AVAILABLE and _wandb.run is not None:
            log_dict = {"gd/loss": loss_val_float, "gd/iteration": iteration + 1}
            for i, w in enumerate(weights_np):
                log_dict[f"gd/weight_{i}"] = float(w)
            _wandb.log(log_dict, step=iteration + 1)
        del mixed_params_cpu, mixed_params_gpu, param_grads_gpu, param_grads_cpu, g_k, current_weights, updates

    print(f"\nBest loss: {best_loss:.6f}, Best weights: {best_weights}")
    if _WANDB_AVAILABLE and _wandb.run is not None:
        _wandb.log({"gd/best_loss": best_loss})
    result = [float(w) for w in (best_weights if best_weights is not None else jax.nn.softmax(log_weights))]
    del params_list_cpu, params_list_jax_cpu, policy, optimizer, opt_state, log_weights
    jax.clear_caches()
    gc.collect()
    return result


def optimize_weights_with_adaptive_gradient_descent(
    checkpoints, config, data_samples_list,
    num_iterations=50, learning_rate=0.1, print_every=1
):
    """Optimize mixing weights with loss-adaptive gradient scaling."""
    print("\n" + "=" * 60)
    print("Optimizing weights with adaptive gradient descent...")
    print("=" * 60)

    policy = _policy_config.create_trained_policy(config, checkpoints[0], norm_stats=None)

    params_list_cpu = []
    for ckpt_path in tqdm(checkpoints, desc="Loading checkpoints"):
        resolved = resolve_ckpt_path(ckpt_path)
        params = _model.restore_params(resolved, restore_type=np.ndarray)
        params_list_cpu.append(flax.traverse_util.flatten_dict(params, sep="/"))

    cpu_device = jax.devices("cpu")[0]
    params_list_jax_cpu = jax.device_put(params_list_cpu, cpu_device)
    n_checkpoints = len(checkpoints)
    log_weights = jnp.zeros(n_checkpoints)
    schedule = optax.cosine_decay_schedule(init_value=learning_rate, decay_steps=num_iterations, alpha=0.01)
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(log_weights)

    @partial(jax.jit, static_argnames=["policy"])
    def compute_loss_wrt_params(flat_params, policy, data_samples):
        model = policy._model
        nested_params = flax.traverse_util.unflatten_dict(flat_params, sep="/")
        nnx.update(model, nnx.State(nested_params))
        loss = model.compute_loss(jax.random.key(0), data_samples[0], data_samples[1])
        return jnp.mean(loss)

    @partial(jax.jit, backend="cpu")
    def mix_params_cpu(params_list, weights):
        def weighted_sum(*args):
            res = jnp.zeros_like(args[0])
            for p, w in zip(args, weights):
                res += p * w
            return res
        return jax.tree.map(weighted_sum, *params_list)

    @partial(jax.jit, backend="cpu")
    def project_grads_cpu(grads, params_list):
        dots = []
        for p_k in params_list:
            term_dots = jax.tree.map(lambda g, p: jnp.sum(g * p), grads, p_k)
            dots.append(jax.tree_util.tree_reduce(jnp.add, term_dots))
        return jnp.array(dots)

    @partial(jax.jit, backend="cpu")
    def compute_weight_gradient(g_k, weights):
        g_bar = jnp.sum(g_k * weights)
        return weights * (g_k - g_bar)

    @partial(jax.jit, backend="cpu")
    def optimizer_step(log_weights, opt_state, grad_log_weights, loss_val):
        # Adaptive scale: larger loss → more aggressive step
        scale = (loss_val / 0.05) ** 2
        scaled_grads = grad_log_weights * scale
        updates, new_opt_state = optimizer.update(scaled_grads, opt_state)
        new_log_weights = optax.apply_updates(log_weights, updates)
        return new_log_weights, new_opt_state

    best_loss = float("inf")
    best_weights = None
    gpu_device = jax.devices("gpu")[0]

    for iteration in range(num_iterations):
        current_weights = jax.nn.softmax(log_weights)
        mixed_params_cpu = mix_params_cpu(params_list_jax_cpu, current_weights)
        mixed_params_gpu = jax.device_put(mixed_params_cpu, gpu_device)
        loss_value, param_grads_gpu = jax.value_and_grad(compute_loss_wrt_params)(
            mixed_params_gpu, policy, data_samples_list[iteration % len(data_samples_list)]
        )

        param_grads_cpu = jax.device_put(param_grads_gpu, cpu_device)
        g_k = project_grads_cpu(param_grads_cpu, params_list_jax_cpu)
        grad_log_weights = compute_weight_gradient(g_k, current_weights)
        loss_val_float = float(loss_value)
        log_weights, opt_state = optimizer_step(log_weights, opt_state, grad_log_weights, loss_val_float)
        weights_np = np.array(current_weights)

        if loss_val_float < best_loss:
            best_loss = loss_val_float
            best_weights = weights_np.copy()
        if (iteration + 1) % print_every == 0 or iteration == 0:
            print(f"Iter {iteration + 1}/{num_iterations}: loss={loss_val_float:.6f}, weights={weights_np}")
        if _WANDB_AVAILABLE and _wandb.run is not None:
            log_dict = {"agd/loss": loss_val_float, "agd/iteration": iteration + 1}
            for i, w in enumerate(weights_np):
                log_dict[f"agd/weight_{i}"] = float(w)
            _wandb.log(log_dict, step=iteration + 1)
        del mixed_params_cpu, mixed_params_gpu, param_grads_gpu, param_grads_cpu, g_k, current_weights

    print(f"\nBest loss: {best_loss:.6f}, Best weights: {best_weights}")
    if _WANDB_AVAILABLE and _wandb.run is not None:
        _wandb.log({"agd/best_loss": best_loss})
    result = [float(w) for w in (best_weights if best_weights is not None else jax.nn.softmax(log_weights))]
    del params_list_cpu, params_list_jax_cpu, policy, optimizer, opt_state, log_weights
    jax.clear_caches()
    gc.collect()
    return result


def optimize_weights_greedy(checkpoints, config, data_samples_list):
    """Greedy strategy: pick best single checkpoint, then iteratively add improving ones."""
    print("\n" + "=" * 60)
    print("Optimizing weights with greedy strategy...")
    print("=" * 60)

    policy = _policy_config.create_trained_policy(config, checkpoints[0], norm_stats=None)

    params_list_cpu = []
    for ckpt_path in tqdm(checkpoints, desc="Loading checkpoints"):
        resolved = resolve_ckpt_path(ckpt_path)
        params = _model.restore_params(resolved, restore_type=np.ndarray)
        params_list_cpu.append(flax.traverse_util.flatten_dict(params, sep="/"))

    cpu_device = jax.devices("cpu")[0]
    gpu_device = jax.devices("gpu")[0]
    params_list_jax_cpu = jax.device_put(params_list_cpu, cpu_device)

    @partial(jax.jit, static_argnames=["policy"])
    def compute_loss_wrt_params(flat_params, policy, data_samples):
        model = policy._model
        nested_params = flax.traverse_util.unflatten_dict(flat_params, sep="/")
        nnx.update(model, nnx.State(nested_params))
        loss = model.compute_loss(jax.random.key(0), data_samples[0], data_samples[1])
        return jnp.mean(loss)

    @partial(jax.jit, backend="cpu")
    def mix_params_cpu(params_list, weights):
        def weighted_sum(*args):
            res = jnp.zeros_like(args[0])
            for p, w in zip(args, weights):
                res += p * w
            return res
        return jax.tree.map(weighted_sum, *params_list)

    def evaluate_combination(indices):
        n_selected = len(indices)
        weights = np.zeros(len(checkpoints))
        weights[indices] = 1.0 / n_selected
        weights_jax = jnp.array(weights)
        mixed_params_cpu = mix_params_cpu(params_list_jax_cpu, weights_jax)
        mixed_params_gpu = jax.device_put(mixed_params_cpu, gpu_device)
        total_loss = 0.0
        for batch_data in data_samples_list:
            loss = compute_loss_wrt_params(mixed_params_gpu, policy, batch_data)
            total_loss += float(loss)
        del mixed_params_gpu
        return total_loss / len(data_samples_list)

    n_checkpoints = len(checkpoints)
    remaining_indices = list(range(n_checkpoints))
    selected_indices = []
    best_loss = float("inf")

    print("\nEvaluating individual checkpoints...")
    for i in remaining_indices:
        loss = evaluate_combination([i])
        print(f"  Checkpoint {i+1}: loss={loss:.6f}")
        if loss < best_loss:
            best_loss = loss
            selected_indices = [i]
    remaining_indices.remove(selected_indices[0])
    print(f"-> Selected best start: Checkpoint {selected_indices[0]+1} (loss={best_loss:.6f})")

    while remaining_indices:
        print(f"\nSearching for best addition to {[i+1 for i in selected_indices]}...")
        iteration_best_loss = best_loss
        best_candidate = -1
        for i in remaining_indices:
            loss = evaluate_combination(selected_indices + [i])
            print(f"  + Checkpoint {i+1}: loss={loss:.6f}")
            if loss < iteration_best_loss:
                iteration_best_loss = loss
                best_candidate = i
        if best_candidate != -1:
            best_loss = iteration_best_loss
            selected_indices.append(best_candidate)
            remaining_indices.remove(best_candidate)
            print(f"-> Improvement found! Added Checkpoint {best_candidate+1}. New loss: {best_loss:.6f}")
            jax.clear_caches()
            gc.collect()
        else:
            print("-> No improvement found. Stopping.")
            break

    final_weights = np.zeros(n_checkpoints)
    final_weights[selected_indices] = 1.0 / len(selected_indices)
    print(f"\nFinal greedy weights: {final_weights}")
    del params_list_cpu, params_list_jax_cpu, policy
    gc.collect()
    return final_weights.tolist()


# ==============================
# Validation helper
# ==============================

def test_mixed_checkpoint_jax(config, checkpoint_path, data_samples_list):
    """Load mixed JAX checkpoint and compute average loss on validation batches."""
    # norm_stats.json saved directly in checkpoint_path by save_norm_stats
    norm_stats = _normalize.load(checkpoint_path)
    ckpt_dir = os.path.join(checkpoint_path, "0")
    policy = _policy_config.create_trained_policy(config, ckpt_dir, norm_stats=norm_stats)
    avg_loss = 0.0
    for data_samples in data_samples_list:
        loss = policy._model.compute_loss(jax.random.key(0), data_samples[0], data_samples[1])
        avg_loss += float(jnp.mean(loss))
    avg_loss /= len(data_samples_list)
    del policy
    return avg_loss


# ==============================
# Main
# ==============================

def main():
    parser = argparse.ArgumentParser(
        description="Mix OpenPI JAX checkpoints (Orbax) with weighted averaging. "
                    "Use arithmetic_torch.py for PyTorch checkpoints."
    )
    parser.add_argument("--config", required=True, help="Config name (e.g. pi05_libero)")
    parser.add_argument("--data-path", required=True, help="Validation data pickle (from dump_data.py)")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Checkpoint directories to mix")
    parser.add_argument("--weights", nargs="+", type=float, help="Manual mixing weights (optional)")
    parser.add_argument("--output", required=True, help="Output directory for mixed checkpoint")
    parser.add_argument(
        "--optimize_method", type=str, default="gradient_descent",
        choices=["average", "inverse_loss", "gradient_descent", "adaptive_gradient_descent", "greedy"],
    )
    parser.add_argument("--num_iterations", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--memory_fraction", type=float, default=0.8)
    parser.add_argument("--gpu_ids", type=str, default="0", help="Comma-separated GPU IDs")
    parser.add_argument("--wandb_project", type=str, default=None, help="W&B project name (enables logging)")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--wandb_dir", type=str, default=None, help="Local dir to save W&B offline logs")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.memory_fraction)
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"

    # W&B: offline mode so compute nodes don't need internet
    if _WANDB_AVAILABLE and args.wandb_project:
        os.environ["WANDB_MODE"] = "offline"
        if args.wandb_dir:
            os.environ["WANDB_DIR"] = args.wandb_dir
        _wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"arithmetic_{args.optimize_method}",
            config={
                "config": args.config,
                "optimize_method": args.optimize_method,
                "num_iterations": args.num_iterations,
                "learning_rate": args.learning_rate,
                "checkpoints": args.checkpoints,
            },
        )
        logger.info("wandb initialized in offline mode.")

    config = _config.get_config(args.config)
    with open(args.data_path, "rb") as f:
        data_samples_list = pickle.load(f)

    losses = []
    if args.weights is None:
        if args.optimize_method == "average":
            n = len(args.checkpoints)
            args.weights = [1.0 / n] * n
            print(f"\n✓ Average weights (1/{n} each): {args.weights}")
        elif args.optimize_method == "gradient_descent":
            args.weights = optimize_weights_with_gradient_descent(
                args.checkpoints, config, data_samples_list,
                num_iterations=args.num_iterations, learning_rate=args.learning_rate
            )
        elif args.optimize_method == "adaptive_gradient_descent":
            args.weights = optimize_weights_with_adaptive_gradient_descent(
                args.checkpoints, config, data_samples_list,
                num_iterations=args.num_iterations, learning_rate=args.learning_rate
            )
        elif args.optimize_method == "inverse_loss":
            losses = compute_checkpoint_losses(args.checkpoints, config, data_samples_list)
            args.weights = compute_optimal_weights(losses)
        elif args.optimize_method == "greedy":
            args.weights = optimize_weights_greedy(args.checkpoints, config, data_samples_list)
        else:
            raise ValueError(f"Invalid optimization method: {args.optimize_method}")
        print(f"\n✓ Optimized weights: {args.weights}")
    else:
        print(f"\nUsing provided weights: {args.weights}")
        losses = compute_checkpoint_losses(args.checkpoints, config, data_samples_list)

    if len(args.weights) != len(args.checkpoints):
        raise ValueError("Number of weights must match number of checkpoints")

    print("\n" + "=" * 60)
    print("Results:")
    if losses:
        for i, (ckpt, loss) in enumerate(zip(args.checkpoints, losses)):
            print(f"  Ckpt {i+1}: {loss:.6f} (w={args.weights[i]:.4f})")
    print("=" * 60)

    print("\nMixing parameters...")
    params_list = [load_jax_params(p) for p in args.checkpoints]
    mixed = mix_params(params_list, args.weights)
    del params_list
    gc.collect()
    save_jax_params(mixed, args.output)
    del mixed
    gc.collect()

    print("\nMixing norm_stats...")
    norm_stats_paths = []
    for ckpt_path in args.checkpoints:
        # resolve to step dir (strip /params suffix if present)
        step_dir = os.path.dirname(ckpt_path) if ckpt_path.endswith("/params") else ckpt_path
        ns_path = find_norm_stats_path(step_dir)
        if ns_path:
            norm_stats_paths.append(ns_path)
        else:
            logger.warning(f"norm_stats.json not found for {ckpt_path}")

    if len(norm_stats_paths) == len(args.checkpoints):
        norm_stats_list = [load_norm_stats(p) for p in norm_stats_paths]
        mixed_norm_stats = mix_norm_stats(norm_stats_list, weights=args.weights)
        save_norm_stats(mixed_norm_stats, os.path.join(args.output, "norm_stats.json"))
        print("\nCleaning GPU memory...")
        jax.clear_caches()
        gc.collect()
        time.sleep(2)
        print("\nTesting mixed checkpoint...")
        mixed_loss = test_mixed_checkpoint_jax(config, args.output, data_samples_list)
        print("\n" + "=" * 60)
        print("Results:")
        if losses:
            for i, (ckpt, loss) in enumerate(zip(args.checkpoints, losses)):
                print(f"  Ckpt {i+1}: {loss:.6f} (w={args.weights[i]:.4f})")
        print(f"  Mixed:  {mixed_loss:.6f}")
        print("=" * 60)
        if _WANDB_AVAILABLE and _wandb.run is not None:
            _wandb.log({"final/mixed_loss": mixed_loss})
            for i, (w, ckpt) in enumerate(zip(args.weights, args.checkpoints)):
                _wandb.log({f"final/weight_{i}": w, f"final/ckpt_{i}": ckpt})
    else:
        logger.warning("Incomplete norm_stats files, skipping validation test")

    if _WANDB_AVAILABLE and _wandb.run is not None:
        _wandb.finish()

    print("\n✓ Completed successfully!")


if __name__ == "__main__":
    main()
