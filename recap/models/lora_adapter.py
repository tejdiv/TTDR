"""LoRA adapter for Octo's DiffusionActionHead.

Injects low-rank A/B matrices into Dense layers of the MLPResNet score network
inside the diffusion action head. Everything else (encoder, world model) stays frozen.

At inference: W' = W + (alpha/rank) * A @ B  (A initialized normal, B initialized zeros so LoRA starts as no-op).
Only targets Dense layers in the reverse_network (MLPResNet), skipping the cond_encoder (time embedding MLP).
"""

import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze



def init_lora_params(rng, policy_params, rank=8, alpha=16):
    """Initialize LoRA A/B matrices for Dense layers in the MLPResNet score network.

    Only targets Dense layers under reverse_network (the MLPResNet inside the
    DiffusionActionHead). Skips cond_encoder (time embedding MLP) and all other
    modules.

    A is initialized with small normal values, B is initialized to zeros,
    so the initial LoRA contribution is W + (alpha/rank) * A @ B = W (no-op).

    Args:
        rng: JAX PRNG key.
        policy_params: Full Octo param dict.
        rank: LoRA rank (default 8).
        alpha: LoRA scaling factor (default 16).

    Returns:
        lora_params: dict with "layers" (per-path A/B pairs), "rank", and "alpha".
    """
    params_flat = _flatten_params(policy_params)

    layers = {}
    for path, value in params_flat.items():
        # Only target Dense kernels inside the reverse_network (MLPResNet)
        if "reverse_network" not in path or not path.endswith("kernel"):
            continue

        in_dim, out_dim = value.shape
        rng, key_a = jax.random.split(rng)
        layers[path] = {
            "A": jax.random.normal(key_a, (in_dim, rank)) * 0.01,
            "B": jnp.zeros((rank, out_dim)),
        }

    return {"layers": layers, "rank": rank, "alpha": alpha}


def apply_lora(policy_params, lora_params):
    """Merge LoRA into policy params: W' = W + (alpha/rank) * A @ B.

    Creates a new param dict — does not mutate the original.

    Args:
        policy_params: Frozen Octo param dict.
        lora_params: dict with "layers", "rank", and "alpha".

    Returns:
        merged_params: New param dict with LoRA applied.
    """
    merged = unfreeze(policy_params) if hasattr(policy_params, 'unfreeze') else _deep_copy(policy_params)
    scale = lora_params["alpha"] / lora_params["rank"]

    for path, ab in lora_params["layers"].items():
        delta = scale * (ab["A"] @ ab["B"])  # (in_dim, out_dim)
        _set_nested(merged, path, _get_nested(merged, path) + delta)

    return freeze(merged) if hasattr(policy_params, 'unfreeze') else merged


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _flatten_params(params, prefix=""):
    """Flatten a nested param dict into {'/'-separated path: array}."""
    flat = {}
    if isinstance(params, dict):
        for key, val in params.items():
            path = f"{prefix}/{key}" if prefix else key
            if isinstance(val, (dict,)):
                flat.update(_flatten_params(val, path))
            else:
                flat[path] = val
    return flat


def _get_nested(d, path):
    """Get a value from a nested dict using a '/'-separated path."""
    keys = path.split("/")
    for k in keys:
        d = d[k]
    return d


def _set_nested(d, path, value):
    """Set a value in a nested dict using a '/'-separated path."""
    keys = path.split("/")
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def _deep_copy(d):
    """Deep copy a nested dict of arrays."""
    if isinstance(d, dict):
        return {k: _deep_copy(v) for k, v in d.items()}
    return d
