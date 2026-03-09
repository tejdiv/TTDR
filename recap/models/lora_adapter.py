"""LoRA adapter for Octo's DiffusionActionHead.

Injects low-rank A/B matrices into Dense layers of the MLPResNet score network
inside the diffusion action head. Everything else (encoder, world model) stays frozen.

At inference: W' = W + (alpha/rank) * A @ B  (A initialized normal, B initialized zeros so LoRA starts as no-op).
Only targets Dense layers in the reverse_network (MLPResNet), skipping the cond_encoder (time embedding MLP).
"""

import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze



# Module-level shape registry — stores original param shapes for 3D→2D reshape.
# Kept outside lora_params to avoid JIT tracing issues.
_lora_orig_shapes = {}


def init_lora_params(rng, policy_params, rank=8, alpha=16, backbone_lora=False):
    """Initialize LoRA A/B matrices for Dense layers.

    Targets:
      - Always: Dense kernels in reverse_network (MLPResNet score network in DiffusionActionHead)
      - If backbone_lora=True: also attention Q/K/V/out kernels in transformer encoder blocks
        (MultiHeadDotProductAttention). This enables end-to-end gradient flow from tracking
        reward through diffusion back to the transformer cross-attention layers.

    A is initialized with small normal values, B is initialized to zeros,
    so the initial LoRA contribution is W + (alpha/rank) * A @ B = W (no-op).

    Args:
        rng: JAX PRNG key.
        policy_params: Full Octo param dict.
        rank: LoRA rank (default 8).
        alpha: LoRA scaling factor (default 16).
        backbone_lora: If True, also target transformer attention layers.

    Returns:
        lora_params: dict with "layers" (per-path A/B pairs), "rank", "alpha",
                     and "backbone_lora" flag.
    """
    params_flat = _flatten_params(policy_params)

    layers = {}
    for path, value in params_flat.items():
        if not path.endswith("kernel"):
            continue

        # Action head: Dense kernels in reverse_network (always, 2D)
        is_action_head = "reverse_network" in path and value.ndim == 2

        # Backbone: attention Q/K/V/out kernels in transformer encoder blocks
        # These are 3D (embed, heads, head_dim) or 2D — we reshape to 2D for LoRA
        is_backbone_attn = (
            backbone_lora
            and "MultiHeadDotProductAttention" in path
            and any(k in path for k in ("/query/", "/key/", "/value/", "/out/"))
        )

        if not (is_action_head or is_backbone_attn):
            continue

        # For 3D attention kernels, reshape to 2D for LoRA:
        #   Q/K/V: (features, heads, head_dim) → in=features, out=heads*head_dim
        #   out:   (heads, head_dim, features) → in=heads*head_dim, out=features
        if value.ndim == 2:
            in_dim, out_dim = value.shape
        elif value.ndim == 3 and "/out/" in path:
            in_dim = value.shape[0] * value.shape[1]
            out_dim = value.shape[2]
        elif value.ndim == 3:
            in_dim = value.shape[0]
            out_dim = value.shape[1] * value.shape[2]
        else:
            continue

        rng, key_a = jax.random.split(rng)
        layers[path] = {
            "A": jax.random.normal(key_a, (in_dim, rank)) * 0.01,
            "B": jnp.zeros((rank, out_dim)),
        }
        _lora_orig_shapes[path] = value.shape

    # Indicator embedding: learned (768,) vector added to readout tokens when I=1.
    # Initialized to zeros so it's a no-op at start (like LoRA's B matrix).
    indicator_embed = jnp.zeros((768,))

    return {
        "layers": layers, "rank": rank, "alpha": alpha,
        "backbone_lora": backbone_lora,
        "indicator_embed": indicator_embed,
    }


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
        delta = scale * (ab["A"] @ ab["B"])  # (in_dim, out_dim) flat
        orig_shape = _lora_orig_shapes.get(path, delta.shape)
        if delta.shape != orig_shape:
            delta = delta.reshape(orig_shape)
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
