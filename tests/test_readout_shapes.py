"""Compare transformer output shapes between octo-base v1.0 and v1.5."""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp

_x = jnp.ones((1, 8, 8, 3))
_k = jnp.ones((3, 3, 3, 16))
_ = jax.lax.conv_general_dilated(_x, _k, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"))
del _x, _k

import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
from octo.model.octo_model import OctoModel


def inspect_recursive(obj, prefix="", depth=0):
    """Recursively inspect object attributes and shapes."""
    if depth > 4:
        return
    if hasattr(obj, "shape"):
        print(f"  {prefix}: shape={obj.shape}, dtype={obj.dtype}")
    elif isinstance(obj, dict):
        for k, v in sorted(obj.items()):
            inspect_recursive(v, f"{prefix}/{k}" if prefix else k, depth + 1)
    elif hasattr(obj, "__dict__"):
        print(f"  {prefix}: {type(obj).__name__}")
        for k, v in sorted(vars(obj).items()):
            if not k.startswith("_"):
                inspect_recursive(v, f"{prefix}.{k}", depth + 1)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            inspect_recursive(v, f"{prefix}[{i}]", depth + 1)
    else:
        print(f"  {prefix}: {type(obj).__name__} = {obj}")


for name, model_id in [("v1.0", "hf://rail-berkeley/octo-base"), ("v1.5", "hf://rail-berkeley/octo-base-1.5")]:
    print(f"\n{'='*60}")
    print(f"  {name}: {model_id}")
    print(f"{'='*60}")
    model = OctoModel.load_pretrained(model_id)

    B, T = 1, 2
    images = jnp.zeros((B, T, 256, 256, 3), dtype=jnp.uint8)

    if name == "v1.0":
        obs = {"image_primary": images, "pad_mask": jnp.ones((B, T), dtype=jnp.float32)}
    else:
        obs = {
            "image_primary": images,
            "timestep_pad_mask": jnp.ones((B, T), dtype=bool),
            "pad_mask_dict": {
                "image_primary": jnp.ones((B, T), dtype=bool),
                "image_wrist": jnp.zeros((B, T), dtype=bool),
                "timestep": jnp.ones((B, T), dtype=bool),
            },
        }

    task = model.create_tasks(texts=["put carrot on plate"])
    pad_mask = jnp.ones((B, T), dtype=jnp.float32)

    transformer_out = model.run_transformer(obs, task, pad_mask, train=False)

    print("\nTransformer output keys:", sorted(transformer_out.keys()))
    for k in sorted(transformer_out.keys()):
        v = transformer_out[k]
        inspect_recursive(v, k)

    # Config
    cfg = model.config.get("model", {})
    print(f"\n  use_correct_attention: {cfg.get('use_correct_attention', 'NOT SET')}")
    print(f"  repeat_task_tokens: {cfg.get('repeat_task_tokens', 'NOT SET')}")
    print(f"  readouts: {cfg.get('readouts', 'NOT SET')}")
