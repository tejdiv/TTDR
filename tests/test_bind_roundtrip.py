"""Test that module.bind → heads["action"] → predict_action preserves LoRA params.

DiffusionActionHead.predict_action calls self.unbind() internally to get
(module, variables), then uses module.apply(variables, ...) in the denoising loop.
This test verifies that the unbind roundtrip preserves the modified params.

Three checks:
  1. bind(modified_params).heads["action"].predict_action produces different
     actions than bind(original_params).heads["action"].predict_action
  2. The bind approach matches octo_model.replace(params=modified).sample_actions
  3. LoRA-merged params produce different actions than original params

Usage:
    python -m tests.test_bind_roundtrip
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import jax
import jax.numpy as jnp
# Warm up cuDNN by running a conv BEFORE TF is imported.
# TF's CUDA plugin registration corrupts JAX's cuDNN state;
# if cuDNN is already initialized, it survives.
_x = jnp.ones((1, 8, 8, 3))
_k = jnp.ones((3, 3, 3, 16))
_ = jax.lax.conv_general_dilated(
    _x, _k, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")
)
del _x, _k

import numpy as np
from absl import app, logging

import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

from octo.model.octo_model import OctoModel
from recap.models.lora_adapter import init_lora_params, apply_lora


def main(_):
    logging.info("Loading Octo model...")
    octo_model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base-1.5")
    octo_params = octo_model.params

    # Create dummy observation and task
    example_obs = octo_model.example_batch["observation"]
    example_obs = jax.tree.map(lambda x: x[:1], example_obs)  # batch=1
    task = octo_model.create_tasks(texts=["pick up the object"])

    rng = jax.random.PRNGKey(42)

    # ─── Test 1: original vs modified params via bind ────────────────
    logging.info("Test 1: bind with original vs modified params")

    # Run transformer once (shared between both)
    # JIT-wrapped because non-JIT cuDNN can fail on CUDA 12.8 + cuDNN 8.9
    @jax.jit
    def run_transformer(params, obs, task, pad_mask):
        return octo_model.module.apply(
            {"params": params}, obs, task, pad_mask,
            train=False, method="octo_transformer",
        )

    rng, t_rng = jax.random.split(rng)
    pad_mask = example_obs["timestep_pad_mask"]
    trans_out = run_transformer(octo_params, example_obs, task, pad_mask)

    # Predict with original params via bind
    rng, act_rng_1 = jax.random.split(rng)
    action_head_orig = octo_model.module.bind(
        {"params": octo_params}
    ).heads["action"]
    actions_orig = action_head_orig.predict_action(
        trans_out, rng=act_rng_1, train=False
    )

    # Create LoRA and merge
    rng, lora_rng = jax.random.split(rng)
    lora_params = init_lora_params(lora_rng, octo_params, rank=8)
    # Make B non-zero so LoRA actually changes weights
    for path, ab in lora_params["layers"].items():
        lora_params["layers"][path]["B"] = jax.random.normal(
            lora_rng, ab["B"].shape
        ) * 0.1
    merged_params = apply_lora(octo_params, lora_params)

    # Predict with merged params via bind (same rng!)
    action_head_merged = octo_model.module.bind(
        {"params": merged_params}
    ).heads["action"]
    actions_merged = action_head_merged.predict_action(
        trans_out, rng=act_rng_1, train=False
    )

    diff_1 = float(jnp.max(jnp.abs(actions_orig - actions_merged)))
    logging.info(f"  Max abs diff (orig vs merged via bind): {diff_1:.6f}")
    assert diff_1 > 1e-4, (
        f"FAIL: bind with merged params produced same actions as original! "
        f"diff={diff_1}. LoRA not taking effect."
    )
    logging.info("  PASS: bind with different params produces different actions")

    # ─── Test 2: bind approach matches replace approach ──────────────
    logging.info("Test 2: bind approach vs replace approach")

    # replace approach: use sample_actions which runs transformer + action head
    merged_model = octo_model.replace(params=merged_params)
    actions_replace = merged_model.sample_actions(
        example_obs, task, rng=act_rng_1
    )

    diff_2 = float(jnp.max(jnp.abs(actions_merged - actions_replace)))
    logging.info(f"  Max abs diff (bind vs replace): {diff_2:.6f}")
    # These should match exactly since transformer weights are the same
    # and action head weights are the same (both use merged_params)
    assert diff_2 < 1e-4, (
        f"FAIL: bind approach differs from replace approach! diff={diff_2}. "
        f"unbind roundtrip is losing params."
    )
    logging.info("  PASS: bind approach matches replace approach")

    # ─── Test 3: different RNG seeds produce different actions ───────
    logging.info("Test 3: different RNG seeds")
    rng, act_rng_2 = jax.random.split(rng)
    actions_diff_rng = action_head_merged.predict_action(
        trans_out, rng=act_rng_2, train=False
    )

    diff_3 = float(jnp.max(jnp.abs(actions_merged - actions_diff_rng)))
    logging.info(f"  Max abs diff (same params, different RNG): {diff_3:.6f}")
    assert diff_3 > 1e-4, (
        f"FAIL: different RNG seeds produced same actions! diff={diff_3}"
    )
    logging.info("  PASS: different RNG seeds produce different actions")

    # ─── Test 4: verify transformer output is same for both param sets
    logging.info("Test 4: transformer output invariant to LoRA")
    trans_out_merged = run_transformer(merged_params, example_obs, task, pad_mask)

    tokens_orig = trans_out["readout_action"].tokens
    tokens_merged = trans_out_merged["readout_action"].tokens
    diff_4 = float(jnp.max(jnp.abs(tokens_orig - tokens_merged)))
    logging.info(f"  Max abs diff (transformer tokens): {diff_4:.6f}")
    assert diff_4 < 1e-5, (
        f"FAIL: transformer outputs differ with LoRA! diff={diff_4}. "
        f"LoRA is leaking into transformer weights."
    )
    logging.info("  PASS: transformer output is invariant to LoRA (as expected)")

    logging.info("\nAll tests passed!")
    logging.info("bind → heads['action'] → predict_action → unbind preserves params.")
    logging.info("Safe to split transformer and action head for efficiency.")


if __name__ == "__main__":
    app.run(main)
