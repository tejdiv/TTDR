"""RECAP diagnostic tests.

Route 1: Detailed logging (built into recap_adapt — see per-step logs)
Route 2: No-perturbation baseline — verify robot moves in unperturbed env
Route 3: Systematic Octo compatibility checks
Route 4: High-level architectural sanity checks

Usage:
    python tests/test_diagnostics.py --route 2   # no-perturbation baseline
    python tests/test_diagnostics.py --route 3   # octo compatibility
    python tests/test_diagnostics.py --route 4   # architecture checks
    python tests/test_diagnostics.py --route all  # run everything
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import sys
import jax
import jax.numpy as jnp

# cuDNN warmup
_x = jnp.ones((1, 8, 8, 3))
_k = jnp.ones((3, 3, 3, 16))
_ = jax.lax.conv_general_dilated(_x, _k, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"))
del _x, _k

import numpy as np

import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

from absl import app, flags, logging

FLAGS = flags.FLAGS
flags.DEFINE_string("route", "all", "Which diagnostic route to run: 2, 3, 4, or all")


# ---------------------------------------------------------------------------
# Route 2: No-perturbation baseline
# ---------------------------------------------------------------------------

def route2_no_perturbation():
    """Run Octo zero-shot in unperturbed SimplerEnv. Verify robot actually moves."""
    logging.info("=" * 60)
    logging.info("ROUTE 2: No-perturbation baseline")
    logging.info("=" * 60)

    from octo.model.octo_model import OctoModel
    from recap.envs.perturbations import make_env
    from recap.training.recap_adaptation import _run_transformer, sample_actions_from_readouts

    octo_model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base")
    octo_params = octo_model.params

    action_mean = jnp.array(octo_model.dataset_statistics["bridge_dataset"]["action"]["mean"])
    action_std = jnp.array(octo_model.dataset_statistics["bridge_dataset"]["action"]["std"])

    env_id = "PutCarrotOnPlateInScene-v1"
    logging.info(f"Creating UNPERTURBED env: {env_id}")
    env = make_env(env_id)  # No perturbation

    task = octo_model.create_tasks(texts=["put carrot on plate"])
    rng = jax.random.PRNGKey(0)

    obs, _ = env.reset()
    logging.info(f"Obs keys: {list(obs.keys())}")
    logging.info(f"image_primary shape: {obs['image_primary'].shape}")
    logging.info(f"pad_mask: {obs['pad_mask']}")

    prev_actions = []
    prev_images = []

    for step in range(20):
        rng, act_rng = jax.random.split(rng)
        pad_mask = obs["pad_mask"]

        trans_out = _run_transformer(
            octo_model.module, octo_params, obs, task, pad_mask
        )
        norm_actions = sample_actions_from_readouts(
            octo_model, octo_params, trans_out, act_rng
        )

        # Denormalize
        actions = norm_actions * action_std[None] + action_mean[None]
        action_np = np.array(actions[0, 0])

        # Log action stats
        norm_a = np.array(norm_actions[0, 0])
        logging.info(
            f"  Step {step}: "
            f"norm_action=[{', '.join(f'{x:.3f}' for x in norm_a)}] "
            f"denorm_action=[{', '.join(f'{x:.3f}' for x in action_np)}]"
        )

        obs, reward, terminated, truncated, info = env.step(action_np)
        prev_actions.append(action_np)

        # Check image changes (robot should be moving)
        img = np.array(obs["image_primary"][0, -1])  # last frame
        prev_images.append(img)

        if terminated or truncated:
            logging.info(f"  Episode ended at step {step}: reward={reward}, info={info}")
            break

    # Analyze: did actions vary? Did images change?
    actions_arr = np.stack(prev_actions)
    action_range = actions_arr.max(axis=0) - actions_arr.min(axis=0)
    logging.info(f"\nAction range over {len(prev_actions)} steps: {action_range}")
    logging.info(f"Action mean magnitude: {np.abs(actions_arr).mean(axis=0)}")

    if len(prev_images) > 1:
        diffs = [np.abs(prev_images[i].astype(float) - prev_images[i-1].astype(float)).mean()
                 for i in range(1, len(prev_images))]
        logging.info(f"Mean frame-to-frame pixel diff: {np.mean(diffs):.2f}")
        if np.mean(diffs) < 0.5:
            logging.warning("WARNING: Very low pixel differences — robot may not be moving!")
        else:
            logging.info("OK: Robot appears to be moving (pixel diffs > 0.5)")

    logging.info("Route 2 complete.")


# ---------------------------------------------------------------------------
# Route 3: Systematic Octo compatibility
# ---------------------------------------------------------------------------

def route3_octo_compatibility():
    """Check Octo's expectations vs what we're providing."""
    logging.info("=" * 60)
    logging.info("ROUTE 3: Octo compatibility checks")
    logging.info("=" * 60)

    from octo.model.octo_model import OctoModel

    octo_model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base")

    # 1. Check expected observation keys
    logging.info("\n--- Check 1: Expected observation spec ---")
    bound = octo_model.module.bind({"params": octo_model.params})
    if hasattr(bound, 'observation_tokenizers'):
        for name, tok in bound.observation_tokenizers.items():
            logging.info(f"  Tokenizer '{name}': {type(tok).__name__}")
    else:
        # Inspect the module config instead
        module = octo_model.module
        logging.info(f"  Module type: {type(module).__name__}")
        if hasattr(module, 'observation_tokenizers'):
            for name, tok in module.observation_tokenizers.items():
                logging.info(f"  Tokenizer '{name}': {type(tok).__name__}")
        else:
            logging.info("  (observation_tokenizers not directly accessible — checking config)")
            config = octo_model.config
            if "model" in config and "observation_tokenizers" in config["model"]:
                for name, spec in config["model"]["observation_tokenizers"].items():
                    logging.info(f"  Tokenizer '{name}': {spec}")
            else:
                logging.info(f"  Config keys: {list(config.keys()) if isinstance(config, dict) else 'N/A'}")

    # 2. Check dataset statistics
    logging.info("\n--- Check 2: Dataset statistics ---")
    ds_stats = octo_model.dataset_statistics
    for ds_name, stats in ds_stats.items():
        logging.info(f"  Dataset: {ds_name}")
        for key, val in stats.items():
            if isinstance(val, dict):
                for subkey, subval in val.items():
                    arr = np.array(subval)
                    logging.info(f"    {key}/{subkey}: shape={arr.shape} range=[{arr.min():.4f}, {arr.max():.4f}]")
            else:
                arr = np.array(val)
                logging.info(f"    {key}: shape={arr.shape}")

    # 3. Check action head config
    logging.info("\n--- Check 3: Action head config ---")
    try:
        action_head = octo_model.module.bind({"params": octo_model.params}).heads["action"]
        logging.info(f"  Action head type: {type(action_head).__name__}")
        logging.info(f"  Action dim: {action_head.action_dim}")
        logging.info(f"  Pred horizon: {action_head.pred_horizon}")
        if hasattr(action_head, "diffusion_steps"):
            logging.info(f"  Diffusion steps: {action_head.diffusion_steps}")
    except Exception as e:
        logging.info(f"  Could not bind action head directly: {e}")
        config = octo_model.config
        if "model" in config and "heads" in config["model"]:
            logging.info(f"  Action head config: {config['model']['heads']}")

    # 4. Check what happens with single vs dual camera
    logging.info("\n--- Check 4: Single vs dual camera impact ---")
    logging.info("  Octo was trained with image_primary + image_wrist on Bridge V2.")
    logging.info("  SimplerEnv only provides image_primary (3rd_view_camera).")
    logging.info("  Missing wrist camera means some tokens are zero/masked.")
    logging.info("  This is expected to degrade performance somewhat but should still work.")

    # 5. Verify action space matches
    logging.info("\n--- Check 5: Action space ---")
    action_mean = np.array(octo_model.dataset_statistics["bridge_dataset"]["action"]["mean"])
    action_std = np.array(octo_model.dataset_statistics["bridge_dataset"]["action"]["std"])
    logging.info(f"  Bridge V2 action dim: {action_mean.shape[0]}")
    logging.info(f"  Action mean: {action_mean}")
    logging.info(f"  Action std: {action_std}")
    logging.info("  Expected: 7-dim (6 joint velocities + 1 gripper)")

    # 6. Check if T5 tokenization differs with/without indicator suffix
    logging.info("\n--- Check 6: Language indicator token test ---")
    task_base = octo_model.create_tasks(texts=["put carrot on plate"])
    task_improved = octo_model.create_tasks(
        texts=["put the carrot on the plate World Model Advantage: positive"]
    )
    base_leaves = jax.tree_util.tree_leaves(task_base)
    imp_leaves = jax.tree_util.tree_leaves(task_improved)
    total_diff = sum(
        float(jnp.abs(jnp.asarray(b).astype(jnp.float32) - jnp.asarray(i).astype(jnp.float32)).sum())
        for b, i in zip(base_leaves, imp_leaves)
    )
    logging.info(f"  Total leaf diff between base and improved tasks: {total_diff:.2f}")
    if total_diff > 0:
        logging.info("  OK — indicator suffix changes the tokenization")
    else:
        logging.warning("  WARNING — indicator suffix has NO effect on tokenization!")

    # 7. Run a forward pass and check readout shapes
    logging.info("\n--- Check 7: Forward pass shape verification ---")
    from recap.training.recap_adaptation import _run_transformer

    dummy_img = jnp.zeros((1, 2, 256, 256, 3), dtype=jnp.uint8)
    dummy_obs = {
        "image_primary": dummy_img,
        "pad_mask": jnp.array([[1.0, 1.0]]),
    }
    pad_mask = dummy_obs["pad_mask"]
    trans_out = _run_transformer(
        octo_model.module, octo_model.params, dummy_obs, task_base, pad_mask
    )
    tokens = trans_out["readout_action"].tokens
    logging.info(f"  readout_action tokens shape: {tokens.shape}")
    logging.info(f"  Expected: (1, T, N, 768) where T=window_size, N=num_readouts")

    logging.info("\nRoute 3 complete.")


# ---------------------------------------------------------------------------
# Route 4: High-level architecture checks
# ---------------------------------------------------------------------------

def route4_architecture():
    """Verify architectural assumptions: LoRA targets, gradient flow, reward signal."""
    logging.info("=" * 60)
    logging.info("ROUTE 4: Architecture checks")
    logging.info("=" * 60)

    from octo.model.octo_model import OctoModel
    from recap.models.lora_adapter import init_lora_params, apply_lora, _flatten_params

    octo_model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base")
    octo_params = octo_model.params

    # 1. LoRA target paths (action-head only)
    logging.info("\n--- Check 1: Action-head LoRA targets ---")
    rng = jax.random.PRNGKey(0)
    lora_ah = init_lora_params(rng, octo_params, rank=8, backbone_lora=False)
    logging.info(f"  Action-head-only LoRA: {len(lora_ah['layers'])} layers")
    for path in sorted(lora_ah["layers"].keys()):
        ab = lora_ah["layers"][path]
        logging.info(f"    {path}: A={ab['A'].shape}, B={ab['B'].shape}")

    # 2. LoRA target paths (backbone + action head)
    logging.info("\n--- Check 2: Backbone + action-head LoRA targets ---")
    lora_bb = init_lora_params(rng, octo_params, rank=8, backbone_lora=True)
    logging.info(f"  Backbone LoRA: {len(lora_bb['layers'])} layers")
    backbone_count = 0
    action_head_count = 0
    for path in sorted(lora_bb["layers"].keys()):
        ab = lora_bb["layers"][path]
        if "reverse_network" in path:
            action_head_count += 1
        else:
            backbone_count += 1
        logging.info(f"    {path}: A={ab['A'].shape}, B={ab['B'].shape}")
    logging.info(f"  Backbone attention layers: {backbone_count}")
    logging.info(f"  Action head layers: {action_head_count}")

    # 3. Verify LoRA starts as no-op
    logging.info("\n--- Check 3: LoRA no-op at init ---")
    merged = apply_lora(octo_params, lora_ah)
    params_flat_orig = _flatten_params(octo_params)
    params_flat_merged = _flatten_params(merged)
    max_diff = 0.0
    for path in params_flat_orig:
        d = float(jnp.abs(params_flat_orig[path] - params_flat_merged[path]).max())
        max_diff = max(max_diff, d)
    logging.info(f"  Max param diff after LoRA init: {max_diff:.2e}")
    if max_diff < 1e-6:
        logging.info("  OK — LoRA is a no-op at initialization (B=0)")
    else:
        logging.warning(f"  WARNING — LoRA changes params at init by {max_diff:.2e}!")

    # 4. World model reward signal variance
    logging.info("\n--- Check 4: World model reward signal ---")
    logging.info("  Checking if WM produces meaningful reward variance...")
    try:
        from recap.training.recap_adaptation import load_world_model
        from recap.losses.tracking_reward import tracking_reward

        class MockConfig:
            wm_checkpoint = "hf://4manifold/ttdr-world-model"
            wm_projection_head_kwargs = {"hidden_dim": 1024, "output_dim": 512}
            wm_dynamics_predictor_kwargs = {"hidden_dim": 2048, "num_layers": 4, "output_dim": 512}
            wm_value_head_kwargs = {"hidden_dim": 256, "num_layers": 2}

        wm_model, wm_params = load_world_model(MockConfig())

        # Random embeddings to check reward range
        z_t = jax.random.normal(jax.random.PRNGKey(0), (4, 768))
        z_t1 = jax.random.normal(jax.random.PRNGKey(1), (4, 768))
        z_target = jax.random.normal(jax.random.PRNGKey(2), (4, 768))

        rewards = tracking_reward(z_t, z_t1, z_target, wm_params, wm_model.apply)
        logging.info(f"  Random inputs → rewards: {np.array(rewards)}")
        logging.info(f"  Reward range: [{float(rewards.min()):.4f}, {float(rewards.max()):.4f}]")
        logging.info(f"  Reward std: {float(rewards.std()):.4f}")

        # Same target = same embedding → should give high reward
        z_target_same = z_t.copy()
        rewards_same = tracking_reward(z_t, z_t1, z_target_same, wm_params, wm_model.apply)
        logging.info(f"  Same-input rewards: {np.array(rewards_same)}")
        logging.info(f"  (Should be higher than random if WM learned meaningful dynamics)")

    except Exception as e:
        logging.warning(f"  Could not load world model: {e}")

    # 5. K=3 action diversity check
    logging.info("\n--- Check 5: K=3 action diversity ---")
    logging.info("  Verifying separate RNG keys produce different actions...")
    from recap.training.recap_adaptation import _run_transformer, sample_K_actions

    dummy_img = jnp.zeros((1, 2, 256, 256, 3), dtype=jnp.uint8)
    dummy_obs = {"image_primary": dummy_img, "pad_mask": jnp.array([[1.0, 1.0]])}
    task = octo_model.create_tasks(texts=["put carrot on plate"])

    trans_out = _run_transformer(
        octo_model.module, octo_params, dummy_obs, task, dummy_obs["pad_mask"]
    )

    rng = jax.random.PRNGKey(42)
    actions_list = sample_K_actions(octo_model, octo_params, trans_out, 3, 1, rng)

    for i in range(3):
        a = np.array(actions_list[i][0, 0])
        logging.info(f"  K={i}: [{', '.join(f'{x:.4f}' for x in a)}]")

    # Check pairwise differences
    for i in range(3):
        for j in range(i+1, 3):
            diff = float(jnp.abs(actions_list[i] - actions_list[j]).mean())
            logging.info(f"  K={i} vs K={j} mean abs diff: {diff:.6f}")
            if diff < 1e-4:
                logging.warning(f"  WARNING — K={i} and K={j} are nearly identical!")

    logging.info("\nRoute 4 complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(_):
    route = FLAGS.route

    if route in ("2", "all"):
        route2_no_perturbation()

    if route in ("3", "all"):
        route3_octo_compatibility()

    if route in ("4", "all"):
        route4_architecture()

    logging.info("\nAll requested diagnostics complete.")


if __name__ == "__main__":
    app.run(main)
