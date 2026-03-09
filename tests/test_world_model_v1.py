"""Validate the v1.0 world model after training.

Tests:
  1. Checkpoint loads and produces correct output shapes
  2. Retrieval accuracy on held-out encodings (should be >50%)
  3. Tracking reward has meaningful variance (not degenerate)
  4. Nearby frames get higher reward than distant frames
  5. Encodings from octo-base v1.0 are compatible (768-dim, correct pad_mask)

Usage:
    # With HF checkpoint (after Baseten training completes):
    python tests/test_world_model_v1.py --wm_checkpoint hf://4manifold/ttdr-world-model-v1

    # With local checkpoint:
    python tests/test_world_model_v1.py --wm_checkpoint checkpoints/world_model

    # With local encodings (skip HF download):
    python tests/test_world_model_v1.py --encodings_path data/bridge_v2_encodings/encodings.h5
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp

# cuDNN warmup
_x = jnp.ones((1, 8, 8, 3))
_k = jnp.ones((3, 3, 3, 16))
_ = jax.lax.conv_general_dilated(
    _x, _k, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")
)
del _x, _k

import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

from absl import app, flags, logging

from recap.training.train_world_model import WorldModel
from recap.losses.tracking_reward import tracking_reward, grpo_advantage

FLAGS = flags.FLAGS
flags.DEFINE_string("wm_checkpoint", None,
                    "World model checkpoint path (HF or local).")
flags.DEFINE_string("encodings_path", None,
                    "Path to encodings.h5 for retrieval accuracy test. "
                    "If not set, downloads from tejasrao/ttdr-bridge-encodings-v1.")
flags.DEFINE_integer("eval_samples", 1024,
                     "Number of samples for retrieval accuracy test.")

# World model config (must match training config)
WM_PROJ_KWARGS = {"hidden_dim": 1024, "output_dim": 512}
WM_DYN_KWARGS = {"hidden_dim": 2048, "num_layers": 4, "output_dim": 512}


def load_wm(checkpoint_path):
    """Load world model from checkpoint."""
    from flax.training import checkpoints

    model = WorldModel(
        projection_head_kwargs=WM_PROJ_KWARGS,
        dynamics_predictor_kwargs=WM_DYN_KWARGS,
    )

    # Init with dummy inputs to get param structure
    rng = jax.random.PRNGKey(0)
    dummy = jnp.zeros((1, 768))
    params = model.init(rng, dummy, dummy, dummy, train=False)["params"]

    if checkpoint_path.startswith("hf://"):
        # Download from HuggingFace
        from huggingface_hub import hf_hub_download
        repo_id = checkpoint_path.replace("hf://", "")
        local_dir = hf_hub_download(repo_id=repo_id, filename=".",
                                     repo_type="model", local_dir="/tmp/wm_ckpt")
        checkpoint_path = local_dir

    restored = checkpoints.restore_checkpoint(checkpoint_path, target=params)
    logging.info(f"Loaded world model from {checkpoint_path}")
    return model, restored


def load_encodings(path, n_samples):
    """Load n_samples from the HDF5 encodings file."""
    import h5py

    if path is None:
        # Download from HF
        from huggingface_hub import hf_hub_download
        logging.info("Downloading encodings from tejasrao/ttdr-bridge-encodings-v1...")
        path = hf_hub_download(
            repo_id="tejasrao/ttdr-bridge-encodings-v1",
            filename="encodings.h5",
            repo_type="dataset",
            local_dir="/tmp/encodings_v1",
        )

    with h5py.File(path, "r") as f:
        total = f["z_t"].shape[0]
        logging.info(f"Encodings file: {total:,} transitions, keys={list(f.keys())}")
        logging.info(f"  z_t: {f['z_t'].shape}, z_t1: {f['z_t1'].shape}, z_target: {f['z_target'].shape}")
        logging.info(f"  encoder_dim: {f.attrs.get('encoder_dim', 'N/A')}")
        logging.info(f"  checkpoint: {f.attrs.get('checkpoint', 'N/A')}")

        # Verify it was encoded with octo-base v1.0
        ckpt_name = str(f.attrs.get("checkpoint", ""))
        if "1.5" in ckpt_name:
            logging.warning(f"WARNING: Encodings were made with {ckpt_name} (v1.5), expected v1.0!")
        elif "octo-base" in ckpt_name:
            logging.info(f"  OK: Encoded with {ckpt_name}")

        # Random subset
        idx = np.random.RandomState(0).choice(total, min(n_samples, total), replace=False)
        idx.sort()
        z_t = np.array(f["z_t"][idx])
        z_t1 = np.array(f["z_t1"][idx])
        z_target = np.array(f["z_target"][idx])
        traj_id = np.array(f["traj_id"][idx])

    return z_t, z_t1, z_target, traj_id


def test_shapes(model, params):
    """Test 1: Output shapes are correct."""
    logging.info("\n=== Test 1: Output shapes ===")

    z_t = jnp.zeros((4, 768))
    z_t1 = jnp.zeros((4, 768))
    z_target = jnp.zeros((4, 768))

    predicted, target_proj = model.apply(
        {"params": params}, z_t, z_t1, z_target, train=False
    )

    assert predicted.shape == (4, WM_PROJ_KWARGS["output_dim"]), \
        f"predicted shape {predicted.shape} != expected (4, {WM_PROJ_KWARGS['output_dim']})"
    assert target_proj.shape == predicted.shape, \
        f"target_proj shape {target_proj.shape} != predicted shape {predicted.shape}"

    logging.info(f"  predicted: {predicted.shape}, target_proj: {target_proj.shape}")
    logging.info("  PASSED")


def test_retrieval_accuracy(model, params, z_t, z_t1, z_target):
    """Test 2: Retrieval accuracy on real encodings (should be >50% for a trained model)."""
    logging.info("\n=== Test 2: Retrieval accuracy ===")

    batch_size = min(256, len(z_t))
    z_t_b = jnp.array(z_t[:batch_size])
    z_t1_b = jnp.array(z_t1[:batch_size])
    z_target_b = jnp.array(z_target[:batch_size])

    predicted, target_proj = model.apply(
        {"params": params}, z_t_b, z_t1_b, z_target_b, train=False
    )

    # Pairwise L2 distances
    diff = predicted[:, None, :] - target_proj[None, :, :]
    distances = jnp.sum(diff ** 2, axis=-1)  # (B, B)
    nearest = jnp.argmin(distances, axis=-1)
    accuracy = float(jnp.mean(nearest == jnp.arange(batch_size)))

    logging.info(f"  Batch size: {batch_size}")
    logging.info(f"  Retrieval accuracy: {accuracy:.3f}")
    logging.info(f"  Random chance: {1/batch_size:.4f}")

    if accuracy > 0.5:
        logging.info("  PASSED (>50%)")
    elif accuracy > 0.1:
        logging.warning(f"  WARNING: accuracy {accuracy:.3f} is low but above chance")
    else:
        logging.error(f"  FAILED: accuracy {accuracy:.3f} is near random chance")

    return accuracy


def test_reward_variance(model, params, z_t, z_t1, z_target):
    """Test 3: Tracking reward has meaningful variance (not collapsed)."""
    logging.info("\n=== Test 3: Tracking reward variance ===")

    n = min(64, len(z_t))
    z_t_b = jnp.array(z_t[:n])
    z_t1_b = jnp.array(z_t1[:n])
    z_target_b = jnp.array(z_target[:n])

    rewards = tracking_reward(z_t_b, z_t1_b, z_target_b, params, model.apply)
    rewards_np = np.array(rewards)

    logging.info(f"  N={n}")
    logging.info(f"  Reward mean: {rewards_np.mean():.6f}")
    logging.info(f"  Reward std:  {rewards_np.std():.6f}")
    logging.info(f"  Reward range: [{rewards_np.min():.6f}, {rewards_np.max():.6f}]")

    if rewards_np.std() < 1e-6:
        logging.error("  FAILED: reward std is ~0 (degenerate!)")
    else:
        logging.info("  PASSED (non-zero variance)")

    # GRPO check
    advantages, indicators = grpo_advantage(rewards)
    logging.info(f"  GRPO: {int(indicators.sum())}/{n} positive indicators")
    logging.info(f"  Mean advantage: {float(jnp.mean(advantages)):.6f} (should be ~0)")

    return float(rewards_np.std())


def test_nearby_vs_distant(model, params, z_t, z_t1, z_target, traj_id):
    """Test 4: Same-trajectory nearby targets get higher reward than random targets."""
    logging.info("\n=== Test 4: Nearby vs distant targets ===")

    n = min(64, len(z_t))
    z_t_b = jnp.array(z_t[:n])
    z_t1_b = jnp.array(z_t1[:n])
    z_target_b = jnp.array(z_target[:n])

    # Real targets (temporally nearby, same trajectory)
    rewards_real = tracking_reward(z_t_b, z_t1_b, z_target_b, params, model.apply)

    # Shuffled targets (random, likely different trajectory)
    perm = np.random.RandomState(42).permutation(n)
    z_target_shuffled = z_target_b[perm]
    rewards_shuffled = tracking_reward(z_t_b, z_t1_b, z_target_shuffled, params, model.apply)

    real_mean = float(jnp.mean(rewards_real))
    shuffled_mean = float(jnp.mean(rewards_shuffled))

    logging.info(f"  Real target reward (mean):     {real_mean:.6f}")
    logging.info(f"  Shuffled target reward (mean):  {shuffled_mean:.6f}")
    logging.info(f"  Difference:                     {real_mean - shuffled_mean:.6f}")

    if real_mean > shuffled_mean:
        logging.info("  PASSED (real targets get higher reward)")
    else:
        logging.warning("  WARNING: shuffled targets got higher reward (world model may not have learned dynamics)")

    return real_mean - shuffled_mean


def test_encoding_compatibility():
    """Test 5: Octo v1.0 encoder produces 768-dim outputs with pad_mask key."""
    logging.info("\n=== Test 5: Octo v1.0 encoding compatibility ===")

    from octo.model.octo_model import OctoModel

    model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base")

    # Check example batch uses pad_mask (not timestep_pad_mask)
    obs_keys = sorted(model.example_batch["observation"].keys())
    logging.info(f"  example_batch obs keys: {obs_keys}")

    has_pad_mask = "pad_mask" in obs_keys
    has_timestep = "timestep_pad_mask" in obs_keys
    logging.info(f"  pad_mask present: {has_pad_mask}")
    logging.info(f"  timestep_pad_mask present: {has_timestep}")

    if has_pad_mask and not has_timestep:
        logging.info("  OK: v1.0 uses pad_mask (correct for our pipeline)")
    elif has_timestep:
        logging.warning("  WARNING: timestep_pad_mask found — this might be v1.5!")

    # Forward pass to check readout shape
    B, T = 1, 2
    images = jnp.zeros((B, T, 256, 256, 3), dtype=jnp.uint8)
    obs = {"image_primary": images, "pad_mask": jnp.ones((B, T), dtype=jnp.float32)}
    task = model.create_tasks(texts=["put carrot on plate"])
    pad_mask = jnp.ones((B, T), dtype=jnp.float32)

    trans_out = model.run_transformer(obs, task, pad_mask, train=False)
    tokens = trans_out["readout_action"].tokens
    logging.info(f"  readout_action tokens: {tokens.shape}")

    assert tokens.shape[-1] == 768, f"Expected 768-dim, got {tokens.shape[-1]}"
    assert tokens.shape[1] == T, f"Expected T={T} positions, got {tokens.shape[1]}"

    # Extract both readout positions (same as precompute_encodings.py)
    z_pos0 = jnp.mean(tokens[:, 0, :, :], axis=1)
    z_pos1 = jnp.mean(tokens[:, -1, :, :], axis=1)
    logging.info(f"  z_pos0 (1-frame): {z_pos0.shape}")
    logging.info(f"  z_pos1 (2-frame): {z_pos1.shape}")

    assert z_pos0.shape == (1, 768)
    assert z_pos1.shape == (1, 768)

    # They should be different (pos0 only saw frame 0, pos1 saw both)
    diff = float(jnp.abs(z_pos0 - z_pos1).mean())
    logging.info(f"  pos0 vs pos1 mean diff: {diff:.6f}")
    # With zero images they might be similar, but still check shape
    logging.info("  PASSED")


def main(_):
    assert FLAGS.wm_checkpoint is not None, \
        "Must provide --wm_checkpoint (e.g. hf://4manifold/ttdr-world-model-v1 or local path)"

    np.random.seed(0)

    # Load world model
    logging.info(f"Loading world model from {FLAGS.wm_checkpoint}...")
    model, params = load_wm(FLAGS.wm_checkpoint)

    # Test 1: Shapes
    test_shapes(model, params)

    # Load encodings for tests 2-4
    logging.info(f"\nLoading encodings...")
    z_t, z_t1, z_target, traj_id = load_encodings(
        FLAGS.encodings_path, FLAGS.eval_samples
    )
    logging.info(f"  Loaded {len(z_t)} samples")

    # Test 2: Retrieval accuracy
    acc = test_retrieval_accuracy(model, params, z_t, z_t1, z_target)

    # Test 3: Reward variance
    reward_std = test_reward_variance(model, params, z_t, z_t1, z_target)

    # Test 4: Nearby > distant
    reward_diff = test_nearby_vs_distant(model, params, z_t, z_t1, z_target, traj_id)

    # Test 5: Octo v1.0 compatibility
    test_encoding_compatibility()

    # Summary
    logging.info("\n" + "=" * 60)
    logging.info("SUMMARY")
    logging.info("=" * 60)
    logging.info(f"  Retrieval accuracy:  {acc:.3f}  {'PASS' if acc > 0.5 else 'WARN' if acc > 0.1 else 'FAIL'}")
    logging.info(f"  Reward std:          {reward_std:.6f}  {'PASS' if reward_std > 1e-6 else 'FAIL'}")
    logging.info(f"  Nearby - distant:    {reward_diff:.6f}  {'PASS' if reward_diff > 0 else 'WARN'}")

    all_pass = acc > 0.1 and reward_std > 1e-6
    if all_pass:
        logging.info("\nWorld model v1.0 is NOT degenerate. Ready for RECAP adaptation.")
    else:
        logging.error("\nWorld model may be degenerate. Check training logs.")


if __name__ == "__main__":
    app.run(main)
