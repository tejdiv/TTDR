"""Phase 1: Train VQ-VAE tokenizer on Bridge V2 frames.

Loads raw RGB frames from Bridge V2 dataset, trains encoder (ViT) + decoder (CNN)
+ vector quantizer to reconstruct frames from discrete tokens.

Bridge V2 images (image_0) are 480x640 — resized to 256x256 in the data pipeline.
Resize runs on CPU via tf.data, fully overlapped with GPU training.

Usage:
    python -m recap.training.train_tokenizer --config configs/train_tokenizer.yaml
"""

import os
import time
from functools import partial
from typing import Tuple

from absl import app, flags, logging
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
import yaml
from flax.training import train_state, checkpoints

from recap.models.vqvae_tokenizer import VQVAETokenizer

FLAGS = flags.FLAGS
flags.DEFINE_string("config", "configs/train_tokenizer.yaml", "Path to config file.",
                    allow_override=True)


def load_bridge_frames(config):
    """Load Bridge V2 frames as a tf.data pipeline.

    Bridge V2 raw images are 480x640x3 uint8. Pipeline:
      1. Extract individual frames from episode trajectories
      2. Resize 480x640 → 256x256 (bilinear, on CPU)
      3. Normalize uint8 [0,255] → float32 [0,1]
      4. Shuffle, batch, prefetch

    Returns tf.data.Dataset yielding (B, 256, 256, 3) float32 batches.
    """
    cache_dir = config["data"].get("cache_dir", None)
    builder_kwargs = {}
    if cache_dir:
        builder_kwargs["data_dir"] = cache_dir

    ds = tfds.load(
        "bridge_dataset",
        split="train",
        **builder_kwargs,
    )

    def extract_steps(episode):
        """RLDS: episode["steps"] is a nested tf.data.Dataset of step dicts."""
        return episode["steps"]

    ds = ds.flat_map(extract_steps)

    def extract_image(step):
        return step["observation"]["image_0"]  # (480, 640, 3)

    ds = ds.map(extract_image, num_parallel_calls=tf.data.AUTOTUNE)

    def preprocess(image):
        # Resize 480x640 → 256x256, normalize to [0, 1]
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.resize(image, [256, 256], method="bilinear")
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image

    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(config["data"].get("shuffle_buffer", 10000))
    ds = ds.batch(config["data"]["batch_size"], drop_remainder=True)
    ds = ds.prefetch(4)  # Multiple batches ready to avoid CPU stalls
    return ds


def create_train_state(config, rng):
    """Initialize VQ-VAE model, optimizer, and training state."""
    model_cfg = config["model"]
    model = VQVAETokenizer(
        patch_size=model_cfg.get("patch_size", 16),
        embed_dim=model_cfg.get("embed_dim", 512),
        num_codes=model_cfg.get("num_codes", 512),
        encoder_layers=model_cfg.get("encoder_layers", 6),
        encoder_heads=model_cfg.get("encoder_heads", 8),
        commitment_cost=model_cfg.get("commitment_cost", 0.25),
    )

    dummy_x = jnp.zeros((2, 256, 256, 3))
    params = model.init(rng, dummy_x, train=False)["params"]

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    logging.info(f"VQ-VAE parameters: {param_count:,}")

    # Log per-component sizes
    for key in params:
        comp_count = sum(x.size for x in jax.tree_util.tree_leaves(params[key]))
        logging.info(f"  {key}: {comp_count:,} params")

    train_cfg = config["training"]
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=train_cfg["lr"],
        warmup_steps=train_cfg["warmup_steps"],
        decay_steps=train_cfg["total_steps"],
        end_value=train_cfg["lr"] * 0.01,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(train_cfg["grad_clip"]),
        optax.adamw(learning_rate=schedule, weight_decay=train_cfg["weight_decay"]),
    )

    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx,
    )


def make_train_step(vq_weight, replicated_sharding, dp_sharding):
    """Create sharded train step for multi-GPU data parallelism."""

    def train_step(
        state: train_state.TrainState,
        batch: jnp.ndarray,
    ) -> Tuple[train_state.TrainState, dict]:
        def loss_fn(params):
            x_recon, indices, vq_loss = state.apply_fn(
                {"params": params}, batch, train=True,
            )
            recon_loss = jnp.mean((x_recon - batch) ** 2)
            total_loss = recon_loss + vq_weight * vq_loss

            # Codebook utilization: how many unique codes used in this batch
            flat_indices = indices.reshape(-1)
            usage_hist = jnp.zeros(512).at[flat_indices].add(1.0)
            num_used = jnp.sum(usage_hist > 0)
            codebook_usage = num_used / 512.0

            mse = jnp.mean((x_recon - batch) ** 2)
            psnr = -10.0 * jnp.log10(mse + 1e-8)

            return total_loss, {
                "recon_loss": recon_loss,
                "vq_loss": vq_loss,
                "total_loss": total_loss,
                "codebook_usage": codebook_usage,
                "num_codes_used": num_used,
                "psnr": psnr,
            }

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)

        metrics = {
            **aux,
            "grad_norm": optax.global_norm(grads),
        }
        return state, metrics

    train_step = jax.jit(
        train_step,
        in_shardings=(replicated_sharding, dp_sharding),
        out_shardings=(replicated_sharding, replicated_sharding),
        donate_argnums=0,
    )
    return train_step


def main(_):
    tf.config.set_visible_devices([], "GPU")

    with open(FLAGS.config, "r") as f:
        config = yaml.safe_load(f)

    # --- System info + multi-GPU mesh ---
    devices = jax.devices()
    num_devices = len(devices)
    logging.info(f"JAX devices: {num_devices} × {devices[0].platform}")
    for i, d in enumerate(devices):
        logging.info(f"  Device {i}: {d}")

    batch_size = config["data"]["batch_size"]
    assert batch_size % num_devices == 0, (
        f"batch_size {batch_size} must be divisible by device_count {num_devices}"
    )
    logging.info(f"Config: batch_size={batch_size} ({batch_size // num_devices}/device), "
                 f"total_steps={config['training']['total_steps']}")

    mesh = Mesh(jax.devices(), axis_names="batch")
    replicated_sharding = NamedSharding(mesh, PartitionSpec())
    dp_sharding = NamedSharding(mesh, PartitionSpec("batch"))

    rng = jax.random.PRNGKey(config["data"]["seed"])

    # --- Load data ---
    logging.info("Loading Bridge V2 dataset...")
    t_data = time.time()
    ds = load_bridge_frames(config)
    logging.info(f"Dataset pipeline built in {time.time() - t_data:.1f}s")

    # Bridge V2 has ~1.15M frames → ~4,492 batches at batch_size=256
    # Skip full dataset scan to avoid idling GPUs
    steps_per_epoch = None

    # --- Initialize model ---
    logging.info("Initializing VQ-VAE model...")
    state = create_train_state(config, rng)

    vq_weight = config["loss"]["vq_weight"]
    total_steps = config["training"]["total_steps"]
    log_every = config["training"]["log_every"]
    save_every = config["training"]["save_every"]
    os.makedirs(config["output"]["checkpoint_dir"], exist_ok=True)

    train_step = make_train_step(vq_weight, replicated_sharding, dp_sharding)

    # --- JIT warmup ---
    logging.info("JIT compiling train_step (first step will be slow)...")
    t_jit = time.time()

    step = 0
    epoch = 0
    t0 = time.time()
    t_last_log = time.time()
    jit_compiled = False

    # Tracking for ETA
    step_times = []  # recent step durations for smoothed ETA

    while step < total_steps:
        epoch += 1
        t_epoch = time.time()

        for batch in ds.as_numpy_iterator():
            if step >= total_steps:
                break

            t_step = time.time()
            batch = jax.device_put(jnp.array(batch), dp_sharding)
            state, metrics = train_step(state, batch)

            # Force sync on first step to measure JIT time
            if not jit_compiled:
                jax.block_until_ready(metrics)
                jit_time = time.time() - t_jit
                logging.info(f"JIT compilation + first step: {jit_time:.1f}s")
                jit_compiled = True
                t0 = time.time()  # Reset timer after JIT

            step += 1
            step_dur = time.time() - t_step
            step_times.append(step_dur)
            if len(step_times) > 100:
                step_times.pop(0)

            if step % log_every == 0:
                metrics = jax.device_get(metrics)
                elapsed = time.time() - t0
                sps = step / elapsed
                avg_step = sum(step_times) / len(step_times)
                remaining_steps = total_steps - step
                eta_seconds = remaining_steps * avg_step
                eta_min = eta_seconds / 60
                eta_hours = eta_seconds / 3600
                pct = 100.0 * step / total_steps

                if eta_hours >= 1:
                    eta_str = f"{eta_hours:.1f}h"
                else:
                    eta_str = f"{eta_min:.1f}min"

                logging.info(
                    f"Step {step:>6d}/{total_steps} ({pct:5.1f}%) | "
                    f"recon={metrics['recon_loss']:.4f} psnr={metrics['psnr']:.1f}dB | "
                    f"vq={metrics['vq_loss']:.4f} | "
                    f"codebook={metrics['num_codes_used']:.0f}/{512} "
                    f"({metrics['codebook_usage']:.1%}) | "
                    f"grad={metrics['grad_norm']:.3f} | "
                    f"{sps:.1f} steps/s | "
                    f"ETA {eta_str}"
                )

            if step % save_every == 0:
                checkpoints.save_checkpoint(
                    config["output"]["checkpoint_dir"], state, step, keep=3,
                )
                logging.info(f"Saved checkpoint at step {step}")

        epoch_time = time.time() - t_epoch
        logging.info(f"Epoch {epoch} complete ({epoch_time:.1f}s, step {step}/{total_steps})")

    # Final save
    if step % save_every != 0:
        checkpoints.save_checkpoint(
            config["output"]["checkpoint_dir"], state, step, keep=3,
        )

    total_time = time.time() - t0
    logging.info(f"Training complete in {total_time/3600:.2f}h ({total_time/60:.1f}min)")
    logging.info(f"Final: recon={metrics['recon_loss']:.4f}, psnr={metrics['psnr']:.1f}dB, "
                 f"codebook_usage={metrics['codebook_usage']:.1%}")


if __name__ == "__main__":
    app.run(main)
