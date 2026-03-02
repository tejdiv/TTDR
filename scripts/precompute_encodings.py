"""Pre-compute Octo encoder outputs for all Bridge V2 trajectories.

Runs the frozen Octo encoder (SmallStem16 + transformer backbone) over all
frames in Bridge V2, caches the readout token representations z_t = φ(o_t)
and language embeddings to HDF5.

This is a one-time cost that turns world model training from
"GPU-bound on ViT forward passes" to "GPU-bound on small MLP/transformer
operating on cached 768-dim vectors" — 10-50x faster.

Usage:
    python scripts/precompute_encodings.py \
        --data_dir /path/to/rlds/data \
        --output_dir /path/to/cache \
        --chunk_size 4
"""

import os
from functools import partial

from absl import app, flags, logging
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
# Prevent TF from grabbing GPU memory — JAX needs it
tf.config.set_visible_devices([], "GPU")
from tqdm import tqdm

from octo.model.octo_model import OctoModel
from recap.data.oxe_contrastive import make_bridge_trajectory_dataset

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", None, "Path to RLDS data directory.")
flags.DEFINE_string("output_dir", None, "Path to output HDF5 cache directory.")
flags.DEFINE_string("checkpoint", "hf://rail-berkeley/octo-base-1.5",
                    "Octo checkpoint to load.")
flags.DEFINE_integer("chunk_size", 4, "Action chunk size m (frames between z_t and z_{t+m}).")
flags.DEFINE_integer("batch_size", 32, "Batch size for encoder forward pass.")
flags.DEFINE_integer("max_trajectories", 0, "Max trajectories to process (0 = all).")


def extract_readout_features(model, obs_batch, task_batch, pad_mask):
    """Run Octo's transformer and extract readout_action tokens.

    Args:
        model: Loaded OctoModel.
        obs_batch: Dict of observation arrays, shape (batch, window=1, *).
        task_batch: Dict of task arrays, shape (batch, *).
        pad_mask: (batch, window=1) boolean mask.

    Returns:
        z: Pooled readout tokens, shape (batch, 768).
    """
    transformer_outputs = model.run_transformer(obs_batch, task_batch, pad_mask)

    # readout_action contains the action readout tokens
    # shape: (batch, window_size=1, n_readout_tokens, 768)
    readout = transformer_outputs["readout_action"]
    tokens = readout.tokens

    # Mean pool over readout tokens and window dimension
    # (batch, 1, n_tokens, 768) → (batch, 768)
    z = jnp.mean(tokens, axis=(1, 2))
    return z


def main(_):
    assert FLAGS.data_dir is not None, "Must provide --data_dir"
    assert FLAGS.output_dir is not None, "Must provide --output_dir"

    os.makedirs(FLAGS.output_dir, exist_ok=True)
    m = FLAGS.chunk_size

    # Load Octo model
    logging.info(f"Loading Octo from {FLAGS.checkpoint}...")
    model = OctoModel.load_pretrained(FLAGS.checkpoint)

    # Load Bridge V2 dataset
    logging.info(f"Loading Bridge V2 from {FLAGS.data_dir}...")
    dataset, stats = make_bridge_trajectory_dataset(
        FLAGS.data_dir, train=True, skip_unlabeled=True
    )

    # JIT the encoder
    @jax.jit
    def encode_batch(obs, task, pad_mask):
        return extract_readout_features(model, obs, task, pad_mask)

    # Collect all transition pairs
    all_z_t = []
    all_z_target = []

    traj_count = 0
    for traj in tqdm(dataset.as_numpy_iterator(), desc="Processing trajectories"):
        traj_count += 1
        if FLAGS.max_trajectories > 0 and traj_count > FLAGS.max_trajectories:
            break

        # Debug: print keys on first trajectory
        if traj_count == 1:
            logging.info(f"Trajectory keys: {list(traj.keys())}")
            logging.info(f"Observation keys: {list(traj['observation'].keys())}")
            for k, v in traj["observation"].items():
                logging.info(f"  obs/{k}: shape={v.shape}, dtype={v.dtype}")
                if v.dtype == object and len(v) > 0:
                    sample = v[0]
                    logging.info(f"    sample type={type(sample)}, "
                                 f"len={len(sample) if hasattr(sample, '__len__') else 'N/A'}, "
                                 f"repr={repr(sample[:30]) if isinstance(sample, (bytes, np.bytes_)) and len(sample) > 0 else repr(sample)}")

        # traj has shape (traj_len, *) for each key
        traj_len = traj["action"].shape[0]

        if traj_len <= m:
            continue

        # Extract language instruction for this trajectory
        # task/language_instruction is per-trajectory (same for all steps)
        lang = traj["task"]["language_instruction"]
        if isinstance(lang, bytes):
            lang = lang.decode("utf-8")
        elif isinstance(lang, np.ndarray):
            lang = str(lang.flat[0])
            if isinstance(lang, bytes):
                lang = lang.decode("utf-8")

        # Create task embedding once for this trajectory
        task = model.create_tasks(texts=[lang])

        # Process frames in batches
        valid_indices = list(range(traj_len - m))

        for batch_start in range(0, len(valid_indices), FLAGS.batch_size):
            batch_indices = valid_indices[batch_start : batch_start + FLAGS.batch_size]
            target_indices = [i + m for i in batch_indices]
            bs = len(batch_indices)

            # Build observation batch for time t: (batch, window=1, *)
            IMAGE_KEYS = {"image_primary", "image_secondary", "image_wrist"}
            obs_t = {}
            obs_target = {}
            for key in traj["observation"]:
                vals_t = traj["observation"][key][batch_indices]
                vals_target = traj["observation"][key][target_indices]

                if key in IMAGE_KEYS and vals_t.dtype == object:
                    # Decode JPEG byte strings into uint8 arrays
                    # (Bridge V2 RLDS stores images as JPEG bytes via tfds.features.Image)
                    def decode_images(byte_array):
                        imgs = []
                        for b in byte_array:
                            if isinstance(b, (bytes, np.bytes_)) and len(b) > 0:
                                img = tf.io.decode_image(
                                    b, channels=3,
                                    expand_animations=False,
                                    dtype=tf.uint8,
                                )
                                imgs.append(img.numpy())
                            else:
                                return None  # padding — skip this key
                        return np.stack(imgs)

                    decoded_t = decode_images(vals_t)
                    decoded_target = decode_images(vals_target)
                    if decoded_t is None or decoded_target is None:
                        continue
                    # Add window dim: (batch, H, W, C) → (batch, 1, H, W, C)
                    obs_t[key] = decoded_t[:, None]
                    obs_target[key] = decoded_target[:, None]
                elif vals_t.dtype == object or vals_t.dtype.kind in ('U', 'S', 'O'):
                    continue
                else:
                    obs_t[key] = vals_t[:, None]
                    obs_target[key] = vals_target[:, None]

            # Debug: log what made it into obs dict
            if traj_count == 1 and batch_start == 0:
                logging.info(f"obs_t keys after decode: {list(obs_t.keys())}")
                for k, v in obs_t.items():
                    logging.info(f"  {k}: shape={v.shape}, dtype={v.dtype}")

            if not any(k.startswith("image_") for k in obs_t):
                logging.warning("No image keys in obs_t — skipping batch")
                continue

            # Add timestep_pad_mask (all valid)
            obs_t["timestep_pad_mask"] = np.ones((bs, 1), dtype=bool)
            obs_target["timestep_pad_mask"] = np.ones((bs, 1), dtype=bool)

            obs_t = jax.tree.map(jnp.array, obs_t)
            obs_target = jax.tree.map(jnp.array, obs_target)

            # Tile task for batch
            task_batch = jax.tree.map(lambda x: jnp.tile(x, (bs, *([1] * (x.ndim - 1)))), task)

            pad_mask = jnp.ones((bs, 1), dtype=bool)

            # Encode — pass obs dict directly (not wrapped in {"observation": ...})
            z_t = encode_batch(obs_t, task_batch, pad_mask)
            z_target = encode_batch(obs_target, task_batch, pad_mask)

            all_z_t.append(np.array(z_t))
            all_z_target.append(np.array(z_target))

    # Concatenate and save
    z_t_all = np.concatenate(all_z_t, axis=0)
    z_target_all = np.concatenate(all_z_target, axis=0)

    output_path = os.path.join(FLAGS.output_dir, "encodings.h5")
    logging.info(f"Saving {z_t_all.shape[0]} transitions to {output_path}...")

    with h5py.File(output_path, "w") as f:
        f.create_dataset("z_t", data=z_t_all, compression="gzip")
        f.create_dataset("z_target", data=z_target_all, compression="gzip")
        f.attrs["chunk_size"] = m
        f.attrs["encoder_dim"] = z_t_all.shape[1]
        f.attrs["num_transitions"] = z_t_all.shape[0]
        f.attrs["checkpoint"] = FLAGS.checkpoint

    logging.info(f"Done. {z_t_all.shape[0]} transitions, encoder_dim={z_t_all.shape[1]}")


if __name__ == "__main__":
    app.run(main)
