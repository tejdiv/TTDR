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

    # Collect all transition triples
    all_z_t = []
    all_text_embed = []
    all_z_target = []

    traj_count = 0
    for traj in tqdm(dataset.as_numpy_iterator(), desc="Processing trajectories"):
        traj_count += 1
        if FLAGS.max_trajectories > 0 and traj_count > FLAGS.max_trajectories:
            break

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
            obs_t = {}
            obs_target = {}
            for key in traj["observation"]:
                frames_t = traj["observation"][key][batch_indices]
                frames_target = traj["observation"][key][target_indices]
                # Add window dimension: (batch, *) → (batch, 1, *)
                obs_t[key] = frames_t[:, None]
                obs_target[key] = frames_target[:, None]

            obs_t = jax.tree_map(jnp.array, obs_t)
            obs_target = jax.tree_map(jnp.array, obs_target)

            # Tile task for batch
            task_batch = jax.tree_map(lambda x: jnp.tile(x, (bs, *([1] * (x.ndim - 1)))), task)

            pad_mask = jnp.ones((bs, 1), dtype=bool)

            # Encode
            z_t = encode_batch({"observation": obs_t}, task_batch, pad_mask)
            z_target = encode_batch({"observation": obs_target}, task_batch, pad_mask)

            all_z_t.append(np.array(z_t))
            all_z_target.append(np.array(z_target))

            # Extract text embedding from task (same for all frames in this traj)
            # Use the task token output from the transformer as text embedding
            task_outputs = model.run_transformer(
                {"observation": obs_t[:1]},
                jax.tree_map(lambda x: x[:1], task_batch),
                jnp.ones((1, 1), dtype=bool),
            )
            if "task" in task_outputs:
                text_emb = jnp.mean(task_outputs["task"].tokens, axis=(0, 1))
            else:
                # Fallback: use zeros
                text_emb = jnp.zeros(768)
            # Repeat for batch
            text_emb_batch = jnp.tile(text_emb[None], (bs, 1))
            all_text_embed.append(np.array(text_emb_batch))

    # Concatenate and save
    z_t_all = np.concatenate(all_z_t, axis=0)
    text_embed_all = np.concatenate(all_text_embed, axis=0)
    z_target_all = np.concatenate(all_z_target, axis=0)

    output_path = os.path.join(FLAGS.output_dir, "encodings.h5")
    logging.info(f"Saving {z_t_all.shape[0]} transitions to {output_path}...")

    with h5py.File(output_path, "w") as f:
        f.create_dataset("z_t", data=z_t_all, compression="gzip")
        f.create_dataset("text_embed", data=text_embed_all, compression="gzip")
        f.create_dataset("z_target", data=z_target_all, compression="gzip")
        f.attrs["chunk_size"] = m
        f.attrs["encoder_dim"] = z_t_all.shape[1]
        f.attrs["num_transitions"] = z_t_all.shape[0]
        f.attrs["checkpoint"] = FLAGS.checkpoint

    logging.info(f"Done. {z_t_all.shape[0]} transitions, encoder_dim={z_t_all.shape[1]}")


if __name__ == "__main__":
    app.run(main)
