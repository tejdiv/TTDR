"""Pre-compute Octo encoder outputs for all Bridge V2 trajectories.

Runs the frozen Octo encoder (SmallStem16 + transformer backbone) over all
frames in Bridge V2 using window_size=2 (two-frame temporal attention),
caches the readout token representations to HDF5.

Supports multi-GPU sharding: run N copies with --shard_id 0..N-1 and
--num_shards N, then merge with merge_encoding_shards.py.

Output HDF5 schema per shard:
    /z_t       (N, 768)  float32  — readout at frame t
    /z_t1      (N, 768)  float32  — readout at frame t+1
    /z_target  (N, 768)  float32  — readout at frame t+m+1
    /traj_id   (N,)      int32

Usage:
    python scripts/precompute_encodings.py \
        --data_dir /path/to/rlds/data \
        --output_dir /path/to/cache \
        --chunk_size 4 --window_size 2 \
        --shard_id 0 --num_shards 8
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
flags.DEFINE_string("checkpoint", "hf://rail-berkeley/octo-base",
                    "Octo checkpoint to load.")
flags.DEFINE_integer("chunk_size", 4, "Action chunk size m (frames between z_t and z_{t+m}).")
flags.DEFINE_integer("batch_size", 64, "Batch size for encoder forward pass.")
flags.DEFINE_integer("max_trajectories", 0, "Max trajectories to process (0 = all).")
flags.DEFINE_integer("window_size", 2, "Octo observation window size.")
flags.DEFINE_integer("shard_id", 0, "This worker's shard index (0-based).")
flags.DEFINE_integer("num_shards", 1, "Total number of parallel workers.")


def decode_all_frames(image_array):
    """Decode all JPEG byte strings in a trajectory upfront.

    Args:
        image_array: Array of JPEG byte strings, shape (T,) with dtype=object.

    Returns:
        np.ndarray of decoded images, shape (T, H, W, 3), dtype=uint8.
        Returns None if any frame fails to decode.
    """
    imgs = []
    for b in image_array:
        if isinstance(b, (bytes, np.bytes_)) and len(b) > 0:
            img = tf.io.decode_image(
                b, channels=3,
                expand_animations=False,
                dtype=tf.uint8,
            )
            imgs.append(img.numpy())
        else:
            return None
    return np.stack(imgs)


def extract_readout_features(model, obs_batch, task_batch, pad_mask):
    """Run Octo's transformer and extract readout_action tokens at both positions.

    For window_size=2 with causal attention:
      - Position 0: read[t-1]      — 1-frame context (only saw frame t-1)
      - Position 1: read[t, t-1]   — 2-frame context (saw both frames)

    Args:
        model: Loaded OctoModel.
        obs_batch: Dict of observation arrays, shape (batch, window_size, *).
        task_batch: Dict of task arrays, shape (batch, *).
        pad_mask: (batch, window_size) boolean mask.

    Returns:
        z_pos0: Position 0 readout (1-frame), shape (batch, 768).
        z_pos1: Position 1 readout (2-frame), shape (batch, 768).
    """
    transformer_outputs = model.run_transformer(obs_batch, task_batch, pad_mask)

    # readout_action shape: (batch, window_size, n_readout_tokens, 768)
    tokens = transformer_outputs["readout_action"].tokens

    # Position 0: read[t-1] — only attended to first frame
    z_pos0 = jnp.mean(tokens[:, 0, :, :], axis=1)  # (batch, 768)

    # Position 1: read[t, t-1] — attended to both frames
    z_pos1 = jnp.mean(tokens[:, -1, :, :], axis=1)  # (batch, 768)

    return z_pos0, z_pos1


def main(_):
    assert FLAGS.data_dir is not None, "Must provide --data_dir"
    assert FLAGS.output_dir is not None, "Must provide --output_dir"
    assert 0 <= FLAGS.shard_id < FLAGS.num_shards, (
        f"shard_id {FLAGS.shard_id} must be in [0, {FLAGS.num_shards})"
    )

    os.makedirs(FLAGS.output_dir, exist_ok=True)
    m = FLAGS.chunk_size
    ws = FLAGS.window_size

    # Load Octo model
    logging.info(f"Loading Octo from {FLAGS.checkpoint}...")
    model = OctoModel.load_pretrained(FLAGS.checkpoint)

    # Load Bridge V2 dataset (deterministic order for sharding)
    logging.info(f"Loading Bridge V2 from {FLAGS.data_dir}...")
    dataset, stats = make_bridge_trajectory_dataset(
        FLAGS.data_dir, train=True, skip_unlabeled=True, shuffle=False
    )

    # Shard the dataset across workers
    if FLAGS.num_shards > 1:
        dataset = dataset.shard(FLAGS.num_shards, FLAGS.shard_id)
        logging.info(f"Shard {FLAGS.shard_id}/{FLAGS.num_shards}")

    # JIT the encoder — returns (z_pos0, z_pos1) per window
    @jax.jit
    def encode_batch(obs, task, pad_mask):
        return extract_readout_features(model, obs, task, pad_mask)

    # Output file — per-shard or single
    if FLAGS.num_shards > 1:
        output_path = os.path.join(FLAGS.output_dir, f"encodings_shard_{FLAGS.shard_id}.h5")
    else:
        output_path = os.path.join(FLAGS.output_dir, "encodings.h5")

    logging.info(f"Writing to {output_path}")
    encoder_dim = 768  # Octo-Base

    with h5py.File(output_path, "w") as f:
        # Create resizable datasets for streaming writes
        z_t_ds = f.create_dataset(
            "z_t", shape=(0, encoder_dim), maxshape=(None, encoder_dim),
            dtype="float32", chunks=(1024, encoder_dim),
        )
        z_t1_ds = f.create_dataset(
            "z_t1", shape=(0, encoder_dim), maxshape=(None, encoder_dim),
            dtype="float32", chunks=(1024, encoder_dim),
        )
        z_target_ds = f.create_dataset(
            "z_target", shape=(0, encoder_dim), maxshape=(None, encoder_dim),
            dtype="float32", chunks=(1024, encoder_dim),
        )
        traj_id_ds = f.create_dataset(
            "traj_id", shape=(0,), maxshape=(None,),
            dtype="int32", chunks=(4096,),
        )

        traj_count = 0
        total_transitions = 0

        for traj in tqdm(dataset.as_numpy_iterator(), desc=f"Shard {FLAGS.shard_id}"):
            if FLAGS.max_trajectories > 0 and traj_count >= FLAGS.max_trajectories:
                break
            traj_count += 1

            # Debug: print keys on first trajectory
            if traj_count == 1:
                logging.info(f"Trajectory keys: {list(traj.keys())}")
                logging.info(f"Observation keys: {list(traj['observation'].keys())}")
                for k, v in traj["observation"].items():
                    logging.info(f"  obs/{k}: shape={v.shape}, dtype={v.dtype}")

            traj_len = traj["action"].shape[0]

            # Need at least window_size frames + m frames for one valid triple
            if traj_len < ws + m:
                continue

            # Extract language instruction (same for all steps in trajectory)
            lang = traj["task"]["language_instruction"]
            if isinstance(lang, bytes):
                lang = lang.decode("utf-8")
            elif isinstance(lang, np.ndarray):
                lang = str(lang.flat[0])
                if isinstance(lang, bytes):
                    lang = lang.decode("utf-8")

            # Create task embedding once for this trajectory
            task = model.create_tasks(texts=[lang])

            # Decode ALL frames for this trajectory upfront (avoid redundant JPEG decoding)
            IMAGE_KEYS = {"image_primary", "image_secondary", "image_wrist"}
            decoded_images = {}
            skip_traj = False
            for key in traj["observation"]:
                if key in IMAGE_KEYS and traj["observation"][key].dtype == object:
                    # Skip empty camera streams (e.g., debug datasets without secondary/wrist)
                    first_frame = traj["observation"][key][0]
                    if isinstance(first_frame, (bytes, np.bytes_)) and len(first_frame) == 0:
                        continue
                    decoded = decode_all_frames(traj["observation"][key])
                    if decoded is None:
                        skip_traj = True
                        break
                    decoded_images[key] = decoded
            if skip_traj:
                continue
            if not decoded_images:
                logging.warning(f"Trajectory {traj_count}: no valid image keys, skipping")
                continue

            # Resize images to match Octo's expected resolution (256x256)
            for key in decoded_images:
                imgs = decoded_images[key]
                if imgs.shape[1] != 256 or imgs.shape[2] != 256:
                    resized = tf.image.resize(imgs, [256, 256]).numpy().astype(np.uint8)
                    decoded_images[key] = resized

            # Build all windows and encode all frames
            # For window_size=2: T-1 overlapping windows → T-1 readouts
            # window[i] = (frame[i], frame[i+1]) → readout[i] = enc(frame[i+1])
            num_windows = traj_len - (ws - 1)  # T - 1 for ws=2

            # Encode all windows in batches
            # Each window (frame_i, frame_{i+1}) produces two readouts:
            #   pos0 = read[i]       (1-frame context)
            #   pos1 = read[i+1, i]  (2-frame context)
            all_pos0 = []
            all_pos1 = []
            for batch_start in range(0, num_windows, FLAGS.batch_size):
                batch_end = min(batch_start + FLAGS.batch_size, num_windows)
                actual_bs = batch_end - batch_start

                # Build observation batch: (batch, window_size, *)
                obs_batch = {}
                for key in traj["observation"]:
                    if key in decoded_images:
                        # Stack windows: each window is ws consecutive frames
                        frames = []
                        for i in range(batch_start, batch_end):
                            window = decoded_images[key][i:i + ws]  # (ws, H, W, 3)
                            frames.append(window)
                        obs_batch[key] = np.stack(frames)  # (batch, ws, H, W, 3)
                    elif traj["observation"][key].dtype == object or \
                         traj["observation"][key].dtype.kind in ('U', 'S', 'O'):
                        continue
                    else:
                        # Non-image keys: stack windows of ws consecutive values
                        frames = []
                        for i in range(batch_start, batch_end):
                            window = traj["observation"][key][i:i + ws]
                            frames.append(window)
                        obs_batch[key] = np.stack(frames)  # (batch, ws, *)

                if not any(k.startswith("image_") for k in obs_batch):
                    logging.warning("No image keys in obs_batch — skipping batch")
                    continue

                # Pad last batch to avoid JIT recompilation
                pad_size = FLAGS.batch_size - actual_bs
                if pad_size > 0:
                    obs_batch = {
                        k: np.concatenate([v, np.zeros((pad_size, *v.shape[1:]), dtype=v.dtype)])
                        for k, v in obs_batch.items()
                    }

                # Add pad_mask (v1.0 uses "pad_mask", not "timestep_pad_mask")
                pad_mask = np.ones((FLAGS.batch_size, ws), dtype=bool)
                obs_batch["pad_mask"] = pad_mask

                obs_batch = jax.tree.map(jnp.array, obs_batch)

                # Tile task for batch
                task_batch = jax.tree.map(
                    lambda x: jnp.tile(x, (FLAGS.batch_size, *([1] * (x.ndim - 1)))),
                    task,
                )
                pad_mask_jnp = jnp.array(pad_mask)

                # Encode — get both readout positions
                z_pos0, z_pos1 = encode_batch(obs_batch, task_batch, pad_mask_jnp)

                # Slice off padding
                all_pos0.append(np.array(z_pos0)[:actual_bs])
                all_pos1.append(np.array(z_pos1)[:actual_bs])

            if not all_pos0:
                continue

            pos0 = np.concatenate(all_pos0, axis=0)  # (num_windows, 768)
            pos1 = np.concatenate(all_pos1, axis=0)  # (num_windows, 768)
            assert pos0.shape[0] == num_windows

            # Construct triples from same-window readout pairs:
            #   Window i = (frame_i, frame_{i+1}) produces:
            #     pos0[i] = read[i]       (1-frame)
            #     pos1[i] = read[i+1, i]  (2-frame)
            #
            # Prediction: read[t-1], read[t, t-1] → read[t+m, t+m-1]
            #   z_t     = pos0[i]   = read[i]           (1-frame)
            #   z_t1    = pos1[i]   = read[i+1, i]      (2-frame)
            #   z_target = pos1[i+m] = read[i+m+1, i+m]  (2-frame)
            # Need i+m <= num_windows-1
            K = num_windows - m
            if K <= 0:
                continue

            z_t = pos0[0:K]           # (K, 768) — read[0..K-1]
            z_t1 = pos1[0:K]          # (K, 768) — read[1,0] .. read[K,K-1]
            z_target = pos1[m:K + m]  # (K, 768) — read[m+1,m] .. read[K+m,K+m-1]

            # Stream to HDF5
            old_size = z_t_ds.shape[0]
            new_size = old_size + K
            z_t_ds.resize(new_size, axis=0)
            z_t1_ds.resize(new_size, axis=0)
            z_target_ds.resize(new_size, axis=0)
            traj_id_ds.resize(new_size, axis=0)

            z_t_ds[old_size:new_size] = z_t
            z_t1_ds[old_size:new_size] = z_t1
            z_target_ds[old_size:new_size] = z_target
            traj_id_ds[old_size:new_size] = np.full(K, traj_count, dtype=np.int32)
            total_transitions += K

        # Write metadata
        f.attrs["chunk_size"] = m
        f.attrs["window_size"] = ws
        f.attrs["encoder_dim"] = encoder_dim
        f.attrs["num_transitions"] = total_transitions
        f.attrs["num_trajectories"] = traj_count
        f.attrs["checkpoint"] = FLAGS.checkpoint
        f.attrs["shard_id"] = FLAGS.shard_id
        f.attrs["num_shards"] = FLAGS.num_shards

    logging.info(
        f"Done. {total_transitions} transitions from {traj_count} trajectories, "
        f"encoder_dim={encoder_dim}, shard={FLAGS.shard_id}/{FLAGS.num_shards}"
    )


if __name__ == "__main__":
    app.run(main)
