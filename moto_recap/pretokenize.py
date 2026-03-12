"""Pre-tokenize Bridge V2 into motion tokens on disk.

Parallelizes data loading with tf.data and GPU tokenization.
Saves tokens + metadata to a single .npz file for instant loading.

Usage:
    python -m moto_recap.pretokenize --data_dir /home/ubuntu/data/rlds \
        --output /home/ubuntu/data/bridge_motion_tokens.npz
"""

import os
import numpy as np
import torch
from absl import app, flags, logging
from concurrent.futures import ThreadPoolExecutor

import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

from moto_recap.tokenizer import load_tokenizer, tokenize_frames

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", "/home/ubuntu/data/rlds", "RLDS data dir")
flags.DEFINE_string("output", "/home/ubuntu/data/bridge_motion_tokens.npz", "Output path")
flags.DEFINE_integer("frame_skip", 3, "Frame skip m")
flags.DEFINE_integer("batch_size", 256, "GPU batch size for tokenization")
flags.DEFINE_integer("max_traj", 0, "Max trajectories (0=all)")

QUEUE_SIZE = 512  # frame pairs to buffer before tokenizing


def main(_):
    device = "cuda"
    logging.info("Loading tokenizer...")
    tokenizer = load_tokenizer("TencentARC/Moto", device=device)
    m = FLAGS.frame_skip

    # Load dataset with parallel reading
    import tensorflow_datasets as tfds
    bridge_dir = os.path.join(FLAGS.data_dir, "bridge", "0.1.0")
    builder = tfds.builder_from_directory(bridge_dir)
    read_config = tfds.ReadConfig(
        interleave_cycle_length=16,
        interleave_block_length=1,
        num_parallel_calls_for_interleave_files=tf.data.AUTOTUNE,
        num_parallel_calls_for_decode=tf.data.AUTOTUNE,
    )
    ds = builder.as_dataset(split="train", read_config=read_config)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    # Pre-allocate lists for results
    all_tokens = []
    all_instructions = []
    all_traj_ids = []
    all_first_frames = []
    traj_pair_counts = []

    # Buffer for batching
    buf_ft = []
    buf_fm = []
    buf_traj_id = []

    def flush_buffer():
        """Tokenize buffered pairs on GPU."""
        if not buf_ft:
            return
        ft = np.stack(buf_ft)
        fm = np.stack(buf_fm)
        tokens = tokenize_frames(tokenizer, ft, fm, device)
        all_tokens.append(tokens)
        buf_ft.clear()
        buf_fm.clear()

    traj_count = 0
    pair_count = 0
    for episode in ds:
        instruction = None
        frames = []

        for step in episode["steps"]:
            obs = step["observation"]
            img = obs["image"].numpy()
            frames.append(img)
            if instruction is None:
                instr = obs["natural_language_instruction"].numpy()
                if isinstance(instr, bytes):
                    instr = instr.decode("utf-8")
                if instr:
                    instruction = instr

        if not instruction or len(frames) <= m:
            continue

        frames = np.stack(frames)
        first_frame = frames[0].copy()  # .copy() breaks numpy view → frees frames

        n_pairs = 0
        for i in range(0, len(frames) - m, m):
            buf_ft.append(frames[i].astype(np.float32) / 255.0)
            buf_fm.append(frames[i + m].astype(np.float32) / 255.0)
            n_pairs += 1

            if len(buf_ft) >= FLAGS.batch_size:
                flush_buffer()

        del frames  # explicitly free trajectory frames

        if n_pairs > 0:
            all_instructions.append(instruction)
            all_first_frames.append(first_frame)
            traj_pair_counts.append(n_pairs)
            pair_count += n_pairs
            traj_count += 1

        if traj_count % 1000 == 0 and traj_count > 0:
            logging.info(f"  {traj_count} trajectories, {pair_count} pairs")

        if FLAGS.max_traj > 0 and traj_count >= FLAGS.max_traj:
            break

    # Flush remaining
    flush_buffer()

    logging.info(f"Total: {traj_count} trajectories, {pair_count} pairs")

    # Concatenate all tokens
    tokens_all = np.concatenate(all_tokens, axis=0)  # (total_pairs, 8)
    pair_counts = np.array(traj_pair_counts, dtype=np.int32)

    # Save first frames as uint8 downsampled to 224x224 to save space
    import torchvision.transforms.functional as TF
    from PIL import Image
    first_frames_small = []
    for ff in all_first_frames:
        img = Image.fromarray(ff)
        img = img.resize((224, 224), Image.BILINEAR)
        first_frames_small.append(np.array(img))
    first_frames_arr = np.stack(first_frames_small)  # (N_traj, 224, 224, 3) uint8

    logging.info(f"Saving to {FLAGS.output}...")
    np.savez_compressed(
        FLAGS.output,
        tokens=tokens_all,
        pair_counts=pair_counts,
        instructions=np.array(all_instructions, dtype=object),
        first_frames=first_frames_arr,
    )
    size_mb = os.path.getsize(FLAGS.output) / 1e6
    logging.info(f"Saved: {size_mb:.1f} MB")


if __name__ == "__main__":
    app.run(main)
