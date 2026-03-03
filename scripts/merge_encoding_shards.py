"""Merge per-shard HDF5 encoding files into a single encodings.h5.

Reads encodings_shard_0.h5 through encodings_shard_{N-1}.h5, concatenates
all datasets (z_t, z_t1, z_target, traj_id), and writes a single encodings.h5.

Optionally uploads the merged file to Hugging Face Hub.

Usage:
    python scripts/merge_encoding_shards.py \
        --input_dir data/bridge_v2_encodings \
        --num_shards 8

    # With HF upload:
    python scripts/merge_encoding_shards.py \
        --input_dir data/bridge_v2_encodings \
        --num_shards 8 \
        --hf_repo tejasrao/ttdr-bridge-encodings
"""

import os

from absl import app, flags, logging
import h5py
import numpy as np

FLAGS = flags.FLAGS
flags.DEFINE_string("input_dir", None, "Directory containing shard HDF5 files.")
flags.DEFINE_integer("num_shards", 8, "Number of shards to merge.")
flags.DEFINE_string("hf_repo", None,
                    "HF Hub repo to upload merged file (e.g. 'tejasrao/ttdr-bridge-encodings'). "
                    "Creates a private dataset repo if it doesn't exist. Requires `huggingface-cli login`.")


def main(_):
    assert FLAGS.input_dir is not None, "Must provide --input_dir"

    dataset_names = ["z_t", "z_t1", "z_target", "traj_id"]
    all_data = {name: [] for name in dataset_names}

    # Accumulate trajectory ID offset so IDs are globally unique
    traj_id_offset = 0

    for shard_id in range(FLAGS.num_shards):
        shard_path = os.path.join(FLAGS.input_dir, f"encodings_shard_{shard_id}.h5")
        if not os.path.exists(shard_path):
            logging.warning(f"Missing shard file: {shard_path}, skipping")
            continue

        logging.info(f"Reading shard {shard_id}: {shard_path}")
        with h5py.File(shard_path, "r") as f:
            for name in dataset_names:
                data = np.array(f[name])
                if name == "traj_id":
                    data = data + traj_id_offset
                all_data[name].append(data)

            num_trajs = f.attrs.get("num_trajectories", 0)
            traj_id_offset += num_trajs

            logging.info(
                f"  Shard {shard_id}: {f.attrs.get('num_transitions', '?')} transitions, "
                f"{num_trajs} trajectories"
            )

    # Concatenate
    merged = {}
    for name in dataset_names:
        if all_data[name]:
            merged[name] = np.concatenate(all_data[name], axis=0)
        else:
            raise RuntimeError(f"No data found for dataset '{name}' across shards")

    total_transitions = merged["z_t"].shape[0]
    logging.info(f"Total: {total_transitions} transitions")

    # Write merged file
    output_path = os.path.join(FLAGS.input_dir, "encodings.h5")
    logging.info(f"Writing merged file to {output_path}")

    with h5py.File(output_path, "w") as f:
        for name in dataset_names:
            f.create_dataset(name, data=merged[name], compression="gzip")

        # Copy metadata from first shard
        first_shard = os.path.join(FLAGS.input_dir, "encodings_shard_0.h5")
        with h5py.File(first_shard, "r") as src:
            f.attrs["chunk_size"] = src.attrs.get("chunk_size", 4)
            f.attrs["window_size"] = src.attrs.get("window_size", 2)
            f.attrs["encoder_dim"] = src.attrs.get("encoder_dim", 768)
            f.attrs["checkpoint"] = src.attrs.get("checkpoint", "unknown")

        f.attrs["num_transitions"] = total_transitions
        f.attrs["num_trajectories"] = traj_id_offset
        f.attrs["num_shards_merged"] = FLAGS.num_shards

    logging.info(f"Done. Merged {FLAGS.num_shards} shards → {output_path}")

    # Upload to HF Hub if requested
    if FLAGS.hf_repo:
        try:
            from huggingface_hub import HfApi
        except ImportError:
            logging.error("huggingface_hub not installed. Run: pip install huggingface_hub")
            return

        logging.info(f"Uploading to HF Hub: {FLAGS.hf_repo}")
        api = HfApi()
        api.create_repo(FLAGS.hf_repo, repo_type="dataset", private=True, exist_ok=True)
        api.upload_file(
            path_or_fileobj=output_path,
            path_in_repo="encodings.h5",
            repo_id=FLAGS.hf_repo,
            repo_type="dataset",
        )
        logging.info(f"Uploaded to https://huggingface.co/datasets/{FLAGS.hf_repo}")


if __name__ == "__main__":
    app.run(main)
