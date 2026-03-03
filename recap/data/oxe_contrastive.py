"""Data loader for contrastive world model training.

Two modes:
1. Raw mode: Loads Bridge V2 trajectories via Octo's OXE dataloader and extracts
   transition triples (o_t, ℓ, o_{t+m}). Used by precompute_encodings.py.
2. Cached mode: Loads pre-computed encoder outputs from HDF5. Used during
   world model training (fast — no image processing).
"""

import os
from collections import defaultdict
from typing import Iterator, Optional, Tuple

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from octo.data.dataset import make_dataset_from_rlds
from octo.data.oxe.oxe_dataset_configs import OXE_DATASET_CONFIGS
from octo.data.oxe.oxe_standardization_transforms import OXE_STANDARDIZATION_TRANSFORMS
from octo.utils.spec import ModuleSpec


def make_bridge_trajectory_dataset(
    data_dir: str,
    train: bool = True,
    skip_unlabeled: bool = True,
    shuffle: Optional[bool] = None,
) -> tf.data.Dataset:
    """Load Bridge V2 trajectories via Octo's OXE dataloader.

    Returns full trajectories (not chunked) for pre-computing encoder outputs.
    Each trajectory has: observation/image_primary, task/language_instruction, action.

    Args:
        data_dir: Path to RLDS data directory containing bridge_dataset.
        train: Whether to use training split.
        skip_unlabeled: Skip trajectories without language annotations.
        shuffle: Whether to shuffle. Defaults to `train` value. Pass False
            explicitly for deterministic ordering (needed for multi-GPU sharding).

    Returns:
        tf.data.Dataset of trajectories.
    """
    if shuffle is None:
        shuffle = train

    config = OXE_DATASET_CONFIGS["bridge_dataset"]
    standardize_fn = ModuleSpec.create(OXE_STANDARDIZATION_TRANSFORMS["bridge_dataset"])

    dataset, dataset_statistics = make_dataset_from_rlds(
        name="bridge_dataset",
        data_dir=data_dir,
        train=train,
        standardize_fn=standardize_fn,
        image_obs_keys=config["image_obs_keys"],
        depth_obs_keys=config.get("depth_obs_keys", {}),
        language_key="language_instruction",
        shuffle=shuffle,
    )

    if skip_unlabeled:
        dataset = dataset.filter(
            lambda x: tf.math.reduce_any(x["task"]["language_instruction"] != "")
        )

    return dataset, dataset_statistics


class CachedContrastiveDataset:
    """Loads pre-computed encoder outputs from HDF5 for world model training.

    Yields batches of (z_t, z_t1, z_target) where:
        z_t:         encoder output at time t,      shape (batch, 768)
        z_t1:        encoder output at time t+1,    shape (batch, 768)
        z_target:    encoder output at time t+m,    shape (batch, 768)
    """

    def __init__(
        self,
        cache_dir: str,
        batch_size: int = 256,
        seed: int = 0,
    ):
        """
        Args:
            cache_dir: Directory containing pre-computed HDF5 files.
            batch_size: Training batch size (also provides negatives for InfoNCE).
            seed: Random seed for shuffling.
        """
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)

        # Load all cached encodings into memory (they're just vectors, not images)
        h5_path = os.path.join(cache_dir, "encodings.h5")
        with h5py.File(h5_path, "r") as f:
            self.z_t = np.array(f["z_t"])              # (N, 768)
            self.z_target = np.array(f["z_target"])     # (N, 768)
            if "z_t1" not in f:
                raise KeyError(
                    "HDF5 file is missing 'z_t1' dataset. "
                    "Re-run precompute_encodings.py with --window_size 2 to generate it."
                )
            self.z_t1 = np.array(f["z_t1"])            # (N, 768)
            if "traj_id" not in f:
                raise KeyError(
                    "HDF5 file is missing 'traj_id' dataset. "
                    "Re-run precompute_encodings.py to generate it."
                )
            self.traj_id = np.array(f["traj_id"])  # (N,) int32

        self.num_samples = self.z_t.shape[0]
        assert self.z_t.shape[0] == self.z_target.shape[0] == self.z_t1.shape[0]

        # Build trajectory index for trajectory-block batching
        self.traj_to_indices = defaultdict(list)
        for i, tid in enumerate(self.traj_id):
            self.traj_to_indices[tid].append(i)
        # Convert to numpy arrays for fast indexing
        self.traj_to_indices = {
            k: np.array(v) for k, v in self.traj_to_indices.items()
        }
        self.traj_ids_list = list(self.traj_to_indices.keys())

    def __len__(self) -> int:
        return self.num_samples // self.batch_size

    def get_iterator(self) -> Iterator[dict]:
        """Yields batches as JAX arrays with trajectory-block batching.

        When traj_id is available, builds batches in trajectory blocks so each
        batch contains multiple transitions from the same trajectories. This
        ensures meaningful same-trajectory negatives for the intra-trajectory
        InfoNCE loss component.

        Strategy: pick K random trajectories, sample S transitions from each
        (K*S ≈ batch_size). Shorter trajectories contribute what they have;
        remaining slots filled from additional random trajectories.
        """
        # Trajectory-block batching
        # Target: ~16 transitions per trajectory in each batch
        samples_per_traj = 16
        num_batches = self.num_samples // self.batch_size

        for _ in range(num_batches):
            batch_idx = []
            remaining = self.batch_size

            # Shuffle trajectory order each batch
            traj_order = self.rng.permutation(self.traj_ids_list)

            for tid in traj_order:
                if remaining <= 0:
                    break
                traj_indices = self.traj_to_indices[tid]
                n_take = min(samples_per_traj, len(traj_indices), remaining)
                chosen = self.rng.choice(traj_indices, size=n_take, replace=False)
                batch_idx.append(chosen)
                remaining -= n_take

            idx = np.concatenate(batch_idx)
            # Shuffle within batch so same-traj transitions aren't contiguous
            self.rng.shuffle(idx)

            yield {
                "z_t": jnp.array(self.z_t[idx]),
                "z_t1": jnp.array(self.z_t1[idx]),
                "z_target": jnp.array(self.z_target[idx]),
                "traj_id": jnp.array(self.traj_id[idx]),
            }
