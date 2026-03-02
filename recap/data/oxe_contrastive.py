"""Data loader for contrastive world model training.

Two modes:
1. Raw mode: Loads Bridge V2 trajectories via Octo's OXE dataloader and extracts
   transition triples (o_t, ℓ, o_{t+m}). Used by precompute_encodings.py.
2. Cached mode: Loads pre-computed encoder outputs from HDF5. Used during
   world model training (fast — no image processing).
"""

import os
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
) -> tf.data.Dataset:
    """Load Bridge V2 trajectories via Octo's OXE dataloader.

    Returns full trajectories (not chunked) for pre-computing encoder outputs.
    Each trajectory has: observation/image_primary, task/language_instruction, action.

    Args:
        data_dir: Path to RLDS data directory containing bridge_dataset.
        train: Whether to use training split.
        skip_unlabeled: Skip trajectories without language annotations.

    Returns:
        tf.data.Dataset of trajectories.
    """
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
        shuffle=train,
    )

    if skip_unlabeled:
        dataset = dataset.filter(
            lambda x: tf.math.reduce_any(x["task"]["language_instruction"] != "")
        )

    return dataset, dataset_statistics


class CachedContrastiveDataset:
    """Loads pre-computed encoder outputs from HDF5 for world model training.

    Yields batches of (z_t, z_target) where:
        z_t:         encoder output at time t,      shape (batch, 768)
        z_target:    encoder output at time t+m,     shape (batch, 768)
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

        self.num_samples = self.z_t.shape[0]
        assert self.z_t.shape[0] == self.z_target.shape[0]

    def __len__(self) -> int:
        return self.num_samples // self.batch_size

    def get_iterator(self) -> Iterator[dict]:
        """Yields shuffled batches as JAX arrays."""
        indices = np.arange(self.num_samples)
        self.rng.shuffle(indices)

        for start in range(0, self.num_samples - self.batch_size + 1, self.batch_size):
            idx = indices[start : start + self.batch_size]
            yield {
                "z_t": jnp.array(self.z_t[idx]),
                "z_target": jnp.array(self.z_target[idx]),
            }
