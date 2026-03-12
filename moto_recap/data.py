"""Bridge V2 data loading for Moto-RECAP.

Handles the GCS RLDS format (bridge/0.1.0) which has structured actions
{world_vector, rotation_delta, open_gripper} and observations {image, state}.

For Moto fine-tuning we only need frames + language instructions (no actions).
"""

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def load_bridge_trajectories(data_dir, train=True, max_trajectories=0):
    """Load Bridge V2 trajectories from GCS-format RLDS data.

    Supports both formats:
    - GCS format: bridge/0.1.0 with structured actions
    - Berkeley format: bridge_dataset/1.0.0 with flat actions

    Args:
        data_dir: Path to RLDS data dir (contains bridge/ or bridge_dataset/).
        train: Use train split.
        max_trajectories: 0 = all.

    Yields:
        dict with "frames" (T, H, W, 3) uint8 and "instruction" str.
    """
    import os

    # Try GCS format first (bridge/0.1.0), then Berkeley format
    bridge_dir = os.path.join(data_dir, "bridge", "0.1.0")
    if os.path.isdir(bridge_dir):
        builder = tfds.builder_from_directory(bridge_dir)
    else:
        # Try bridge_dataset/1.0.0
        bd_dir = os.path.join(data_dir, "bridge_dataset", "1.0.0")
        if os.path.isdir(bd_dir):
            builder = tfds.builder_from_directory(bd_dir)
        else:
            raise FileNotFoundError(
                f"No bridge data found in {data_dir}. "
                f"Expected bridge/0.1.0/ or bridge_dataset/1.0.0/"
            )

    split = "train" if train else "test"
    ds = builder.as_dataset(split=split)

    count = 0
    for episode in ds:
        frames = []
        instruction = None

        for step in episode["steps"]:
            obs = step["observation"]
            # GCS format: 'image' key, Berkeley: 'image_primary' or similar
            if "image" in obs:
                img = obs["image"].numpy()  # (H, W, 3) uint8
            elif "image_primary" in obs:
                img = obs["image_primary"].numpy()
            else:
                continue
            frames.append(img)

            if instruction is None:
                if "natural_language_instruction" in obs:
                    instr = obs["natural_language_instruction"].numpy()
                elif "language_instruction" in obs:
                    instr = obs["language_instruction"].numpy()
                else:
                    instr = b""
                if isinstance(instr, bytes):
                    instr = instr.decode("utf-8")
                if instr:
                    instruction = instr

        if len(frames) < 2 or not instruction:
            continue

        yield {
            "frames": np.stack(frames),  # (T, H, W, 3) uint8
            "instruction": instruction,
        }

        count += 1
        if max_trajectories > 0 and count >= max_trajectories:
            break
