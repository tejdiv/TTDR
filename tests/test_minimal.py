"""Minimal test: replicate SimplerEnv's OctoInference exactly with raw model calls.

No OctoEnvWrapper. Direct ManiSkill env. Same image processing as SimplerEnv.
Same RNG seeding. Same action processing.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp

_x = jnp.ones((1, 8, 8, 3))
_k = jnp.ones((3, 3, 3, 16))
_ = jax.lax.conv_general_dilated(
    _x, _k, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")
)
del _x, _k

import numpy as np
import torch
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

from collections import deque
from functools import partial

import gymnasium as gym
import mani_skill.envs  # noqa
from octo.model.octo_model import OctoModel
from mani_skill.utils.geometry import rotation_conversions


IMAGE_SIZE = 256
HORIZON = 2


def resize_image(image):
    """(B, H, W, 3) float -> (B, 256, 256, 3) uint8"""
    image = jax.vmap(
        partial(jax.image.resize,
                shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                method="lanczos3", antialias=True)
    )(image)
    return jnp.clip(jnp.round(image), 0, 255).astype(jnp.uint8)


def main():
    model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base-1.5")
    action_mean = jnp.array(model.dataset_statistics["bridge_dataset"]["action"]["mean"])
    action_std = jnp.array(model.dataset_statistics["bridge_dataset"]["action"]["std"])

    env = gym.make("PutCarrotOnPlateInScene-v1", obs_mode="rgb+segmentation")

    # Match SimplerEnv's RNG: split 5 times before use
    rng = jax.random.PRNGKey(0)
    for _ in range(5):
        rng, _ = jax.random.split(rng)

    successes = 0
    num_eps = 20

    for ep in range(num_eps):
        raw_obs, _ = env.reset(seed=ep, options={"episode_id": torch.tensor([ep])})
        instruction = env.unwrapped.get_language_instruction()
        if isinstance(instruction, list):
            instruction = instruction[0]

        task = model.create_tasks(texts=[instruction])

        # Image history (deque, max HORIZON=2, matching SimplerEnv)
        image_history = deque(maxlen=HORIZON)
        num_image_history = 0

        # Action ensemble (matching SimplerEnv exactly)
        action_history = deque(maxlen=4)

        done = False
        steps = 0

        while not done:
            # Extract image from obs
            image = raw_obs["sensor_data"]["3rd_view_camera"]["rgb"]
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
            image = image.astype(np.float32)

            # Resize
            image = resize_image(image)

            # Add to history
            image_history.append(image)
            num_image_history = min(num_image_history + 1, HORIZON)

            # Build observation dict (matching SimplerEnv exactly)
            images = jnp.stack(list(image_history), axis=1)  # (B, T, H, W, 3)
            B = images.shape[0]
            T = len(image_history)
            pad_mask = jnp.ones((B, T), dtype=jnp.float32)
            pad_mask = pad_mask.at[:, :T - min(T, num_image_history)].set(0)

            input_obs = {"image_primary": images, "pad_mask": pad_mask}

            # Sample actions
            rng, key = jax.random.split(rng)
            norm_actions = model.sample_actions(input_obs, task, rng=key)

            # Denormalize
            raw_actions = norm_actions * action_std[None] + action_mean[None]
            raw_actions_np = np.array(raw_actions)  # (B, pred_horizon, 7)

            # Action ensemble (matching SimplerEnv: operates on (B, pred_horizon, 7))
            action_history.append(raw_actions_np)
            n = len(action_history)
            curr_act_preds = jnp.stack(
                [pred[:, i] for (i, pred) in zip(range(n - 1, -1, -1), action_history)]
            )  # (n, B, 7)
            weights = jnp.exp(-0.0 * jnp.arange(n))
            weights = weights / weights.sum()
            ensembled = jnp.sum(weights[:, None, None] * curr_act_preds, axis=0)  # (B, 7)

            # Post-process: euler -> axis-angle (matching SimplerEnv)
            euler = torch.tensor(np.array(ensembled[:, 3:6]), dtype=torch.float32)
            mat = rotation_conversions.euler_angles_to_matrix(euler, "XYZ")
            rot_axangle = rotation_conversions.matrix_to_axis_angle(mat)

            # Binarize gripper
            gripper = 2.0 * (ensembled[:, 6:7] > 0.5) - 1.0

            # Build action tensor
            world_vector = torch.tensor(np.array(ensembled[:, :3]), dtype=torch.float32)
            action = torch.cat([world_vector, rot_axangle, torch.tensor(np.array(gripper), dtype=torch.float32)], dim=1)

            raw_obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            done = bool(truncated.any())

        success = info.get("success", False)
        if isinstance(success, torch.Tensor):
            success = success.item()
        successes += int(success)
        print(f"Episode {ep}: success={success} steps={steps} running={successes/(ep+1):.3f}")

    print(f"\nFinal: {successes}/{num_eps} = {successes/num_eps:.3f}")
    env.close()


if __name__ == "__main__":
    main()
