"""Test octo-base-1.5 with full observation schema (pad_mask_dict + timestep_pad_mask)."""

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


@jax.jit
def resize_image(image):
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

        image_history = deque(maxlen=HORIZON)
        num_image_history = 0
        action_history = deque(maxlen=4)

        done = False
        steps = 0

        while not done:
            image = raw_obs["sensor_data"]["3rd_view_camera"]["rgb"]
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
            image = resize_image(image.astype(np.float32))

            image_history.append(image)
            num_image_history = min(num_image_history + 1, HORIZON)

            images = jnp.stack(list(image_history), axis=1)  # (B, T, 256, 256, 3)
            B = images.shape[0]
            T = len(image_history)

            # Build boolean pad mask (True = valid, False = padding)
            pad_mask_bool = jnp.ones((B, T), dtype=bool)
            # No padding needed since we grow history (no zero-padding)

            # Build full obs dict matching octo-base-1.5 example_batch
            input_obs = {
                "image_primary": images,
                "timestep_pad_mask": pad_mask_bool,
                "pad_mask_dict": {
                    "image_primary": pad_mask_bool,
                    "image_wrist": jnp.zeros((B, T), dtype=bool),  # no wrist cam
                    "timestep": pad_mask_bool,
                },
            }

            rng, key = jax.random.split(rng)
            norm_actions = model.sample_actions(input_obs, task, rng=key)

            raw_actions = norm_actions * action_std[None] + action_mean[None]
            raw_actions_np = np.array(raw_actions)

            # Action ensemble
            action_history.append(raw_actions_np)
            n = len(action_history)
            curr_act_preds = jnp.stack(
                [pred[:, i] for (i, pred) in zip(range(n - 1, -1, -1), action_history)]
            )
            weights = jnp.exp(-0.0 * jnp.arange(n))
            weights = weights / weights.sum()
            ensembled = jnp.sum(weights[:, None, None] * curr_act_preds, axis=0)

            # Euler -> axis-angle (mani_skill conversion)
            euler = torch.tensor(np.array(ensembled[:, 3:6]), dtype=torch.float32)
            mat = rotation_conversions.euler_angles_to_matrix(euler, "XYZ")
            rot_axangle = rotation_conversions.matrix_to_axis_angle(mat)

            gripper = 2.0 * (ensembled[:, 6:7] > 0.5) - 1.0

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

    print(f"\nocto-base-1.5 (full schema): {successes}/{num_eps} = {successes/num_eps:.3f}")
    env.close()


if __name__ == "__main__":
    main()
