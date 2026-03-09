"""Use SimplerEnv's OctoInference class directly - exactly as test_simplerenv_eval.py
but import our model and verify it gets 25%."""

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

import gymnasium as gym
from mani_skill.envs.tasks.digital_twins.bridge_dataset_eval import *  # noqa
from simpler_env.utils.env.observation_utils import get_image_from_maniskill3_obs_dict
from simpler_env.policies.octo.octo_model import OctoInference

env = gym.make("PutCarrotOnPlateInScene-v1", obs_mode="rgb+segmentation", num_envs=1)
model = OctoInference(
    model_type="octo-base", policy_setup="widowx_bridge", init_rng=0, action_scale=1
)

successes = 0
num_eps = 20

for ep in range(num_eps):
    seed = ep
    obs, _ = env.reset(seed=seed, options={"episode_id": torch.tensor([seed])})
    instruction = env.unwrapped.get_language_instruction()
    model.reset(instruction)

    image = get_image_from_maniskill3_obs_dict(env, obs)
    done = False
    steps = 0

    while not done:
        raw_action, action = model.step(image, instruction)
        act = torch.cat(
            [action["world_vector"], action["rot_axangle"], action["gripper"]], dim=1
        )

        obs, reward, terminated, truncated, info = env.step(act)
        steps += 1
        done = bool(truncated.any())
        image = get_image_from_maniskill3_obs_dict(env, obs)

    success = info.get("success", False)
    if isinstance(success, torch.Tensor):
        success = success.item()
    successes += int(success)
    print(f"Episode {ep}: success={success} steps={steps}  running={successes/(ep+1):.3f}")

print(f"\nFinal: {successes}/{num_eps} = {successes/num_eps:.3f}")
env.close()
