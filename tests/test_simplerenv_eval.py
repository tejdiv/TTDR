"""Run SimplerEnv's official OctoInference on ManiSkill3 to verify zero-shot success."""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import gymnasium as gym
import numpy as np
import torch
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

        if steps < 3 or steps == 30:
            wv = action["world_vector"][0].cpu().numpy()
            gr = action["gripper"][0].cpu().item()
            print(f"  ep{ep} step {steps}: xyz={wv.round(5)} grip={gr:.2f}")

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
