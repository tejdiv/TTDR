"""Side-by-side comparison: SimplerEnv OctoInference vs our wrapper."""

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
import mani_skill.envs  # noqa
from octo.model.octo_model import OctoModel
from simpler_env.policies.octo.octo_model import OctoInference
from simpler_env.utils.env.observation_utils import get_image_from_maniskill3_obs_dict

from recap.envs.perturbations import OctoEnvWrapper, _torch_to_numpy, _extract_image, _resize_image_batch, ActionEnsembler, postprocess_octo_action


def main():
    # Load model
    model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base-1.5")
    action_mean = jnp.array(model.dataset_statistics["bridge_dataset"]["action"]["mean"])
    action_std = jnp.array(model.dataset_statistics["bridge_dataset"]["action"]["std"])

    # SimplerEnv inference
    simpler = OctoInference(model=model, policy_setup="widowx_bridge", init_rng=0)

    # Create env
    env = gym.make("PutCarrotOnPlateInScene-v1", obs_mode="rgb+segmentation")

    obs, _ = env.reset(seed=0, options={"episode_id": torch.tensor([0])})
    instruction = env.unwrapped.get_language_instruction()
    if isinstance(instruction, list):
        instruction = instruction[0]
    print(f"Instruction: '{instruction}'")

    # SimplerEnv: reset policy
    simpler.reset(instruction)

    # Our wrapper: setup
    our_task = model.create_tasks(texts=[instruction])
    our_rng = jax.random.PRNGKey(0)

    # Get image from obs (same for both)
    image_simpler = get_image_from_maniskill3_obs_dict(env, obs)  # torch uint8 (B,H,W,3)
    print(f"Image shape (simpler): {image_simpler.shape}, dtype: {image_simpler.dtype}")
    print(f"Image range: [{image_simpler.min()}, {image_simpler.max()}]")

    # Our extraction
    obs_np = _torch_to_numpy(obs)
    image_ours = _extract_image(obs_np)  # numpy uint8 (B,H,W,3)
    print(f"Image shape (ours): {image_ours.shape}, dtype: {image_ours.dtype}")

    # Check images are identical
    image_simpler_np = image_simpler.cpu().numpy()
    print(f"Images identical: {np.array_equal(image_simpler_np, image_ours)}")

    # Compare resize
    from functools import partial
    image_float = image_ours.astype(np.float32)
    our_resized = _resize_image_batch(image_float)
    simpler_resized = simpler._resize_image(jnp.array(image_simpler_np))
    print(f"Resized identical: {np.array_equal(np.array(our_resized), np.array(simpler_resized))}")

    # Now step both for 5 steps and compare actions
    print("\n=== Step-by-step comparison ===")
    # Reset env for clean state
    obs, _ = env.reset(seed=0, options={"episode_id": torch.tensor([0])})
    simpler.reset(instruction)

    # For our pipeline, create wrapper
    our_wrapper = OctoEnvWrapper(env)
    # Manually process the first obs (env already reset)
    obs_np = _torch_to_numpy(obs)
    image = _extract_image(obs_np)
    image = _resize_image_batch(image.astype(np.float32))
    our_wrapper._image_history.append(image)
    our_wrapper._num_valid = 1

    our_rng = jax.random.PRNGKey(0)
    our_ensembler = ActionEnsembler(pred_action_horizon=4, temp=0.0)

    for step in range(5):
        # SimplerEnv step
        image_step = get_image_from_maniskill3_obs_dict(env, obs)
        raw_action_s, action_s = simpler.step(image_step, instruction)

        # Our step - get obs from wrapper state
        images_ours = jnp.stack(list(our_wrapper._image_history), axis=1)
        B = images_ours.shape[0]
        T = len(our_wrapper._image_history)
        pad_mask = jnp.ones((B, T), dtype=jnp.float32)
        our_obs = {"image_primary": images_ours, "pad_mask": pad_mask}

        our_rng, act_rng = jax.random.split(our_rng)
        norm_actions_ours = model.sample_actions(our_obs, our_task, rng=act_rng)
        actions_ours = norm_actions_ours * action_std[None] + action_mean[None]
        raw_ours = np.array(actions_ours[0])  # (4, 7)

        # SimplerEnv raw actions (before post-processing)
        raw_simpler = np.array(raw_action_s["world_vector"][0].cpu().numpy())

        # Compare
        ens_ours = our_ensembler.ensemble_action(raw_ours)

        print(f"\nStep {step}:")
        print(f"  SimplerEnv rng: {simpler.rng}")
        print(f"  Our rng:        {act_rng}")
        print(f"  Our raw[0,:3]:     {raw_ours[0,:3].round(6)}")
        print(f"  Simpler raw[:3]:   {raw_simpler.round(6)}")
        print(f"  Our ensembled[:3]: {ens_ours[:3].round(6)}")

        # Execute SimplerEnv's action on env
        act = torch.cat(
            [action_s["world_vector"], action_s["rot_axangle"], action_s["gripper"]], dim=1
        )
        obs, reward, terminated, truncated, info = env.step(act)

        # Update our wrapper's image history with the new obs
        obs_np = _torch_to_numpy(obs)
        image = _extract_image(obs_np)
        image = _resize_image_batch(image.astype(np.float32))
        our_wrapper._image_history.append(image)
        our_wrapper._num_valid = min(our_wrapper._num_valid + 1, 2)


if __name__ == "__main__":
    main()
