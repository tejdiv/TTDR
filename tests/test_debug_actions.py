"""Debug: Compare actions from our pipeline vs SimplerEnv side-by-side on same env."""

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

from recap.envs.perturbations import make_env, postprocess_octo_action, ActionEnsembler


def test_our_pipeline_with_simpler_env_actions():
    """Use SimplerEnv's OctoInference for actions, but our env wrapper for obs.

    This tells us if the issue is in obs processing or action processing.
    """
    model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base-1.5")

    # Raw ManiSkill env (for SimplerEnv)
    raw_env = gym.make("PutCarrotOnPlateInScene-v1", obs_mode="rgb+segmentation")
    simpler = OctoInference(model=model, policy_setup="widowx_bridge", init_rng=0)

    successes = 0
    num_eps = 20

    for ep in range(num_eps):
        obs, _ = raw_env.reset(seed=ep, options={"episode_id": torch.tensor([ep])})
        instruction = raw_env.unwrapped.get_language_instruction()
        if isinstance(instruction, list):
            instruction = instruction[0]
        simpler.reset(instruction)

        image = get_image_from_maniskill3_obs_dict(raw_env, obs)
        done = False
        steps = 0

        while not done:
            raw_action, action = simpler.step(image, instruction)
            act = torch.cat(
                [action["world_vector"], action["rot_axangle"], action["gripper"]], dim=1
            )
            obs, reward, terminated, truncated, info = raw_env.step(act)
            steps += 1
            done = bool(truncated.any())
            image = get_image_from_maniskill3_obs_dict(raw_env, obs)

        success = info.get("success", False)
        if isinstance(success, torch.Tensor):
            success = success.item()
        successes += int(success)
        print(f"SimplerEnv ep {ep}: success={success} steps={steps} running={successes/(ep+1):.3f}")

    print(f"\nSimpler baseline: {successes}/{num_eps} = {successes/num_eps:.3f}")
    raw_env.close()


def test_our_full_pipeline():
    """Our full pipeline: OctoEnvWrapper + our action processing."""
    model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base-1.5")
    action_mean = jnp.array(model.dataset_statistics["bridge_dataset"]["action"]["mean"])
    action_std = jnp.array(model.dataset_statistics["bridge_dataset"]["action"]["std"])

    env = make_env("PutCarrotOnPlateInScene-v1")

    # Match SimplerEnv's RNG: split 5 times
    rng = jax.random.PRNGKey(0)
    for _ in range(5):
        rng, _ = jax.random.split(rng)

    successes = 0
    num_eps = 20

    for ep in range(num_eps):
        obs, _ = env.reset(seed=ep, options={"episode_id": torch.tensor([ep])})
        instruction = env.unwrapped.get_language_instruction()
        if isinstance(instruction, list):
            instruction = instruction[0]
        task = model.create_tasks(texts=[instruction])

        done = False
        steps = 0
        ensembler = ActionEnsembler(pred_action_horizon=4, temp=0.0)

        while not done:
            rng, act_rng = jax.random.split(rng)
            norm_actions = model.sample_actions(obs, task, rng=act_rng)
            actions = norm_actions * action_std[None] + action_mean[None]
            raw_actions = np.array(actions[0])

            ensembled = ensembler.ensemble_action(raw_actions)
            action_np = postprocess_octo_action(ensembled)

            obs, reward, terminated, truncated, info = env.step(action_np)
            steps += 1
            done = bool(terminated) or bool(truncated)

        success = info.get("success", False)
        if hasattr(success, "item"):
            success = success.item()
        successes += int(success)
        print(f"Our pipeline ep {ep}: success={success} steps={steps} running={successes/(ep+1):.3f}")

    print(f"\nOur pipeline: {successes}/{num_eps} = {successes/num_eps:.3f}")
    env.close()


def test_hybrid():
    """Use our OctoEnvWrapper for obs, but SimplerEnv's OctoInference for model call.

    This isolates: is the issue in our obs wrapper or our model call / action processing?
    """
    model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base-1.5")
    action_mean = jnp.array(model.dataset_statistics["bridge_dataset"]["action"]["mean"])
    action_std = jnp.array(model.dataset_statistics["bridge_dataset"]["action"]["std"])

    # Create raw env + our wrapper
    raw_env = gym.make("PutCarrotOnPlateInScene-v1", obs_mode="rgb+segmentation")
    simpler = OctoInference(model=model, policy_setup="widowx_bridge", init_rng=0)

    successes = 0
    num_eps = 10

    for ep in range(num_eps):
        raw_obs, _ = raw_env.reset(seed=ep, options={"episode_id": torch.tensor([ep])})
        instruction = raw_env.unwrapped.get_language_instruction()
        if isinstance(instruction, list):
            instruction = instruction[0]
        simpler.reset(instruction)

        done = False
        steps = 0

        while not done:
            # Use SimplerEnv's OctoInference for the full action pipeline
            image = get_image_from_maniskill3_obs_dict(raw_env, raw_obs)
            raw_action, action = simpler.step(image, instruction)

            # But step the raw env with the action
            act = torch.cat(
                [action["world_vector"], action["rot_axangle"], action["gripper"]], dim=1
            )
            raw_obs, reward, terminated, truncated, info = raw_env.step(act)
            steps += 1
            done = bool(truncated.any())

        success = info.get("success", False)
        if isinstance(success, torch.Tensor):
            success = success.item()
        successes += int(success)
        print(f"Hybrid ep {ep}: success={success} steps={steps}")

    print(f"\nHybrid: {successes}/{num_eps} = {successes/num_eps:.3f}")
    raw_env.close()


if __name__ == "__main__":
    print("=" * 60)
    print("TEST: Our full pipeline")
    print("=" * 60)
    test_our_full_pipeline()
