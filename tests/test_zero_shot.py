"""Test Octo zero-shot — matching SimplerEnv protocol exactly."""

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

from octo.model.octo_model import OctoModel
from recap.envs.perturbations import make_env, postprocess_octo_action, ActionEnsembler


def main():
    print("Loading Octo model...")
    model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base")

    action_mean = jnp.array(
        model.dataset_statistics["bridge_dataset"]["action"]["mean"]
    )
    action_std = jnp.array(
        model.dataset_statistics["bridge_dataset"]["action"]["std"]
    )

    env = make_env("PutCarrotOnPlateInScene-v1")

    # Use env's instruction (matches SimplerEnv protocol)
    env.reset(seed=0, options={"episode_id": torch.tensor([0])})
    instruction = env.unwrapped.get_language_instruction()
    if isinstance(instruction, list):
        instruction = instruction[0]
    print(f"Instruction: '{instruction}'")
    task = model.create_tasks(texts=[instruction])

    rng = jax.random.PRNGKey(0)
    successes = 0
    num_trials = 20

    for trial in range(num_trials):
        rng, _ = jax.random.split(rng)
        obs, _ = env.reset(
            seed=trial, options={"episode_id": torch.tensor([trial])}
        )
        done = False
        steps = 0
        total_reward = 0.0
        ensembler = ActionEnsembler(pred_action_horizon=4, temp=0.0)

        while not done:
            rng, act_rng = jax.random.split(rng)
            norm_actions = model.sample_actions(obs, task, rng=act_rng)

            actions = norm_actions * action_std[None] + action_mean[None]
            raw_actions = np.array(actions[0])

            ensembled = ensembler.ensemble_action(raw_actions)
            action_np = postprocess_octo_action(ensembled)

            obs, reward, terminated, truncated, info = env.step(action_np)
            total_reward += reward
            steps += 1
            done = bool(terminated) or bool(truncated)

        success = info.get("success", False)
        if isinstance(success, torch.Tensor):
            success = success.item()
        successes += int(success)
        print(f"Trial {trial:3d}: success={success}  steps={steps}  "
              f"running={successes/(trial+1):.3f}")

    rate = successes / num_trials
    print(f"\nSuccess rate: {successes}/{num_trials} = {rate:.3f}")
    env.close()


if __name__ == "__main__":
    main()
