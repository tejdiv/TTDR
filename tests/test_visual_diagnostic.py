"""Visual diagnostic: save rollout videos + per-step tracking rewards.

Saves MP4 videos for:
  A) Zero-shot policy (2 rollouts)
  B) RECAP-adapted policy (2 rollouts)

Also logs per-step tracking reward r_aux so we can see if the WM reward
correlates with what the robot is actually doing.

Videos are saved to /home/ubuntu/videos/ and can be downloaded via scp.
"""

import os
os.environ["TF_CPP_MIN_LOG_ALLOW_GROWTH"] = "3"
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
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

import gymnasium as gym
import mani_skill.envs  # noqa: registers envs
import imageio

from absl import app, logging

from octo.model.octo_model import OctoModel
from recap.envs.perturbations import OctoEnvWrapper, postprocess_octo_action, ActionEnsembler
from recap.training.recap_adaptation import (
    _run_transformer, sample_actions_from_readouts,
    inject_indicator, recap_adapt,
)
from recap.losses.tracking_reward import tracking_reward_with_value
from recap.training.train_world_model import WorldModel
from flax.training import checkpoints


def make_video_env(env_id):
    """Create env that returns both Octo-formatted obs AND raw rendered frames."""
    raw_env = gym.make(env_id, obs_mode="rgb+segmentation", render_mode="rgb_array")
    return OctoEnvWrapper(raw_env), raw_env


def rollout_with_video(octo_model, params, task, raw_env, octo_env, rng,
                       wm_model=None, wm_params=None, indicator_embed=None,
                       label="rollout"):
    """Run one rollout, save video + log tracking rewards."""
    action_mean = jnp.array(
        octo_model.dataset_statistics["bridge_dataset"]["action"]["mean"]
    )
    action_std = jnp.array(
        octo_model.dataset_statistics["bridge_dataset"]["action"]["std"]
    )

    obs, _ = octo_env.reset()
    done = False
    ensembler = ActionEnsembler(pred_action_horizon=4, temp=0.0)
    frames = []
    step_data = []
    step_num = 0
    prev_z = None

    while not done:
        rng, act_rng = jax.random.split(rng)

        # Render frame from raw env
        frame = raw_env.render()
        if hasattr(frame, 'cpu'):
            frame = frame.cpu().numpy()
        if frame.ndim == 4:
            frame = frame[0]  # (H, W, 3)
        frames.append(frame)

        # Forward pass
        pad_mask = obs["pad_mask"]
        trans_out = _run_transformer(
            octo_model.module, params, obs, task, pad_mask
        )

        # WM readouts for tracking reward
        tokens = trans_out["readout_action"].tokens
        z_t = jnp.mean(tokens[:, 0, :, :], axis=1)
        z_t1 = jnp.mean(tokens[:, -1, :, :], axis=1)

        # Inject indicator if adapted
        trans_out_act = trans_out
        if indicator_embed is not None:
            trans_out_act = inject_indicator(trans_out, indicator_embed)

        norm_actions = sample_actions_from_readouts(
            octo_model, params, trans_out_act, act_rng
        )
        actions = norm_actions * action_std[None] + action_mean[None]
        raw_actions = np.array(actions[0])
        ensembled = ensembler.ensemble_action(raw_actions)
        action_np = postprocess_octo_action(ensembled)

        obs, reward, terminated, truncated, info = octo_env.step(action_np)
        done = terminated or truncated

        # Compute tracking reward if we have WM and previous z
        r_aux_val = 0.0
        v_val = 0.0
        if wm_model is not None and wm_params is not None and step_num > 0:
            tokens_next = _run_transformer(
                octo_model.module, octo_model.params, obs, task, obs["pad_mask"]
            )["readout_action"].tokens
            z_next = jnp.mean(tokens_next[:, 0, :, :], axis=1)
            r_aux, v_pred = tracking_reward_with_value(
                prev_z_t, prev_z_t1, z_next, wm_params, wm_model.apply
            )
            r_aux_val = float(r_aux)
            v_val = float(v_pred) if v_pred is not None else 0.0

        success = info.get("success", False)
        if hasattr(success, "item"):
            success = success.item()

        step_data.append({
            "step": step_num,
            "r_aux": r_aux_val,
            "v_pred": v_val,
            "env_reward": float(reward),
            "success": success,
        })

        logging.info(
            f"  {label} step {step_num:3d} | "
            f"r_aux={r_aux_val:.4f} V={v_val:.4f} | "
            f"env_r={float(reward):.3f} | "
            f"{'SUCCESS' if success else ''}"
        )

        prev_z_t = z_t
        prev_z_t1 = z_t1
        step_num += 1

    # Save video
    os.makedirs("/home/ubuntu/videos", exist_ok=True)
    video_path = f"/home/ubuntu/videos/{label}.mp4"
    writer = imageio.get_writer(video_path, fps=10)
    for f in frames:
        if f.dtype != np.uint8:
            f = np.clip(f * 255, 0, 255).astype(np.uint8) if f.max() <= 1.0 else f.astype(np.uint8)
        writer.append_data(f)
    writer.close()
    logging.info(f"  Saved {len(frames)} frames to {video_path}")

    final_success = step_data[-1]["success"] if step_data else False
    return final_success, step_data


def main(_):
    logging.info("Loading Octo model...")
    octo_model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base")
    octo_params = octo_model.params
    rng = jax.random.PRNGKey(42)

    instruction = "put carrot on plate"
    task = octo_model.create_tasks(texts=[instruction])

    # Load world model for tracking reward logging
    from huggingface_hub import snapshot_download
    wm_path = snapshot_download("4manifold/ttdr-world-model")
    wm_state = checkpoints.restore_checkpoint(wm_path, target=None)
    wm_model = WorldModel(
        projection_head_kwargs={"hidden_dim": 1024, "output_dim": 512},
        dynamics_predictor_kwargs={"hidden_dim": 2048, "num_layers": 4, "output_dim": 512},
        value_head_kwargs={"hidden_dim": 256, "num_layers": 2},
    )
    wm_params = wm_state["params"]

    # --- A: Zero-shot rollouts ---
    logging.info("\n" + "=" * 60)
    logging.info("ZERO-SHOT ROLLOUTS (with tracking reward logging)")
    logging.info("=" * 60)

    for trial in range(3):
        rng, trial_rng = jax.random.split(rng)
        octo_env, raw_env = make_video_env("PutCarrotOnPlateInScene-v1")
        success, data = rollout_with_video(
            octo_model, octo_params, task, raw_env, octo_env, trial_rng,
            wm_model=wm_model, wm_params=wm_params,
            label=f"zero_shot_trial{trial}",
        )
        logging.info(f"  Trial {trial}: success={success}")
        raw_env.close()

    # --- B: RECAP adaptation + adapted rollouts ---
    logging.info("\n" + "=" * 60)
    logging.info("RECAP ADAPTATION (1 episode)")
    logging.info("=" * 60)

    class Config:
        rank = 8
        backbone_lora = False
        lr = 1e-4
        update_freq = 4
        num_bc_steps = 200
        bc_batch_size = 16
        buffer_size = 256
        min_buffer = 16
        num_episodes = 1
        recap_alpha = 1.0
        wm_checkpoint = "hf://4manifold/ttdr-world-model"
        wm_projection_head_kwargs = {"hidden_dim": 1024, "output_dim": 512}
        wm_dynamics_predictor_kwargs = {"hidden_dim": 2048, "num_layers": 4, "output_dim": 512}
        wm_value_head_kwargs = {"hidden_dim": 256, "num_layers": 2}

    adapt_octo_env, adapt_raw_env = make_video_env("PutCarrotOnPlateInScene-v1")
    rng, adapt_rng = jax.random.split(rng)
    adapted_params, indicator_embed = recap_adapt(
        octo_model, octo_params,
        adapt_octo_env, instruction, Config(), adapt_rng,
    )
    adapt_raw_env.close()

    logging.info("\n" + "=" * 60)
    logging.info("ADAPTED ROLLOUTS (with tracking reward logging)")
    logging.info("=" * 60)

    for trial in range(3):
        rng, trial_rng = jax.random.split(rng)
        octo_env, raw_env = make_video_env("PutCarrotOnPlateInScene-v1")
        success, data = rollout_with_video(
            octo_model, adapted_params, task, raw_env, octo_env, trial_rng,
            wm_model=wm_model, wm_params=wm_params,
            indicator_embed=indicator_embed,
            label=f"adapted_trial{trial}",
        )
        logging.info(f"  Trial {trial}: success={success}")
        raw_env.close()

    logging.info("\nDone! Videos saved to /home/ubuntu/videos/")


if __name__ == "__main__":
    app.run(main)
