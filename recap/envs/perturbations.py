"""SimplerEnv wrapper with dynamics perturbations and state save/restore.

SimplerEnv (ManiSkill3 branch) provides GPU-parallelized WidowX Bridge tasks
with visual matching to Octo's Bridge V2 training data. The InScene task
variants (env IDs ending in -v1) have visual matching built in — camera angles,
lighting, textures, and objects are matched to Bridge V2.

Perturbations modify low-level dynamics (friction, mass, damping) via SAPIEN
API to degrade zero-shot performance, which RECAP adaptation then recovers.

OctoEnvWrapper converts ManiSkill observations to Octo's expected format:
  - Extracts RGB from sensor_data/3rd_view_camera/rgb
  - Resizes to 256x256
  - Maintains a 2-frame image history window
  - Returns {"image_primary": (B,T,256,256,3), "timestep_pad_mask": (B,T)}
"""

from collections import deque
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

IMAGE_SIZE = 256
HORIZON = 2  # Octo observation window size


def postprocess_octo_action(raw_action):
    """Convert Octo's 7-dim output to SimplerEnv's expected action format.

    Octo outputs: [dx, dy, dz, roll, pitch, yaw, gripper] (denormalized).
    ManiSkill expects: [dx, dy, dz, ax, ay, az, gripper] where rotation is
    axis-angle and gripper is binarized to +1 (open) or -1 (close).

    Uses mani_skill's rotation_conversions (euler→matrix→axis-angle) to match
    SimplerEnv's OctoInference.step() exactly.
    """
    import torch
    from mani_skill.utils.geometry import rotation_conversions

    raw_action = np.asarray(raw_action).flatten()
    world_vector = raw_action[:3]
    euler = torch.tensor([[raw_action[3], raw_action[4], raw_action[5]]], dtype=torch.float32)
    gripper_raw = raw_action[6]

    # Euler (XYZ) → rotation matrix → axis-angle (matches SimplerEnv exactly)
    mat = rotation_conversions.euler_angles_to_matrix(euler, "XYZ")
    rot_axangle = rotation_conversions.matrix_to_axis_angle(mat).numpy()[0]

    # Binarize gripper: > 0.5 → open (+1), else → close (-1)
    gripper = 1.0 if gripper_raw > 0.5 else -1.0

    return np.concatenate([world_vector, rot_axangle, [gripper]])


def _torch_to_numpy(obs):
    """Recursively convert torch tensors in obs dict to numpy arrays."""
    import torch
    if isinstance(obs, torch.Tensor):
        return obs.cpu().numpy()
    elif isinstance(obs, dict):
        return {k: _torch_to_numpy(v) for k, v in obs.items()}
    elif isinstance(obs, (list, tuple)):
        return type(obs)(_torch_to_numpy(v) for v in obs)
    return obs


def _extract_image(obs, camera_name="3rd_view_camera"):
    """Extract RGB from ManiSkill obs dict. Returns (B, H, W, 3) numpy."""
    return obs["sensor_data"][camera_name]["rgb"]


@jax.jit
def _resize_image_batch(images):
    """Resize (B, H, W, 3) float to (B, IMAGE_SIZE, IMAGE_SIZE, 3) uint8."""
    resized = jax.vmap(
        partial(jax.image.resize,
                shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                method="lanczos3", antialias=True)
    )(images)
    return jnp.clip(jnp.round(resized), 0, 255).astype(jnp.uint8)


class ActionEnsembler:
    """Temporal action ensemble matching SimplerEnv's implementation.

    Averages overlapping action predictions across time steps. With temp=0.0,
    all predictions get equal weight (uniform average of the diagonal).
    """

    def __init__(self, pred_action_horizon, temp=0.0):
        self.pred_action_horizon = pred_action_horizon
        self.action_history = deque(maxlen=pred_action_horizon)
        self.temp = temp

    def reset(self):
        self.action_history.clear()

    def ensemble_action(self, cur_action):
        """cur_action: (pred_horizon, 7). Returns (7,) ensembled action."""
        self.action_history.append(cur_action)
        n = len(self.action_history)
        curr_act_preds = np.stack(
            [pred_actions[i]
             for (i, pred_actions) in zip(range(n - 1, -1, -1), self.action_history)]
        )
        weights = np.exp(-self.temp * np.arange(n))
        weights = weights / weights.sum()
        return np.sum(weights[:, None] * curr_act_preds, axis=0)


class OctoEnvWrapper:
    """Wraps ManiSkill env to return Octo-formatted observations.

    Matches SimplerEnv's OctoInference observation handling:
      - No zero-padding: history grows from 1 to HORIZON frames
      - Returns {"image_primary": (B,T,H,W,3), "timestep_pad_mask": (B,T)}
      - T starts at 1 on first step, grows to HORIZON
    """

    def __init__(self, env):
        self._env = env
        self._image_history = deque(maxlen=HORIZON)
        self._num_valid = 0

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _process_obs(self, raw_obs):
        """Convert raw ManiSkill obs to Octo format (matching SimplerEnv)."""
        raw_obs = _torch_to_numpy(raw_obs)
        image = _extract_image(raw_obs)  # (B, H, W, 3) uint8 numpy
        image = _resize_image_batch(image.astype(np.float32))  # (B, 256, 256, 3)
        self._image_history.append(image)
        self._num_valid = min(self._num_valid + 1, HORIZON)

        # Stack actual history (no zero-padding — matches SimplerEnv)
        images = jnp.stack(list(self._image_history), axis=1)  # (B, T, H, W, 3)
        B = images.shape[0]
        T = len(self._image_history)
        pad_mask = jnp.ones((B, T), dtype=jnp.float32)
        pad_mask = pad_mask.at[:, :T - min(T, self._num_valid)].set(0)

        return {"image_primary": images, "pad_mask": pad_mask}

    def reset(self, **kwargs):
        self._image_history.clear()
        self._num_valid = 0
        raw_obs, info = self._env.reset(**kwargs)
        return self._process_obs(raw_obs), info

    def step(self, action):
        # Convert JAX arrays to numpy for ManiSkill
        action = np.asarray(action)
        raw_obs, reward, terminated, truncated, info = self._env.step(action)
        return self._process_obs(raw_obs), float(reward), terminated, truncated, info


def make_env(env_id, perturbation=None, perturbation_scale=None):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401 — registers ManiSkill envs
    """Create a SimplerEnv environment with optional dynamics perturbation.

    Uses the same control mode as SimplerEnv's evaluation scripts:
    arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos

    Returns observations in Octo format: {"image_primary", "timestep_pad_mask"}.

    Args:
        env_id: SimplerEnv task ID, e.g. "PutCarrotOnPlateInScene-v1".
        perturbation: One of "object_friction", "object_mass",
                      "joint_damping", "gripper_friction", or None.
        perturbation_scale: Multiplicative scale factor for the perturbation.

    Returns:
        env: OctoEnvWrapper producing Octo-formatted observations.
    """
    env = gym.make(
        env_id,
        obs_mode="rgb+segmentation",
    )

    if perturbation is not None:
        apply_perturbation(env, perturbation, perturbation_scale)
    return OctoEnvWrapper(env)


def apply_perturbation(env, perturbation, scale):
    """Apply dynamics perturbation via SAPIEN API after env.reset().

    Hooks into env.reset() so the perturbation is applied every time the
    scene is rebuilt. The SAPIEN actor/joint API is accessed through the
    unwrapped ManiSkill env.
    """
    original_reset = env.reset

    def perturbed_reset(**kwargs):
        obs, info = original_reset(**kwargs)
        scene = env.unwrapped.scene

        if perturbation == "object_friction":
            for actor in scene.get_all_actors():
                if "object" in actor.name or "target" in actor.name:
                    for shape in actor.get_collision_shapes():
                        mat = shape.get_physical_material()
                        mat.set_static_friction(mat.static_friction * scale)
                        mat.set_dynamic_friction(mat.dynamic_friction * scale)

        elif perturbation == "object_mass":
            for actor in scene.get_all_actors():
                if "object" in actor.name or "target" in actor.name:
                    actor.set_mass(actor.mass * scale)

        elif perturbation == "joint_damping":
            robot = env.unwrapped.agent.robot
            for joint in robot.get_active_joints():
                joint.set_drive_properties(
                    joint.stiffness, joint.damping * scale
                )

        elif perturbation == "gripper_friction":
            robot = env.unwrapped.agent.robot
            for link in robot.get_links():
                if "finger" in link.name or "gripper" in link.name:
                    for shape in link.get_collision_shapes():
                        mat = shape.get_physical_material()
                        mat.set_static_friction(mat.static_friction * scale)
                        mat.set_dynamic_friction(mat.dynamic_friction * scale)

        return obs, info

    env.reset = perturbed_reset
    return env


def save_env_state(env):
    """Save simulator state for K-way forking."""
    return env.unwrapped.get_state()


def restore_env_state(env, state):
    """Restore simulator state after forking."""
    env.unwrapped.set_state(state)
