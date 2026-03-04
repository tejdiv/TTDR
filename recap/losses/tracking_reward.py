"""Dense per-chunk tracking reward from the frozen world model.

Takes raw Octo encoder outputs (768-dim) and the frozen world model,
projects them into contrastive space, computes the anchor, and returns
a scale-invariant reward signal.
"""

import jax.numpy as jnp

from recap.models.projection_head import ProjectionHead
from recap.models.dynamics_predictor import DynamicsPredictor


def tracking_reward(z_t, z_t1, z_t_plus_m, wm_params, eps=1e-6):
    """Compute the normalized auxiliary tracking reward.

    Args:
        z_t: Octo encoder readout at time t, 1-frame context (batch, 768).
        z_t1: Octo encoder readout at time t, 2-frame context (batch, 768).
        z_t_plus_m: Octo encoder readout m steps later (batch, 768).
        wm_params: Frozen world model params dict with keys
                   "projection_head" and "dynamics_predictor".
        eps: Small constant to avoid division by zero.

    Returns:
        r_aux: (batch,) scale-invariant tracking reward in [-1/eps, 0].
    """
    h = ProjectionHead()
    f_psi = DynamicsPredictor()

    # Project all encoder outputs into contrastive space (256-dim, unit sphere)
    z_prime_t = h.apply(wm_params["projection_head"], z_t)
    z_prime_t1 = h.apply(wm_params["projection_head"], z_t1)
    z_prime_target = h.apply(wm_params["projection_head"], z_t_plus_m)

    # Dynamics predictor: predict where we should be m steps ahead
    dynamics_input = jnp.concatenate([z_prime_t, z_prime_t1], axis=-1)
    anchor = f_psi.apply(wm_params["dynamics_predictor"], dynamics_input)

    # Tracking error: how far actual outcome is from prediction
    tracking_error = jnp.sum((z_prime_target - anchor) ** 2, axis=-1)

    # Displacement: how far the prediction is from current state
    displacement = jnp.sum((anchor - z_prime_t) ** 2, axis=-1) + eps

    return -tracking_error / displacement
