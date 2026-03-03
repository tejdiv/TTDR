"""InfoNCE contrastive loss with L2 distance for world model training.

Pulls predicted next-state ẑ'_{t+m} toward actual next-state z'_{t+m},
pushes away from other batch elements (negatives).

Uses L2 distance (not dot product) because the contrastive space Z' is
L2-normalized — distances on the unit sphere are the meaningful metric
for dynamics similarity.

Loss (Eq. 5 in paper):
    L = -1/N Σ_i log [ exp(-||ẑ_i - z⁺_i||² / τ) / Σ_j exp(-||ẑ_i - z_j||² / τ) ]

where:
    ẑ_i  = f_ψ(h(φ(o_t^i)))          — predicted anchor
    z⁺_i = h(φ(o_{t+m}^i))           — actual next-state (positive)
    z_j  = h(φ(o_{t+m}^j))           — other batch elements (negatives)
"""

import jax
import jax.numpy as jnp


def infonce_loss(
    predicted: jnp.ndarray,
    targets: jnp.ndarray,
    temperature: float = 0.1,
    traj_ids: jnp.ndarray = None,
    intra_weight: float = 0.5,
) -> jnp.ndarray:
    """Compute InfoNCE loss with L2 distance and optional mixed intra/cross-trajectory negatives.

    When traj_ids is provided, computes a weighted combination of:
    - Cross-trajectory loss: all batch elements as negatives (scene discrimination)
    - Intra-trajectory loss: only same-trajectory negatives (temporal dynamics)

    Both share the same (B, B) distance matrix — the only difference is masking
    logits to -inf for excluded negatives before softmax. Zero compute overhead.

    Args:
        predicted: Predicted anchors ẑ'_{t+m}, shape (batch, dim). L2-normalized.
        targets: Actual next-states z'_{t+m}, shape (batch, dim). L2-normalized.
        temperature: Temperature parameter τ. Lower = sharper distribution.
        traj_ids: Trajectory ID for each batch element, shape (batch,). If None,
            falls back to standard cross-trajectory InfoNCE.
        intra_weight: Weight for intra-trajectory loss. 0.0 = cross-only,
            1.0 = intra-only, 0.5 = equal mix.

    Returns:
        Scalar loss value (mean over batch).
    """
    batch_size = predicted.shape[0]

    # Pairwise squared L2 distances: ||ẑ_i - z_j||²  — computed once
    diff = predicted[:, None, :] - targets[None, :, :]
    sq_distances = jnp.sum(diff ** 2, axis=-1)  # (B, B)

    # Logits: -distance / temperature (higher logit = closer = more similar)
    logits = -sq_distances / temperature  # (B, B)

    # Labels: diagonal (each predicted should match its own target)
    labels = jnp.arange(batch_size)

    if traj_ids is None:
        raise ValueError(
            "traj_ids is required. Ensure the HDF5 file contains 'traj_id' "
            "(re-run precompute_encodings.py) and the dataloader includes it in batches."
        )

    # Cross-trajectory loss (all negatives)
    loss_cross = optax_softmax_cross_entropy(logits, labels)

    # Intra-trajectory loss (mask to same-trajectory negatives only)
    same_traj = (traj_ids[:, None] == traj_ids[None, :])  # (B, B) bool
    intra_logits = jnp.where(same_traj, logits, -1e9)
    loss_intra = optax_softmax_cross_entropy(intra_logits, labels)

    return (1 - intra_weight) * jnp.mean(loss_cross) + intra_weight * jnp.mean(loss_intra)


def optax_softmax_cross_entropy(
    logits: jnp.ndarray, labels: jnp.ndarray
) -> jnp.ndarray:
    """Cross-entropy loss with integer labels.

    Args:
        logits: (batch, num_classes)
        labels: (batch,) integer class indices
    Returns:
        (batch,) per-example losses
    """
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -log_probs[jnp.arange(labels.shape[0]), labels]
