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
) -> jnp.ndarray:
    """Compute InfoNCE loss with L2 distance.

    Args:
        predicted: Predicted anchors ẑ'_{t+m}, shape (batch, dim). L2-normalized.
        targets: Actual next-states z'_{t+m}, shape (batch, dim). L2-normalized.
        temperature: Temperature parameter τ. Lower = sharper distribution.

    Returns:
        Scalar loss value (mean over batch).
    """
    batch_size = predicted.shape[0]

    # Pairwise squared L2 distances: ||ẑ_i - z_j||²
    # predicted: (B, D), targets: (B, D)
    # diff: (B, 1, D) - (1, B, D) = (B, B, D)
    diff = predicted[:, None, :] - targets[None, :, :]
    sq_distances = jnp.sum(diff ** 2, axis=-1)  # (B, B)

    # Logits: -distance / temperature (higher logit = closer = more similar)
    logits = -sq_distances / temperature  # (B, B)

    # Labels: diagonal (each predicted should match its own target)
    labels = jnp.arange(batch_size)

    # Cross-entropy: each row is a distribution over targets
    loss = optax_softmax_cross_entropy(logits, labels)
    return jnp.mean(loss)


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
