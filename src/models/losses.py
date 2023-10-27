import jax
import jax.numpy as jnp


@jax.jit
def binary_logistic_loss(labels: jnp.ndarray, logits: jnp.ndarray) -> float:
    """Binary logistic loss.

    Args:
      label: ground-truth integer label (0 or 1).
      logit: score produced by the model (float).
    Returns:
      loss value
    """
    # Softplus is the Fenchel conjugate of the Fermi-Dirac negentropy on [0, 1].
    # softplus = proba * logit - xlogx(proba) - xlogx(1 - proba),
    # where xlogx(proba) = proba * log(proba).
    # Use -log sigmoid(logit) = softplus(-logit)
    # and 1 - sigmoid(logit) = sigmoid(-logit).
    return jax.nn.softplus(jnp.where(labels, -logits, logits))


def binary_sparsemax_loss(label: int, logit: float) -> float:
    """Binary sparsemax loss.

    Args:
      label: ground-truth integer label (0 or 1).
      logit: score produced by the model (float).
    Returns:
      loss value

    References:
      Learning with Fenchel-Young Losses. Mathieu Blondel, André F. T. Martins,
      Vlad Niculae. JMLR 2020. (Sec. 4.4)
    """
    def sparse_plus(x):
        return jnp.where(x <= -1.0, 0.0, jnp.where(x >= 1.0, x, (x + 1.0)**2/4))
    return sparse_plus(jnp.where(label, -logit, logit))
