import jax
import jax.numpy as jnp

from operator import getitem

Array = jnp.ndarray

@jax.jit
def log_likelihood(labels: Array, logits: Array) -> float:
    """Log-likelihood loss.

    Args:
      label: ground-truth integer label (0 or 1).
      logit: score produced by the model (float).
    Returns:
      loss value
    """
    # labels = jax.nn.one_hot(labels, 2)
    log_likelihood = jnp.mean(labels * jax.nn.log_softmax(logits))
    return -log_likelihood
  
def cross_entropy(labels: Array, logits: Array) -> float:
    """Cross-entropy loss.

    Args:
      label: ground-truth integer label (0 or 1).
      logit: score produced by the model (float).
    Returns:
      loss value
    """
    logits = jax.nn.log_softmax(logits)
    loss = jax.vmap(getitem)(logits, labels)
    return -loss

@jax.jit
def binary_logistic_loss(labels: Array, logits: Array) -> float:
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

@jax.jit
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

@jax.jit
def sigmoid_cross_entropy_with_logits(labels: Array, logits: Array) -> Array:
  """Sigmoid cross entropy loss."""
  zeros = jnp.zeros_like(logits, dtype=logits.dtype)
  condition = logits >= zeros
  relu_logits = jnp.where(condition, logits, zeros)
  neg_abs_logits = jnp.where(condition, -logits, logits)
  return relu_logits - logits * labels + jnp.log1p(jnp.exp(neg_abs_logits))

def binary_cross_entropy(y_hat, y):
    bce = y * jnp.log(y_hat) + (1 - y) * jnp.log(1 - y_hat)
    return jnp.mean(-bce) 