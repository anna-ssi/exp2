import optax
import jax
import jax.numpy as jnp
from flax import struct
from flax.training import train_state

from typing import Any, Dict, Tuple, Optional

Array = jnp.ndarray
TrainState = train_state.TrainState

class Metrics(struct.PyTreeNode):
  """Computed metrics."""

  loss: float
  accuracy: float
  count: Optional[int] = None

@jax.vmap
def sigmoid_cross_entropy_with_logits(*, labels: Array, logits: Array) -> Array:
  """Sigmoid cross entropy loss."""
  zeros = jnp.zeros_like(logits, dtype=logits.dtype)
  condition = logits >= zeros
  relu_logits = jnp.where(condition, logits, zeros)
  neg_abs_logits = jnp.where(condition, -logits, logits)
  return relu_logits - logits * labels + jnp.log1p(jnp.exp(neg_abs_logits))

def compute_metrics(*, labels: Array, logits: Array) -> Metrics:
  """Computes the metrics, summed across the batch if a batch is provided."""
  if labels.ndim == 1:  # Prevent the labels from broadcasting over the logits.
    labels = jnp.expand_dims(labels, axis=1)
    
  loss = sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
  binary_predictions = logits >= 0.0
  binary_accuracy = jnp.equal(binary_predictions, labels)
  return Metrics(
      loss=jnp.sum(loss),
      accuracy=jnp.sum(binary_accuracy),
      count=logits.shape[0],
  )

@jax.jit
def train_step(
    state: TrainState,
    batch: Dict[str, Array],
    rngs: Dict[str, Any],
) -> Tuple[TrainState, Dict]:
    
  """Train for a single step."""
  # Make sure to get a new RNG at every step.
  step = state.step
  rngs = {name: jax.random.fold_in(rng, step) for name, rng in rngs.items()}

  def loss_fn(params):
    variables = {'params': params}
    logits = state.apply_fn(
        variables,
        batch['token_ids'],
        batch['length'],
        deterministic=False,
        rngs=rngs,
    )

    labels = batch['label']
    if labels.ndim == 1:
      labels = jnp.expand_dims(labels, 1)
    loss = jnp.mean(
        sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    )
    return loss, logits

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  value, grads = grad_fn(state.params)
  (_, logits) = value

  new_state = state.apply_gradients(grads=grads)
  metrics = compute_metrics(labels=batch['label'], logits=logits)
  return new_state, metrics