import wandb
import logging

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training import train_state

from typing import Any, Dict, Tuple, Optional, Callable, Sequence
from tqdm import tqdm

Array = jnp.ndarray
TrainState = train_state.TrainState

logger = logging.getLogger("EXP2")


class Metrics(struct.PyTreeNode):
    """Computed metrics."""

    loss: float
    accuracy: float
    count: Optional[int] = None


def compute_metrics(loss, labels: Array, logits: Array, threshold: float = 0.5) -> Metrics:
    """Computes the metrics, summed across the batch if a batch is provided."""
    if labels.ndim == 1:
        labels = jnp.expand_dims(labels, 1)

    # binary_predictions = logits >= threshold
    # binary_accuracy = jnp.equal(binary_predictions, labels)
    binary_accuracy = jnp.argmax(logits, -1) == labels

    return Metrics(
        loss=jnp.sum(loss) if loss is not None else 0,
        accuracy=jnp.sum(binary_accuracy),
        count=logits.shape[0],
    )


def normalize_batch_metrics(batch_metrics: Sequence[Metrics]) -> Metrics:
    """Consolidates and normalizes a list of per-batch metrics dicts."""
    # Here we sum the metrics that were already summed per batch.
    total = np.sum([metrics.count for metrics in batch_metrics])
    total_loss = np.sum([metrics.loss for metrics in batch_metrics]) / total
    total_accuracy = np.sum(
        [metrics.accuracy for metrics in batch_metrics]) / total
    # Divide each metric by the total number of items in the data set.
    return Metrics(
        loss=total_loss.item(), accuracy=total_accuracy.item()
    )


def train_step(
    state: TrainState,
    eeg: Array,
    labels: Array,
    rngs: Dict[str, Any],
    carry: Optional[Any] = None,
) -> Tuple[TrainState, Dict]:
    """Train for a single step."""
    # Make sure to get a new RNG at every step.
    step = state.step
    rngs = {name: jax.random.fold_in(rng, step) for name, rng in rngs.items()}

    def loss_fn(params, eeg, labels, carry):
        logits, carry = state.apply_fn(
            params,
            eeg,
            carry
        )

        if labels.ndim == 1:
            labels = jnp.expand_dims(labels, 1)

        loss = optax.sigmoid_binary_cross_entropy(logits, labels).mean()
        metrics = compute_metrics(loss, labels, logits)
        return loss, (metrics, carry)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    value, grads = grad_fn(state.params, eeg, labels, carry)
    print(grads)
    metrics, carry = value[1]

    new_state = state.apply_gradients(grads=grads)
    return new_state, metrics, carry


def eval_step(
    state: TrainState, eeg: Array, labels: Array, rngs: Dict[str, Any], carry: Optional[Any] = None
) -> Metrics:
    """Evaluate for a single step. Model should be in deterministic mode."""
    logits, _ = state.apply_fn(
        state.params,
        eeg,
        carry
    )
    metrics = compute_metrics(None, labels=labels, logits=logits)
    return metrics


def evaluate_model(
    eval_step_fn: Callable[..., Any],
    state: TrainState,
    batches,
    epoch: int,
    rngs: Optional[Dict[str, Any]] = None,
    carry: Optional[Any] = None
) -> Metrics:
    """Evaluate a model on a dataset."""
    batch_metrics = []
    for i, (eeg, labels) in enumerate(batches):
        eeg = jax.tree_map(lambda x: x._numpy(), eeg)
        labels = jax.tree_map(lambda x: x._numpy(), labels)
        if rngs is not None:  # New RNG for each step.
            rngs = {name: jax.random.fold_in(rng, i)
                    for name, rng in rngs.items()}

        metrics = eval_step_fn(state, eeg, labels, rngs, carry)
        batch_metrics.append(metrics)

    batch_metrics = jax.device_get(batch_metrics)
    metrics = normalize_batch_metrics(batch_metrics)

    logger.info(
        'eval  epoch %03d loss %.4f accuracy %.2f',
        epoch,
        metrics.loss,
        metrics.accuracy * 100,
    )
    return metrics


def train_epoch(
    train_step_fn: Callable[..., Tuple[TrainState, Metrics]],
    state: TrainState,
    train_batches,
    epoch: int,
    rngs: Optional[Dict[str, Any]] = None,
    carry: Optional[Any] = None

) -> Tuple[TrainState, Metrics]:
    """Train for a single epoch."""
    batch_metrics = []
    for eeg, labels in train_batches:
        eeg = jax.tree_map(lambda x: x._numpy(), eeg)
        labels = jax.tree_map(lambda x: x._numpy(), labels)
        state, metrics, carry = train_step_fn(state, eeg, labels, rngs, carry)
        batch_metrics.append(metrics)

    # Compute the metrics for this epoch.
    batch_metrics = jax.device_get(batch_metrics)
    metrics = normalize_batch_metrics(batch_metrics)

    logger.info(
        'train epoch %03d \n loss %.4f accuracy %.2f',
        epoch,
        metrics.loss,
        metrics.accuracy * 100,
    )

    return state, metrics, carry


def train_and_evaluate(
    params,
    train_batches,
    test_batches,
    state,
    carry,
    rng
) -> TrainState:
    """Execute model training and evaluation loop.

    Args:
      config: Hyperparameter configuration for training and evaluation.
      workdir: Directory where the tensorboard summaries are written to.
    Returns:
      The final train state that includes the trained parameters.
    """
    # connecting wandb
    wandb.init(project="exp2", config=params)

    # Compile step functions.
    # train_step_fn = jax.jit(train_step)
    # eval_step_fn = jax.jit(eval_step)
    
    train_step_fn = train_step
    eval_step_fn = eval_step

    # Main training loop.
    print('Starting training...')
    for epoch in tqdm(range(1, params.epochs + 1), total=params.epochs, desc='Epochs: '):
        # Train for one epoch.
        rng, epoch_rng = jax.random.split(rng)
        rngs = {'dropout': epoch_rng}
        state, train_metrics, carry = train_epoch(
            train_step_fn, state, train_batches, epoch, rngs, carry=carry
        )

        # Evaluate current model on the validation data.
        eval_metrics = evaluate_model(
            eval_step_fn, state, test_batches, epoch, carry=carry)

        # Write metrics to TensorBoard.
        log = {'train_loss': train_metrics.loss,
               'train_accuracy': train_metrics.accuracy * 100,
               'eval_loss': eval_metrics.loss,
               'eval_accuracy': eval_metrics.accuracy * 100}
        wandb.log(log, epoch)

    return state
