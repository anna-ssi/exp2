import wandb
import logging

import numpy as np
import jax
import jax.numpy as jnp

import optax
import haiku as hk
from flax import struct

from typing import Any, Dict, Tuple, Optional, Callable, Sequence
from tqdm import tqdm

from src.models.losses import *
from src.models.eegnet import TrainState

Array = jnp.ndarray
logger = logging.getLogger("EXP2")


class Metrics(struct.PyTreeNode):
    """Computed metrics."""
    loss: float
    accuracy: float

@jax.jit
def compute_metrics(labels: Array, logits: Array) -> Metrics:
    """Computes the metrics, summed across the batch if a batch is provided."""
    loss = optax.sigmoid_binary_cross_entropy(
                logits, labels).mean()
    
    predictions = jnp.argmax(logits, axis=-1)
    binary_accuracy = jnp.mean(predictions == labels)

    return Metrics(
        loss=loss,
        accuracy=binary_accuracy
    )


def normalize_batch_metrics(batch_metrics: Sequence[Metrics]) -> Metrics:
    """Consolidates and normalizes a list of per-batch metrics dicts."""
    # Here we sum the metrics that were already summed per batch.
    total_loss = np.mean([metrics.loss for metrics in batch_metrics])
    total_accuracy = np.mean(
        [metrics.accuracy for metrics in batch_metrics])

    return Metrics(
        loss=total_loss.item(), accuracy=total_accuracy.item()
    )


def train_epoch(
    optimizer: optax.GradientTransformation,
    network: Any,
    state: TrainState,
    train_batches,
    epoch: int,
) -> Tuple[TrainState, Metrics]:
    """Train for a single epoch."""

    @jax.jit
    def train_step(eeg: Array, labels: Array):
        """Train for a single step."""

        def loss_fn(params):
            logits = network.apply(params, eeg)
            # TODO: change loss function
            loss = optax.sigmoid_binary_cross_entropy(logits, labels).mean()
            return loss, logits

        grad_fn = jax.grad(loss_fn, has_aux=True, allow_int=True)
        grads, logits = grad_fn(state.params)
        updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)

        new_state = TrainState(params=params, opt_state=opt_state)
        metrics = compute_metrics(labels, logits)
        return new_state, metrics

    batch_metrics = []
    for eeg, labels in train_batches:
        eeg = jax.tree_map(lambda x: x._numpy(), eeg)
        labels = jax.tree_map(lambda x: x._numpy(), labels)
        state, metrics = train_step(eeg, labels)

        batch_metrics.append(metrics)

    # Compute the metrics for this epoch.
    batch_metrics = jax.device_get(batch_metrics)
    metrics = normalize_batch_metrics(batch_metrics)

    print(
        f'Train {epoch}: loss {metrics.loss:.4f} accuracy {(metrics.accuracy * 100):.2f}')
    return state, metrics


def evaluate_model(
    network: Any,
    state: TrainState,
    batches,
    epoch: int
) -> Metrics:
    """Evaluate a model on a dataset."""

    def eval_step(eeg: Array, labels: Array
                  ) -> Metrics:
        """Evaluate for a single step. Model should be in deterministic mode."""
        logits = network.apply(state.params, eeg)
        metrics = compute_metrics(labels=labels, logits=logits)
        return metrics

    batch_metrics = []
    for eeg, labels in batches:
        eeg = jax.tree_map(lambda x: x._numpy(), eeg)
        labels = jax.tree_map(lambda x: x._numpy(), labels)
        
        metrics = eval_step(eeg, labels)
        batch_metrics.append(metrics)

    batch_metrics = jax.device_get(batch_metrics)
    metrics = normalize_batch_metrics(batch_metrics)

    print(
        f'Eval {epoch}: loss {metrics.loss:.4f} accuracy {(metrics.accuracy * 100):.2f}')
    return metrics


def train_and_evaluate(
    network,
    optimizer,
    params,
    train_batches,
    test_batches,
    state
    ) -> TrainState:
    """Execute model training and evaluation loop.

    Args:
      config: Hyperparameter configuration for training and evaluation.
      workdir: Directory where the tensorboard summaries are written to.
    Returns:
      The final train state that includes the trained parameters.
    """
    # connecting wandb
    # wandb.init(project="exp2", config=params)

    # Main training loop.
    print('Starting training...')
    for epoch in tqdm(range(1, params.epochs + 1), total=params.epochs, desc='Epochs: '):
        # Train for one epoch.
        state, train_metrics = train_epoch(
            optimizer, network, state, train_batches, epoch
        )

        # Evaluate current model on the validation data.
        eval_metrics = evaluate_model(
            network, state, test_batches, epoch)

        # Write metrics to TensorBoard.
        log = {'train_loss': train_metrics.loss,
               'train_accuracy': train_metrics.accuracy * 100,
               'eval_loss': eval_metrics.loss,
               'eval_accuracy': eval_metrics.accuracy * 100}
        # wandb.log(log, epoch)

    return state
