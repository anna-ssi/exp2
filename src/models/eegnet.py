from typing import Any, Tuple, NamedTuple

import optax
import jax
import jax.numpy as jnp
import haiku as hk

SHOW_MODEL_ARCH = False

F1 = 8
D = 2
F2 = F1 * D
KernelLength = 64

class TrainState(NamedTuple):
    """Training state."""
    params: Any
    opt_state: optax.OptState


def eeg_net(inputs: Any, is_training: bool = True):
    layers = hk.Sequential([
        hk.Conv2D(F1, (1, KernelLength), padding='SAME'),
        # lambda x: hk.BatchNorm(False, False, 0.99)(x, is_training=is_training),

        hk.DepthwiseConv2D(D, (KernelLength, 1)),
        # lambda x: hk.BatchNorm(False, False, 0.99)(x, is_training=is_training),
        jax.nn.elu,
        hk.AvgPool(1, 4, 'VALID'),
        lambda x: hk.dropout(hk.next_rng_key(), 0.25, x),

        hk.SeparableDepthwiseConv2D(F2, (1, 16), padding='SAME'),
        # lambda x: hk.BatchNorm(False, False, 0.99)(x, is_training=is_training),
        jax.nn.elu,
        hk.AvgPool(1, 8, 'VALID'),
        lambda x: hk.dropout(hk.next_rng_key(), 0.25, x),

        hk.Flatten(),
        hk.Linear(2)
    ])
    return layers(inputs)


def build_net(inputs: Tuple, params, rng: Any):
    network = hk.without_apply_rng(hk.transform(eeg_net))

    sample_input = jnp.zeros((1, ) + tuple(inputs))
    net_params = network.init(rng, sample_input)

    if SHOW_MODEL_ARCH:
        print(hk.experimental.tabulate(network)(sample_input))

    optimizer = optax.adam(learning_rate=params.optim.lr)
    opt_state = optimizer.init(net_params)

    train_state = TrainState(params=net_params, opt_state=opt_state)
    return train_state, network, optimizer
