import numpy as np
import functools
from typing import Any, Dict, List, Tuple

import optax
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state

TrainState = train_state.TrainState

def flatten(x):
    return jnp.reshape(x, (x.shape[0], -1))

def create_train_state(config, params, model):
  """Create initial training state."""

  tx = optax.chain(
      optax.adam(learning_rate=config.lr, b1=config.beta1, b2=config.beta2),
      optax.add_decayed_weights(weight_decay=config.weight_decay),
  )
  state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
  return state

class LSTM(nn.Module):
  hidden_size: int

  @functools.partial(
      nn.transforms.scan,
      variable_broadcast='params',
      in_axes=1,
      out_axes=1,
      split_rngs={'params': False},
  )
  @nn.compact
  def __call__(self, x, carry):
    w_init = nn.initializers.orthogonal(np.sqrt(2))
    b_init = nn.initializers.constant(0)
    
    carry, y = nn.OptimizedLSTMCell(self.hidden_size)(carry, x)
    y = flatten(y)
    logits = nn.Dense(2, kernel_init=w_init, bias_init=b_init)(y)
    return carry, logits

  def initialize_carry(self, input_shape):
    # Use fixed random key since default state init fn is just zeros.
    return nn.OptimizedLSTMCell(self.hidden_size, parent=None).initialize_carry(
        jax.random.key(0), input_shape
    )
    
class CNN(nn.Module):
  """A simple unidirectional LSTM."""

  layers: List[int]

  @nn.compact
  def __call__(self, x, carry=None):
    w_init = nn.initializers.orthogonal(np.sqrt(2))
    b_init = nn.initializers.constant(0)

    for width in self.layers:
        x = nn.Conv(width, (3, 3), kernel_init=w_init, bias_init=b_init)(x)
        x = nn.relu(x)
        
    x = flatten(x)
    logits = nn.Dense(2, kernel_init=w_init, bias_init=b_init)(x)
    return logits, carry

class MLP(nn.Module):
  """A simple unidirectional LSTM."""

  layers: List[int]

  @nn.compact
  def __call__(self, x, carry=None):
    w_init = nn.initializers.orthogonal(np.sqrt(2))
    b_init = nn.initializers.constant(0)
    
    x = flatten(x)
    for width in self.layers:
        x = nn.Dense(width, kernel_init=w_init, bias_init=b_init)(x)
        x = nn.relu(x)
        
    logits = nn.Dense(2, kernel_init=w_init, bias_init=b_init)(x)
    return logits, carry


def build_net(inputs: Tuple, params, rng: Any):
    def _inner():
        type = params.network.type
        hidden = params.network.hidden
        size = params.network.size
        
        layer_size = [hidden] * size

        if type == 'mlp':
            layers = MLP(layer_size)

        elif type  == 'cnn':
            layers = CNN(layer_size)
        
        elif type  == 'lstm':
            layers = LSTM(layer_size[0])
            
        return layers

    network = _inner()
    sample_input = jnp.zeros((1,) + tuple(inputs))
    carry = network.initialize_carry(inputs) if params.network.type == 'lstm' else None
    net_params = network.init(rng, sample_input, carry)
    
    train_state = create_train_state(params.optim, net_params, network)
    return train_state, carry


