import numpy as np
import functools
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn

def flatten(x):
    return jnp.reshape(x, (x.shape[0], -1))

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

    for width in self.layers:
        x = nn.Dense(width, kernel_init=w_init, bias_init=b_init)(x)
        x = nn.relu(x)
        
    logits = nn.Dense(2, kernel_init=w_init, bias_init=b_init)(x)
    return logits, carry


def build_net(inputs: Tuple, hypers: Dict[str, Any], rng: Any):
    def _inner():
        name = hypers['type']
        size = hypers['size']
        hidden = hypers['hidden']
        
        layer_size = [hidden] * size
        print(f'Building {name} with {layer_size} layers')

        if name == 'mlp':
            layers = MLP(layer_size)

        elif name == 'cnn':
            layers = CNN(layer_size)
        
        elif name == 'lstm':
            layers = LSTM(layer_size[0])
            
        return layers

    network = _inner()
    sample_input = jnp.zeros((1,) + tuple(inputs))
    carry = network.initialize_carry(inputs) if params['type'] == 'lstm' else None
    net_params = network.init(rng, sample_input, carry)
        
    return network, net_params, carry


if __name__ == '__main__':
    params = {
        'type': 'cnn',
        'size': 3,
        'hidden': 13
    }
    
    # x = jnp.ones((1, 10)) # mlps
    # x = jnp.ones((1, 1, 10)) # lstms
    x = jnp.ones((1, 10, 10)) # cnns
    rng = jax.random.PRNGKey(1)
    net, net_params, carry = build_net(x.shape, params, rng)
    
    out, carry = net.apply(net_params, x, carry)
    print(out)