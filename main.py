import json
import argparse

import jax
import jax.numpy as jnp
from flax.training import checkpoints

from src.models.network import build_net
from src.utils.config_loader import ConfigLoader
from src.utils.load_dataset import EEGDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp', type=str, required=True)
    parser.add_argument('--save_path', type=str, default='./results/')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--silent', action='store_true', default=False)
    parser.add_argument('--gpu', action='store_true', default=False)

    args = parser.parse_args()
    dataset = EEGDataset(args.data_path)
    params = ConfigLoader(json.load(open(args.exp, 'r')))
    
    eeg, labels = dataset.get_batch(1)
    x = jnp.expand_dims(eeg[0], axis=0) 
    y = labels[0]
    
    rng = jax.random.PRNGKey(params.seed)
    train_state, carry = build_net(x.shape, params, rng)
    
    out, carry = train_state.apply_fn(train_state.params, x, carry)
    print(out)
    
    # saving checkpoint
    checkpoints.save_checkpoint(args.checkpoint_path, train_state, {'step': 0})