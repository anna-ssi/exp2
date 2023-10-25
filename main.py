import json
import argparse

import jax
from flax.training import checkpoints

from src.models.network import build_net
from src.models.train import train_and_evaluate
from src.utils.config_loader import ConfigLoader
from src.utils.load_dataset import TrainTestSplit

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp', type=str, required=True)
    parser.add_argument('--save_path', type=str, default='./results/')
    parser.add_argument('--checkpoint_path', type=str,
                        default='./checkpoints/')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--silent', action='store_true', default=False)
    parser.add_argument('--gpu', action='store_true', default=False)

    args = parser.parse_args()
    params = ConfigLoader(json.load(open(args.exp, 'r')))
    dataset = TrainTestSplit(args.data_path, params.test_size)
    train_data, test_data = dataset.split()

    eeg, _ = train_data.batch(1).as_numpy_iterator().next()
    rng = jax.random.PRNGKey(params.seed)
    train_state, carry = build_net(eeg.shape, params, rng)

    train_batches = train_data.batch(params.batch_size)
    test_batches = test_data.batch(params.batch_size)

    train_and_evaluate(params, train_batches, test_batches,
                       train_state, carry, rng)

    # saving checkpoint
    # checkpoints.save_checkpoint(args.checkpoint_path, train_state, {'step': 0}, overwrite=True)
