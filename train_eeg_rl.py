import os
import json
import argparse

import torch

from src.utils.config_loader import ConfigLoader
from src.utils.helper import *

from torcheeg.models import EEGNet, LSTM, DGCNN

import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str,
                        default='./checkpoints/')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--gpu', action='store_true', default=False)
    parser.add_argument('--balance', action='store_true', default=False)
    parser.add_argument('--data', type=str, default='Safe',
                        choices=['Safe', 'Risk', 'All'])
    parser.add_argument('--net_type', type=str, default='eeg',
                        choices=['eeg', 'lstm', 'dgcnn'])

    args = parser.parse_args()

    device = torch.device('cuda' if args.gpu else 'cpu')
    print("Device: ", device)


    # Loading EEG model
    if args.net_type == 'eeg':
        eeg_model = EEGNet(chunk_size=601, num_electrodes=61,
                       num_classes=2).to(device)
    elif args.net_type == 'lstm':
        eeg_model = LSTM(num_electrodes=61, num_classes=2).to(device)
    else:
        eeg_model = DGCNN(in_channels=601, num_electrodes=61,
                      num_classes=2).to(device)

    eeg_model.load_state_dict(torch.load(args.checkpoint_path))

