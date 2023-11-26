import os
import json
import argparse

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, accuracy_score

from src.utils.config_loader import ConfigLoader
from src.utils.dataset import EEGDatasetAction
from src.utils.helper import *

from torcheeg.models import BEncoder, BDecoder


import warnings
warnings.filterwarnings('ignore')


def evaluate(model, batch_size, dataset):
    accuracy = 0
    precision = 0
    recall = 0
    f_score = 0

    predictions = []

    with torch.no_grad():
        for eegs, labels in zip(dataset.data, dataset.labels):
            h_n = torch.zeros(2, batch_size, model.hid_channels).to(device)
            c_n = torch.zeros(2, batch_size, model.hid_channels).to(device)

            traj_preds, traj_labels = [], []
            for i in range(len(labels)):
                eeg, label = eegs[:, :, i], labels[i]
                eeg = torch.Tensor(eeg).unsqueeze(0)
                eeg = eeg.to(device)

                pred, h_n, c_n = model(eeg, h_n, c_n)
                pred = pred.cpu().numpy()
                pred = np.argmax(pred, axis=1)
                predictions.append(pred)

                traj_preds.append(pred)
                traj_labels.append(label)

            # metrics
            accuracy += accuracy_score(traj_labels, traj_preds)
            precision += precision_score(traj_labels,
                                         traj_preds, average='macro')
            recall += recall_score(traj_labels, traj_preds, average='macro')
            f_score += 2 * (precision * recall) / (precision + recall)

    predictions = np.concatenate(predictions, axis=0)
    counts = np.unique(predictions, return_counts=True)
    counts_str = f'Data stats: '
    for label, count in zip(*counts):
        counts_str += f'label {label}: {count}, '
    print(counts_str[:-2])

    dataset_len = sum([len(label) for label in dataset.labels])

    return {'acc': (accuracy / dataset_len) * 100,
            'recall': recall / dataset_len,
            'precision': precision / dataset_len,
            'fmeasure': f_score / dataset_len
            }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str,
                        default='./checkpoints/vae/')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--silent', action='store_true', default=False)
    parser.add_argument('--gpu', action='store_true', default=False)
    parser.add_argument('--balance', action='store_true', default=False)
    parser.add_argument('--data', type=str, default='Safe',
                        choices=['Safe', 'Risk', 'All'])

    args = parser.parse_args()

    device = torch.device('cuda' if args.gpu else 'cpu')
    print("Device: ", device)

    params = ConfigLoader(json.load(open(args.exp, 'r')))
    chk_path = os.path.join(args.checkpoint_path,
                            f'{params.seed}', f'{params.net_type}', f'{args.data}')

    model_save_path = os.path.join(chk_path, f'{args.data}_{args.balance}.pt')
    results_data_save_path = os.path.join(
        chk_path, f'{args.data}_{args.balance}.txt')

    if not os.path.exists(chk_path):
        os.makedirs(chk_path)
    results_file = open(results_data_save_path, 'w')

    torch.manual_seed(params.seed)
    np.random.seed(params.seed)

    # Loading dataset
    train_set = EEGDatasetAction(args.data_path, train=True,
                                 type=args.data, net_type=params.net_type,
                                 balance=args.balance)
    test_set = EEGDatasetAction(args.data_path, train=False,
                                type=args.data, net_type=params.net_type,
                                balance=args.balance)

    train_set.get_data_stats()
    test_set.get_data_stats()

    train_loader = DataLoader(
        train_set, batch_size=params.batch_size, shuffle=True)
    test_loader = DataLoader(
        test_set, batch_size=params.batch_size, shuffle=False)
    
    encoder = BEncoder(in_channels=601).to(device)

    # if os.path.exists(model_save_path):
    #     model.load_state_dict(torch.load(model_save_path))

    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=params.optim.lr)

for epoch in tqdm(range(1, params.epochs + 1), total=params.epochs, desc='Epochs: '):
    # running_loss = 0
    for data in train_loader:
        eeg, labels = data
        eeg = eeg.unsqueeze(-1).permute(0, 2, 1, 3)
        eeg, labels = eeg.to(device), labels.to(device)
        print(eeg.shape)

        # optimizer.zero_grad()

        mu, logvar = encoder(eeg)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mu
        print(z.shape)
        exit()
        # loss = criterion(outputs, labels.squeeze())

        # loss.backward()
        # optimizer.step()

        # running_loss += loss.item()

