import os
import json
import argparse

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, accuracy_score

from src_pytorch.utils.config_loader import ConfigLoader
from src_pytorch.utils.dataset import EEGDataset
from src_pytorch.models.eegnet import EEGNet

import warnings
warnings.filterwarnings('ignore')


def evaluate(model, data_loader, data_size):
    accuracy = 0
    precision = 0
    recall = 0
    fmeasure = 0

    for data in data_loader:
        eeg, labels = data
        pred = model(eeg).detach().numpy()
        pred = np.argmax(pred, axis=1)

        # metrics
        accuracy += accuracy_score(labels, pred)
        precision += precision_score(labels, pred)
        recall += recall_score(labels, pred)
        fmeasure += (2*precision*recall / (precision+recall))

    return {'acc': accuracy / data_size,
            'recall': recall / data_size,
            'precision': precision / data_size,
            'fmeasure': fmeasure / data_size
            }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str,
                        default='./checkpoints/')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--silent', action='store_true', default=False)
    parser.add_argument('--gpu', action='store_true', default=False)
    parser.add_argument('--data', type=str, default='Safe',
                        choices=['Safe', 'Risk', 'All']) # TODO: all

    args = parser.parse_args()

    params = ConfigLoader(json.load(open(args.exp, 'r')))
    chk_path = os.path.join(args.checkpoint_path, f'{params.seed}')
    save_path = os.path.join(chk_path, f'{args.data}.pt')

    # Loading dataset
    dataset = EEGDataset(args.data_path, type=args.data)
    train_size = int(len(dataset) * (1 - params.test_size))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    print("Dataset length: ", len(dataset))
    print("Train length: ", len(train_set))
    print("Test length: ", len(test_set))

    train_loader = DataLoader(
        train_set, batch_size=params.batch_size, shuffle=True)
    test_loader = DataLoader(
        test_set, batch_size=params.batch_size, shuffle=False)

    # Loading model
    model = EEGNet()
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.optim.lr)

for epoch in tqdm(range(1, params.epochs + 1), total=params.epochs, desc='Epochs: '):
    running_loss = 0
    for data in train_loader:
        eeg, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(eeg)
        loss = criterion(outputs, labels.squeeze(1))

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validation accuracy
    params = ["acc", "auc", "fmeasure"]

    print("Training Loss ", running_loss / train_size)
    print("Train - ", evaluate(model, train_loader, train_size))
    print("Test - ", evaluate(model, test_loader, test_size))

    # Save model
    if not os.path.exists(chk_path):
        os.makedirs(chk_path)
    torch.save(model.state_dict(), os.path.join(chk_path, f'{args.data}.pt'))
