import os
import json
import argparse

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import KFold

from src.utils.config_loader import ConfigLoader
from src.utils.dataset import EEGDataset
from src.utils.helper import *

from torcheeg.models import EEGNet, LSTM, DGCNN

import warnings
warnings.filterwarnings('ignore')


def evaluate(model, data_loader):
    accuracy = 0
    precision = 0
    recall = 0
    f_score = 0

    for data in data_loader:
        eeg, labels = data
        eeg = eeg.to(device)
        pred = model(eeg).detach().cpu().numpy()
        pred = np.argmax(pred, axis=1).reshape(-1, 1)

        # metrics
        accuracy += accuracy_score(labels, pred)
        precision += precision_score(labels, pred)
        recall += recall_score(labels, pred)
        f_score += 2 * (precision * recall) / (precision + recall)

    return {'acc': (accuracy / len(data_loader)) * 100,
            'recall': recall / len(data_loader),
            'precision': precision / len(data_loader),
            'fmeasure': f_score / len(data_loader)
            }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str,
                        default='./checkpoints/')
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
    results_data_save_path = os.path.join(chk_path, f'{args.data}_{args.balance}.txt')
    print(results_data_save_path)

    if not os.path.exists(chk_path):
        os.makedirs(chk_path)
    results_file = open(results_data_save_path, 'w')

    # Loading dataset
    dataset = EEGDataset(args.data_path, type=args.data,
                         balanced=args.balance, net_type=params.net_type)
    train_size = int(len(dataset) * (1 - params.test_size))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])
    

    dataset.get_data_stats()
    print(f'Train: {get_data_stats(train_set)}')
    print(f'Test: {get_data_stats(test_set)}')

    print("Dataset length: ", len(dataset))
    print("Train length: ", len(train_set))
    print("Test length: ", len(test_set))

    train_loader = DataLoader(
        train_set, batch_size=params.batch_size, shuffle=True)
    test_loader = DataLoader(
        test_set, batch_size=params.batch_size, shuffle=False)

    # Loading model
    if params.net_type == 'eeg':
        model = EEGNet(chunk_size=601, num_electrodes=61,
                       num_classes=2).to(device)
    elif params.net_type == 'lstm':
        model = LSTM(num_electrodes=61, num_classes=2).to(device)
    else:
        model = DGCNN(in_channels=601, num_electrodes=61, num_classes=2).to(device)

    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.optim.lr)

best_acc = 0
best_results = None
for epoch in tqdm(range(1, params.epochs + 1), total=params.epochs, desc='Epochs: '):
    running_loss = 0
    for data in train_loader:
        eeg, labels = data
        eeg, labels = eeg.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(eeg)
        loss = criterion(outputs, labels.squeeze(1))

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validation accuracy
    params = ["acc", "auc", "fmeasure"]
    train_results = evaluate(model, train_loader)
    test_results = evaluate(model, test_loader)

    print("Training Loss ", running_loss / len(train_loader))
    print("Train - ", train_results)
    print("Test - ", test_results)

    if test_results['acc'] > best_acc:
        # Save model
        print("Saving the model...")
        torch.save(model.state_dict(), model_save_path)
        best_acc = test_results['acc']
        best_results = test_results

    # Save results
    results_file.write(
        f"Epoch: {epoch}\nTrain: {dict_to_string(train_results)}\nTest: {dict_to_string(test_results)}\n\n")

print("Best Test Accuracy: ", best_acc)
print("Best Results: ", best_results)