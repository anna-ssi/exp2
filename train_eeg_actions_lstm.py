import os
import json
import argparse

import numpy as np
import torch
from torch import nn

from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, accuracy_score

from src.models.lstm import LSTM
from src.utils.config_loader import ConfigLoader
from src.utils.dataset import EEGDatasetActionTrajectory
from src.utils.helper import *


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
            precision += precision_score(traj_labels, traj_preds, average='macro')
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
                        default='./checkpoints/actions/')
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
    train_set = EEGDatasetActionTrajectory(args.data_path, train=True,
                                           type=args.data, net_type=params.net_type,
                                           balance=args.balance)
    test_set = EEGDatasetActionTrajectory(args.data_path, train=False,
                                          type=args.data, net_type=params.net_type,
                                          balance=args.balance)

    model = LSTM(num_electrodes=61, num_classes=4).to(device)

    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.optim.lr)

best_acc = 0
best_results = None
train_len = sum([len(label) for label in train_set.labels])
for epoch in tqdm(range(1, params.epochs + 1), total=params.epochs, desc='Epochs: '):
    running_loss = 0
    for eegs, labels in zip(train_set.data, train_set.labels):
        h_n = torch.zeros(2, params.batch_size, model.hid_channels).to(device)
        c_n = torch.zeros(2, params.batch_size, model.hid_channels).to(device)

        for i in range(len(labels)):
            eeg, label = eegs[:, :, i], labels[i]
            eeg, label = torch.Tensor(eeg).unsqueeze(
                0), torch.LongTensor([label])
            eeg, label = eeg.to(device), label.to(device)

            optimizer.zero_grad()

            output, h_n, c_n = model(eeg, h_n, c_n)
            loss = criterion(output, label)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    # Validation accuracy
    train_results = evaluate(model, params.batch_size, train_set)
    test_results = evaluate(model, params.batch_size, test_set)

    print("Training Loss ", running_loss / train_len)
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
