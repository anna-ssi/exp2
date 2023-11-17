import os
import json
import argparse

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, accuracy_score

from src.utils.config_loader import ConfigLoader
from src.utils.dataset import EEGDataset

from torcheeg.models import EEGNet, LSTM

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
    parser.add_argument('--data', type=str, default='Safe',
                        choices=['Safe', 'Risk', 'All']) 

    args = parser.parse_args()
    
    device = torch.device('cuda' if args.gpu else 'cpu')
    print("Device: ", device)

    params = ConfigLoader(json.load(open(args.exp, 'r')))
    chk_path = os.path.join(args.checkpoint_path, f'{params.seed}')
    save_path = os.path.join(chk_path, f'{args.data}.pt')
    if not os.path.exists(chk_path):
        os.makedirs(chk_path)

    # Loading dataset
    dataset = EEGDataset(args.data_path, type=args.data, balanced=True)
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
    model = EEGNet(chunk_size=601, num_electrodes=61, num_classes=2).to(device)
    # model = LSTM(num_electrodes=61, num_classes=2).to(device)
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.optim.lr)

best_acc = 0
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
    
    print("Training Loss ", running_loss / len(train_loader))
    print("Train - ", evaluate(model, train_loader))
    
    test_results = evaluate(model, test_loader)
    print("Test - ", test_results)
    
    if test_results['acc'] > best_acc:
        # Save model
        print("Saving the model...")
        torch.save(model.state_dict(), os.path.join(chk_path, f'{args.data}.pt'))
        best_acc = test_results['acc']
