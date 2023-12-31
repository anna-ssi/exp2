import os
import argparse

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, accuracy_score

from src.utils.dataset import EEGDatasetAction
from src.utils.helper import *

from torcheeg.models import EEGNet, LSTM, DGCNN, VanillaTransformer

import warnings
warnings.filterwarnings('ignore')


def evaluate(model, data_loader):
    accuracy = 0
    precision = 0
    recall = 0
    f_score = 0

    predictions = []
    with torch.no_grad():
        for data in data_loader:
            eeg, labels = data
            eeg = eeg.to(device)

            pred = model(eeg).detach().cpu().numpy()
            pred = np.argmax(pred, axis=1)
            predictions.append(pred)
            labels = labels.numpy().squeeze()

            # metrics
            accuracy += accuracy_score(labels, pred)
            precision += precision_score(labels, pred, average='macro')
            recall += recall_score(labels, pred, average='macro')
            f_score += 2 * (precision * recall) / (precision + recall)

    predictions = np.concatenate(predictions, axis=0)
    counts = np.unique(predictions, return_counts=True)
    counts_str = f'Data stats: '
    for label, count in zip(*counts):
        counts_str += f'label {label}: {count}, '
    print(counts_str[:-2])

    return {'acc': (accuracy / len(data_loader)) * 100,
            'recall': recall / len(data_loader),
            'precision': precision / len(data_loader),
            'fmeasure': f_score / len(data_loader)
            }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str,
                        default='./checkpoints/actions/')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--silent', action='store_true', default=False)
    parser.add_argument('--gpu', action='store_true', default=False)
    parser.add_argument('--balance', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--kfold', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--net_type', type=str, default='eeg',
                        choices=['eeg', 'lstm', 'dcgnn', 'vit'])
    parser.add_argument('--data', type=str, default='Safe',
                        choices=['Safe', 'Risk', 'All'])

    args = parser.parse_args()

    device = torch.device('cuda' if args.gpu else 'cpu')
    print("Device: ", device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    participants = np.arange(2, 53)

    avg_best_acc, avg_best_prec, avg_best_recall, avg_best_f = 0, 0, 0, 0
    for idx in range(args.kfold):
        chk_path = os.path.join(args.checkpoint_path,
                                f'{args.seed}', f'{args.net_type}', f'{args.data}')

        model_save_path = os.path.join(
            chk_path, f'{args.data}_{args.balance}_{idx}.pt')
        results_data_save_path = os.path.join(
            chk_path, f'{args.data}_{args.balance}_{idx}.txt')

        if not os.path.exists(chk_path):
            os.makedirs(chk_path)
        results_file = open(results_data_save_path, 'w')

        test_participants = np.random.choice(participants, 5, replace=False)
        train_participants = np.setdiff1d(participants, test_participants)
        print("Test participants: ", test_participants)

        # Loading dataset
        train_set = EEGDatasetAction(args.data_path, train=True, par_numbers=train_participants,
                                     type=args.data, net_type=args.net_type,
                                     balance=args.balance)
        test_set = EEGDatasetAction(args.data_path, train=False, par_numbers=test_participants,
                                    type=args.data, net_type=args.net_type,
                                    balance=args.balance)
        train_set.get_data_stats()
        test_set.get_data_stats()

        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(
            test_set, batch_size=args.batch_size, shuffle=False)

        # Loading model
        if args.net_type == 'eeg':
            model = EEGNet(chunk_size=601, num_electrodes=61,
                           num_classes=4).to(device)
        elif args.net_type == 'lstm':
            model = LSTM(num_electrodes=61, num_classes=4).to(device)
        elif args.net_type == 'dcgnn':
            model = DGCNN(in_channels=601, num_electrodes=61,
                          num_classes=4).to(device)
        else:
            model = VanillaTransformer(num_electrodes=61, chunk_size=601,
                                       t_patch_size=601, num_classes=4).to(device)

        if os.path.exists(model_save_path):
            model.load_state_dict(torch.load(model_save_path))

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        best_acc, best_prec, best_recall, best_f = 0, 0, 0, 0
        for epoch in tqdm(range(1, args.epochs + 1), total=args.epochs, desc='Epochs: '):
            running_loss = 0
            for data in train_loader:
                eeg, labels = data
                eeg, labels = eeg.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = model(eeg)
                loss = criterion(outputs, labels.squeeze())

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # Validation accuracy
            train_results = evaluate(model, train_loader)
            test_results = evaluate(model, test_loader)

            if test_results['precision'] > best_prec:
                # Save model
                print("Saving the model...")
                torch.save(model.state_dict(), model_save_path)

                best_prec = test_results['precision']
                best_acc = test_results['acc']
                best_recall = test_results['recall']
                best_f = test_results['fmeasure']

            # Save results
            results_file.write(
                f"Epoch: {epoch}\nTrain: {dict_to_string(train_results)}\nTest: {dict_to_string(test_results)}\n\n")

        print("Best Test Accuracy: ", best_acc, "Best Test Precision: ",
              best_prec, "Best Test Recall: ", best_recall, "Best Test F1: ", best_f)
        avg_best_acc += best_acc
        avg_best_prec += best_prec
        avg_best_recall += best_recall
        avg_best_f += best_f

        del model
        del train_set
        del test_set
        del train_loader
        del test_loader
        torch.cuda.empty_cache()

    print("Avergae Test Accuracy: ", avg_best_acc / args.kfold)
    print("Average Test Precision: ", avg_best_prec / args.kfold)
    print("Average Test Recall: ", avg_best_recall / args.kfold)
    print("Average Test F1: ", avg_best_f / args.kfold)