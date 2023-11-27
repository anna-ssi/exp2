import os
import argparse

import numpy as np
import torch
from torch import nn

from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, accuracy_score

from src.models.lstm import LSTM
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
    parser.add_argument('--checkpoint_path', type=str,
                        default='./checkpoints/actions/')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--silent', action='store_true', default=False)
    parser.add_argument('--gpu', action='store_true', default=False)
    parser.add_argument('--balance', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--kfold', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1)
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
        chk_path = os.path.join(args.checkpoint_path, f'{args.seed}', 'seq')
        model_save_path = os.path.join(chk_path, f'{args.data}_{idx}.pt')
        results_data_save_path = os.path.join(
            chk_path, f'{args.data}_{idx}.txt')

        if not os.path.exists(chk_path):
            os.makedirs(chk_path)
        results_file = open(results_data_save_path, 'w')

        test_participants = np.random.choice(participants, 5, replace=False)
        train_participants = np.setdiff1d(participants, test_participants)
        print("Test participants: ", test_participants)

        # Loading dataset
        train_set = EEGDatasetActionTrajectory(args.data_path, train=True,
                                               par_numbers=train_participants,
                                               type=args.data)
        test_set = EEGDatasetActionTrajectory(args.data_path, train=False,
                                              par_numbers=test_participants,
                                              type=args.data)

        model = LSTM(num_electrodes=61, num_classes=4).to(device)

        if os.path.exists(model_save_path):
            model.load_state_dict(torch.load(model_save_path))

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        best_acc, best_prec, best_recall, best_f = 0, 0, 0, 0
        train_len = sum([len(label) for label in train_set.labels])
        for epoch in tqdm(range(1, args.epochs + 1), total=args.epochs, desc='Epochs: '):
            running_loss = 0
            for eegs, labels in zip(train_set.data, train_set.labels):
                h_n = torch.zeros(2, args.batch_size,
                                  model.hid_channels).to(device)
                c_n = torch.zeros(2, args.batch_size,
                                  model.hid_channels).to(device)

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
            train_results = evaluate(model, args.batch_size, train_set)
            test_results = evaluate(model, args.batch_size, test_set)

            print("Training Loss ", running_loss / train_len)
            print("Train - ", train_results)
            print("Test - ", test_results)

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

        avg_best_acc += best_acc
        avg_best_prec += best_prec
        avg_best_recall += best_recall
        avg_best_f += best_f

        del model
        del train_set
        del test_set
        torch.cuda.empty_cache()

    print("Avergae Test Accuracy: ", avg_best_acc / args.kfold)
    print("Average Test Precision: ", avg_best_prec / args.kfold)
    print("Average Test Recall: ", avg_best_recall / args.kfold)
    print("Average Test F1: ", avg_best_f / args.kfold)
