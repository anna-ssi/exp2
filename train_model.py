import os
import argparse

import numpy as np
import torch
from torch import nn

from tqdm import tqdm
from torcheeg.models import BUNet

from src.utils.dataset import EEGDatasetActionTrajectory
from src.utils.helper import *

import warnings
warnings.filterwarnings('ignore')


def evaluate(model, criterion, dataset):
    running_loss = 0

    with torch.no_grad():
        for eegs in dataset.data:
            for t in range(len(eegs)):
                if t >= len(eegs) - 1:
                    continue
                eeg, next_eeg = eegs[:, :, t], eegs[:, :, t+1]
                eeg, next_eeg = torch.Tensor(eeg).unsqueeze(
                    0), torch.Tensor(next_eeg).unsqueeze(0)
                eeg, next_eeg = eeg.to(device), next_eeg.to(device)

                t = torch.randint(low=1, high=1000, size=(1, ))
                pred = model(eeg, t)
                loss = criterion(pred, next_eeg)
                running_loss += loss.item()

    dataset_len = sum([len(label) for label in dataset.labels])
    return running_loss / dataset_len


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str,
                        default='./checkpoints/model/')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--silent', action='store_true', default=False)
    parser.add_argument('--gpu', action='store_true', default=False)
    parser.add_argument('--balance', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--kfold', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--data', type=str, default='Safe',
                        choices=['Safe', 'Risk'])

    args = parser.parse_args()

    device = torch.device('cuda' if args.gpu else 'cpu')
    print("Device: ", device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    participants = np.arange(2, 53)

    chk_path = os.path.join(args.checkpoint_path, f'{args.seed}')
    model_save_path = os.path.join(chk_path, f'{args.data}.pt')
    results_data_save_path = os.path.join(
        chk_path, f'{args.data}.txt')

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

    model = BUNet(grid_size=(61, 601), in_channels=1).to(device)

    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    train_len = sum([len(label) for label in train_set.labels])
    for epoch in tqdm(range(1, args.epochs + 1), total=args.epochs, desc='Epochs: '):
        running_loss = 0
        for eegs in train_set.data:
            for t in range(len(eegs)):
                if t >= len(eegs) - 1:
                    continue

                eeg, next_eeg = eegs[:, :, t], eegs[:, :, t+1]
                eeg, next_eeg = torch.Tensor(eeg).unsqueeze(
                    0).unsqueeze(0), torch.Tensor(next_eeg).unsqueeze(0).unsqueeze(0)
                eeg, next_eeg = eeg.to(device), next_eeg.to(device)
                print(eeg.shape, next_eeg.shape)

                optimizer.zero_grad()

                t = torch.randint(low=1, high=1000, size=(1, ))
                fake_eeg = model(eeg, t)
                print(fake_eeg.shape)
                loss = criterion(fake_eeg, next_eeg)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

        # Validation accuracy
        train_loss = running_loss / train_len
        test_loss = evaluate(model, args.batch_size, test_set)

        print("Training Loss ", train_loss)
        print("Test Loss ", test_loss)

        print("Saving the model...")
        torch.save(model.state_dict(), model_save_path)
        # Save results
        results_file.write(
            f"Epoch: {epoch} - train: {train_loss}, test: {test_loss}\n\n")
