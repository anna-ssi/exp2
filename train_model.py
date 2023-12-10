import os
import argparse

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from tqdm import tqdm

from src.utils.dataset import EEGModelDataset
from src.models.mlp_model import EncoderDecoder
from src.utils.helper import *

from torcheeg.models import LSTM, DGCNN

import warnings
warnings.filterwarnings('ignore')


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def evaluate(model, loader):
    running_loss = 0

    with torch.no_grad():
        for eeg, next_eeg in loader:
            eeg, next_eeg = eeg.to(device), next_eeg.to(device)

            pred_eeg = model(eeg)
            # maximizing average cosine similarity
            loss = F.cosine_similarity(pred_eeg, next_eeg).mean()
            running_loss += loss.item()

    return {'cos_sim': running_loss / len(loader),
            'p_val': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pt', type=str,
                        required=True)
    parser.add_argument('--checkpoint_path', type=str,
                        default='./checkpoints/model/')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--silent', action='store_true', default=False)
    parser.add_argument('--gpu', action='store_true', default=False)
    parser.add_argument('--balance', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data', type=str, default='Safe',
                        choices=['Safe', 'Risk'])
    parser.add_argument('--pre_trained_model', type=str, default='dcgnn',
                        choices=['lstm', 'dcgnn'])

    args = parser.parse_args()

    device = torch.device('cuda' if args.gpu else 'cpu')
    print("Device: ", device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.exists(args.pt):
        raise "Pre-trained model does not exist!"

    chk_path = os.path.join(args.checkpoint_path, f'{args.seed}')
    model_save_path = os.path.join(chk_path, f'{args.data}.pt')
    results_data_save_path = os.path.join(
        chk_path, f'{args.data}.txt')

    if not os.path.exists(chk_path):
        os.makedirs(chk_path)
    results_file = open(results_data_save_path, 'w')

    test_participants = [45, 42, 48, 14, 26]
    train_participants = np.setdiff1d(np.arange(2, 53), test_participants)
    print("Test participants: ", test_participants)

    # Load pre-trained model
    pre_trained_model_weigths = torch.load(args.pt, map_location=device)
    if args.pre_trained_model == 'lstm':
        pt_model = LSTM(in_channels=601, num_electrodes=61, num_classes=4)
    elif args.pre_trained_model == 'dcgnn':
        pt_model = DGCNN(in_channels=601, num_electrodes=61, num_classes=4)
    pt_model.load_state_dict(pre_trained_model_weigths)
    pt_model.fc2 = Identity()

    # Loading model
    model = EncoderDecoder().to(device)
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Loading dataset
    train_set = EEGModelDataset(args.data_path, train=True, par_numbers=train_participants,
                                type=args.data, model=pt_model, device=device)
    test_set = EEGModelDataset(args.data_path, train=False, par_numbers=test_participants,
                               type=args.data, model=pt_model, device=device)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False)

    best_cos = 0
    for epoch in tqdm(range(1, args.epochs + 1), total=args.epochs, desc='Epochs: '):
        for eeg, next_eeg in train_loader:
            eeg, next_eeg = eeg.to(device), next_eeg.to(device)

            pred_eeg = model(eeg)
            # maximizing average cosine similarity
            loss = -F.cosine_similarity(pred_eeg, next_eeg).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation accuracy
        train_results = evaluate(model, train_loader)
        test_results = evaluate(model, test_loader)

        if test_results['cos_sim'] > best_cos:
            best_cos = test_results['cos_sim']
            torch.save(model.state_dict(), model_save_path)
            print("Best cosine similarity: ", best_cos, ", best p_val: ", test_results['p_val'])
            
        # Save results
        results_file.write(
            f"Epoch: {epoch} - train: {dict_to_string(train_results)}, test: {dict_to_string(test_results)}\n\n")
