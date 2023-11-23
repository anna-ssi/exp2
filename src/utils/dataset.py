import os
import numpy as np

import torch
from torch.utils.data import Dataset

from src.utils.helper import get_file_names
from src.utils.preprocess import read_erp_file, read_csv_file, normalize


class EEGDataset(Dataset):
    def __init__(self, data_path: str, type: str = 'Safe', balanced: bool = False, net_type: str = 'eeg') -> None:
        self.path = data_path
        self.type = type
        self.balanced = balanced
        self.net_type = net_type

        self.erp_path, self.label_path = self.get_paths()
        self.data, self.labels = self.load()
        
        self.shuffle()
        # self.clean_data()
        
        # if self.balanced:
        #     self.balance()
        # self.save()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.net_type == 'eeg':
            return torch.Tensor(self.data[idx]).unsqueeze(0), torch.LongTensor(self.labels[idx])
        else:
            return torch.Tensor(self.data[idx]), torch.LongTensor(self.labels[idx])
        
    def get_data_stats(self):
        counts = np.unique(self.labels, return_counts=True)
        print(f'Label {counts[0][0]}: {counts[1][0]/ len(self.labels) * 100:.2f}%')
        print(f'Label {counts[0][1]}: {counts[1][1]/ len(self.labels) * 100:.2f}%')

    def load(self):
        if os.path.exists(self.erp_path) and os.path.exists(self.label_path):
            return (np.load(self.erp_path), np.load(self.label_path))

        erps, labels = [], []
        erp_paths = get_file_names(os.path.join(
            self.path, 'erp'), ext='.mat', keyword=self.type)

        for erp_path in erp_paths:
            csv_path = erp_path.replace('ERP', 'Trial').replace(
                'mat', 'csv').replace('erp', 'csv')

            erp = read_erp_file(erp_path)
            label = read_csv_file(csv_path)

            erps.append(erp)
            labels.append(label)

        erp_data = np.concatenate(erps, axis=2)
        erp_data = np.transpose(erp_data, axes=(2, 0, 1))
        erp_data = normalize(erp_data)

        labels = np.concatenate(labels, axis=0)
        return erp_data, labels

    def clean_data(self):
        del_idx = np.where(self.labels == -1)[0]
        self.labels = np.delete(self.labels, del_idx, axis=0)
        self.data = np.delete(self.data, del_idx, axis=0)

    def balance(self):
        _, counts = np.unique(self.labels, return_counts=True)
        minor_class = np.argmin(counts)
        difference = np.abs(counts[0] - counts[1])

        minor_class_idx = np.where(self.labels == minor_class)[0]
        data = self.data[minor_class_idx]
        labels = np.array([minor_class] * difference).reshape(-1, 1)

        new_samples_idx = np.random.choice(
            len(minor_class_idx), size=difference)
        new_data = np.concatenate((self.data, data[new_samples_idx]), axis=0)
        new_labels = np.concatenate((self.labels, labels), axis=0)

        self.data = new_data
        self.labels = new_labels

    def save(self):
        np.save(self.erp_path, self.data)
        np.save(self.label_path, self.labels)

    def get_paths(self):
        eeg_path = os.path.join(self.path, f'{self.type}_erp.npy')
        label_path = os.path.join(self.path, f'{self.type}_label.npy')

        if self.balanced:
            eeg_path = eeg_path.replace('.npy', '_balanced.npy')
            label_path = label_path.replace('.npy', '_balanced.npy')
        return eeg_path, label_path

    def shuffle(self):
        idx = np.arange(len(self.labels))
        np.random.shuffle(idx)

        self.data = self.data[idx]
        self.labels = self.labels[idx]