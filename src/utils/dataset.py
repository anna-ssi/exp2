import os
import numpy as np

import torch
from torch.utils.data import Dataset

from src.utils.helper import get_file_names
from src.utils.preprocess import *


TEST_PARTICIPANT_NUMBERS = [3, 21, 10, 48]
TRAIN_PARTICIPANT_NUMBERS = np.setdiff1d(
    np.arange(2, 52), TEST_PARTICIPANT_NUMBERS)

class EEGDataset(Dataset):
    def __init__(self, data_path: str, train: bool, type: str = 'Safe', net_type: str = 'eeg') -> None:
        self.path = data_path
        self.type = type
        self.net_type = net_type
        self.train = train

        self.erp_path, self.label_path = self.get_paths()
        self.load(train=train)

        if train:
            self.balance()
            self.shuffle()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.net_type == 'eeg':
            return torch.Tensor(self.data[idx]).unsqueeze(0), torch.LongTensor(self.labels[idx])
        else:
            return torch.Tensor(self.data[idx]), torch.LongTensor(self.labels[idx])

    def get_data_stats(self):
        counts = np.unique(self.labels, return_counts=True)
        print(
            f'Label {counts[0][0]}: {counts[1][0]/ len(self.labels) * 100:.2f}%')
        print(
            f'Label {counts[0][1]}: {counts[1][1]/ len(self.labels) * 100:.2f}%')

    def load(self, train=True):
        ...

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
        eeg_path = os.path.join(self.path, f'{self.type}_erp_{self.train}.npy')
        label_path = os.path.join(self.path, f'{self.type}_label_{self.train}.npy')
        return eeg_path, label_path

    def shuffle(self):
        idx = np.arange(len(self.labels))
        np.random.shuffle(idx)

        self.data = self.data[idx]
        self.labels = self.labels[idx]



class EEGDatasetExp(Dataset):
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
        print(
            f'Label {counts[0][0]}: {counts[1][0]/ len(self.labels) * 100:.2f}%')
        print(
            f'Label {counts[0][1]}: {counts[1][1]/ len(self.labels) * 100:.2f}%')

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


class EEGDatasetAction(EEGDataset):
    def __init__(self, data_path: str, train: bool, type: str = 'Safe', net_type: str = 'eeg') -> None:
        super().__init__(data_path, train, type, net_type)

    def load(self, train=True):
        if os.path.exists(self.erp_path) and os.path.exists(self.label_path):
            return (np.load(self.erp_path), np.load(self.label_path))

        par_numbers = TRAIN_PARTICIPANT_NUMBERS if train else TEST_PARTICIPANT_NUMBERS
        erps, labels = [], []

        for number in par_numbers:
            number = str(number).zfill(3)
            erp_path = os.path.join(
                self.path, 'erp', f'{self.type}_1B_ERP_{number}.mat')
            csv_path = os.path.join(
                self.path, 'behavior', f'{self.type}_Exp2_{number}.csv')
            
            if not (os.path.exists(csv_path) and os.path.exists(erp_path)):
                continue

            erp = read_erp_file(erp_path)
            beh_df = read_behavior_data(csv_path)
            beh_df = beh_df.reset_index()
            
            assert erp.shape[2] == beh_df.shape[0]

            index_to_drop = beh_df[beh_df['Response'] == 0].index
            erp = np.delete(erp, index_to_drop, axis=2)
            beh_df = beh_df.drop(index_to_drop)

            label = beh_df['SqChs'].values - 1

            erps.append(erp)
            labels.append(label)

        erp_data = np.concatenate(erps, axis=2)
        erp_data = np.transpose(erp_data, axes=(2, 0, 1))
        erp_data = normalize(erp_data)

        labels = np.concatenate(labels, axis=0)
        
        self.erps = erps
        self.labels = labels

   