import os
import numpy as np
import pickle

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from src.utils.helper import one_hot
from src.utils.preprocess import *


TEST_PARTICIPANT_NUMBERS = [3, 21, 10, 48]
TRAIN_PARTICIPANT_NUMBERS = np.setdiff1d(
    np.arange(2, 53), TEST_PARTICIPANT_NUMBERS)


class EEGDataset(Dataset):
    def __init__(self, data_path: str, train: bool, par_numbers: list, type: str = 'Safe', net_type: str = 'eeg', balance: bool = True) -> None:
        self.path = data_path
        self.type = type
        self.net_type = net_type
        self.train = train
        self.balanced = balance
        self.par_numbers = par_numbers

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx].reshape(-1, 1)
        if self.net_type == 'eeg':
            return torch.Tensor(self.data[idx]).unsqueeze(0), torch.LongTensor(label)
        else:
            return torch.Tensor(self.data[idx]), torch.LongTensor(label)

    def load(self):
        ...

    def balance(self):
        ...

    def shuffle(self):
        idx = np.arange(len(self.labels))
        np.random.shuffle(idx)

        self.data = self.data[idx]
        self.labels = self.labels[idx]

    def save(self):
        np.save(self.erp_path, self.data)
        np.save(self.label_path, self.labels)

    def get_paths(self, folder_name='actions_seq'):
        path = os.path.join(self.path, folder_name)
        os.makedirs(path, exist_ok=True)
        schema = 'train' if self.train else 'test'
        balance = 'balanced' if self.balanced else ''

        eeg_path = os.path.join(
            path, f'{self.type}_erp_{schema}_{balance}.npy')
        label_path = os.path.join(
            path, f'{self.type}_label_{schema}_{balance}.npy')
        return eeg_path, label_path

    def get_data_stats(self):
        counts = np.unique(self.labels, return_counts=True)
        schema = 'Train' if self.train else 'Test'
        counts_str = f'{schema} data stats: '
        for label, count in zip(*counts):
            counts_str += f'label {label}: {count}, '
        print(counts_str[:-2])
        print(f'{schema} dataset length: {len(self.labels)}')


class EEGDatasetExp2(EEGDataset):
    def __init__(self, data_path: str, train: bool, par_numbers: list, type: str = 'Safe', net_type: str = 'eeg', balance: bool = True) -> None:
        super().__init__(data_path, train, par_numbers, type, net_type, balance)

        self.erp_path, self.label_path = self.get_paths(folder_name='exp2')
        self.data, self.labels = self.load()
        self.num_classes = 2

        self.clean_data()

        if train:
            if self.balanced:
                self.balance()
            self.shuffle()

    def __getitem__(self, idx):
        label = one_hot(self.labels[idx].astype(int), self.num_classes)
        if self.net_type == 'eeg':
            return torch.Tensor(self.data[idx]).unsqueeze(0), torch.Tensor(label)
        else:
            return torch.Tensor(self.data[idx]), torch.Tensor(label)

    def load(self):
        erps, labels = [], []

        for number in self.par_numbers:
            number = str(number).zfill(3)
            erp_path = os.path.join(
                self.path, 'erp', f'{self.type}_1B_ERP_{number}.mat')
            csv_path = os.path.join(
                self.path, 'csv', f'{self.type}_1B_Trial_{number}.csv')

            if not (os.path.exists(csv_path) and os.path.exists(erp_path)):
                continue

            erp = read_erp_file(erp_path)
            label = read_csv_file(csv_path)

            erp = normalize(erp)
            erps.append(erp)
            labels.append(label)

        erp_data = np.concatenate(erps, axis=2)
        erp_data = np.transpose(erp_data, axes=(2, 0, 1))
        erp_data = normalize(erp_data)

        labels = np.concatenate(labels, axis=0)
        return erp_data, labels

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

    def clean_data(self):
        del_idx = np.where(self.labels == -1)[0]
        self.labels = np.delete(self.labels, del_idx, axis=0)
        self.data = np.delete(self.data, del_idx, axis=0)


class EEGDatasetAction(EEGDataset):
    def __init__(self, data_path: str, train: bool, par_numbers: list, type: str = 'Safe', net_type: str = 'eeg', balance: bool = True) -> None:
        super().__init__(data_path, train, par_numbers, type, net_type, balance)
        self.erp_path, self.label_path = self.get_paths()
        self.data, self.labels = self.load()
        self.num_classes = 4

        if train:
            if self.balanced:
                self.balance()
            self.shuffle()

    def load(self):
        erps, labels = [], []

        for number in self.par_numbers:
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

            # remove trials with no response
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

        return erp_data, labels

    def balance(self):
        _, counts = np.unique(self.labels, return_counts=True)
        max_class = np.argmax(counts)
        difference = [np.abs(counts[max_class] - counts[i]) for i in range(4)]

        for i, diff in enumerate(difference):
            if diff == 0:
                continue
            class_idx = np.where(self.labels == i)[0]
            data = self.data[class_idx]
            labels = np.array([i] * diff)

            new_samples_idx = np.random.choice(
                len(class_idx), size=diff)

            new_data = np.concatenate(
                (self.data, data[new_samples_idx]), axis=0)
            new_labels = np.concatenate((self.labels, labels), axis=0)

            self.data = new_data
            self.labels = new_labels


class EEGDatasetActionTrajectory(EEGDataset):
    def __init__(self, data_path: str, train: bool, par_numbers: list, type: str = 'Safe', net_type: str = 'lstm', balance: bool = False) -> None:
        super().__init__(data_path, train, par_numbers, type, net_type, balance)
        self.erp_path, self.label_path = self.get_paths()
        self.data, self.labels = self.load()
        self.num_classes = 4

        if not (os.path.exists(self.erp_path) or os.path.exists(self.label_path)):
            self.save()

    def load(self):
        if os.path.exists(self.erp_path) and os.path.exists(self.label_path):
            return (pickle.load(open(self.erp_path, 'rb')), pickle.load(open(self.label_path, 'rb')))

        erps, labels = [], []

        for number in self.par_numbers:
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

            # remove trials with no response
            index_to_drop = beh_df[beh_df['Response'] == 0].index
            erp = np.delete(erp, index_to_drop, axis=2)
            beh_df = beh_df.drop(index_to_drop)

            assert erp.shape[2] == beh_df.shape[0]

            label = beh_df['SqChs'].values - 1

            # erps.append(normalize(erp))
            erps.append(erp)
            labels.append(label)

        return erps, labels

    def save(self):
        pickle.dump(self.data, open(self.erp_path, 'wb'))
        pickle.dump(self.labels, open(self.label_path, 'wb'))

    def get_paths(self, folder_name='actions_seq'):
        path = os.path.join(self.path, folder_name)
        os.makedirs(path, exist_ok=True)
        schema = 'train' if self.train else 'test'
        balance = 'balanced' if self.balanced else ''

        eeg_path = os.path.join(
            path, f'{self.type}_erp_{schema}_{balance}.pickle')
        label_path = os.path.join(
            path, f'{self.type}_label_{schema}_{balance}.pickle')
        return eeg_path, label_path
