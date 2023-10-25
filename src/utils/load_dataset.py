import os
import numpy as np
import tensorflow as tf

from src.utils.helper import get_file_names
from src.utils.preprocess import read_eeg_file, read_csv_file


class TrainTestSplit:
    def __init__(self, data_path: str, test_size: float = 0.1, type: str = 'safe') -> None:
        self.test_size = test_size
        self.path = data_path
        self.type = type

        self.data = self.load_eeg()
        self.labels = self.load_label()
        self.clean_data()
        self.save()

    def load_eeg(self):
        if os.path.exists(os.path.join(self.path, f'{self.type}_eeg.npy')):
            return np.load(os.path.join(self.path, f'{self.type}_eeg.npy'))

        eegs = []
        eeg_paths = get_file_names(os.path.join(self.path, 'eeg'), ext='.mat')

        for eeg_path in eeg_paths:
            if "ERP" not in eeg_path:
                continue
            eeg = read_eeg_file(eeg_path)
            eegs.append(eeg)
        data = np.concatenate(eegs, axis=2)
        data = np.transpose(data, axes=(2, 0, 1))
        return data

    def load_label(self):
        if os.path.exists(os.path.join(self.path, f'{self.type}_labels.npy')):
            return np.load(os.path.join(self.path, f'{self.type}_labels.npy'))

        labels = []
        label_paths = get_file_names(
            os.path.join(self.path, 'csv'), ext='.csv')

        for label_path in label_paths:
            label = read_csv_file(label_path)
            labels.append(label)

        return np.concatenate(labels)

    def clean_data(self):
        del_idx = np.where(self.labels == -1)[0]
        self.labels = np.delete(self.labels, del_idx, axis=0)
        self.data = np.delete(self.data, del_idx, axis=0)

    def split(self):
        test_size = int(self.test_size * self.data.shape[0])
        train_size = self.data.shape[0] - test_size

        train_data, train_labels = self.data[:
                                             train_size], self.labels[:train_size]
        test_data, test_labels = self.data[train_size:], self.labels[train_size:]

        return (tf.data.Dataset.from_tensor_slices((train_data, train_labels)),
                tf.data.Dataset.from_tensor_slices((test_data, test_labels)))

    def save(self):
        np.save(os.path.join(self.path, f'{self.type}_eeg.npy'), self.data)
        np.save(os.path.join(self.path, f'{self.type}_labels.npy'), self.labels)