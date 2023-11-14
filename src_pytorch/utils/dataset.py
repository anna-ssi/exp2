import os
import numpy as np

import torch
from torch.utils.data import Dataset

from src.utils.helper import get_file_names
from src.utils.preprocess import read_erp_file, read_csv_file, normalize


class EEGDataset(Dataset):
    def __init__(self, data_path: str, type: str = 'Safe') -> None:
        self.path = data_path
        self.type = type if type in ['Safe', 'Risk'] else None

        self.data, self.labels = self.load()
        self.clean_data()
        self.save()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.Tensor(self.data[idx]).unsqueeze(0), torch.LongTensor(self.labels[idx])

    def load(self):
        if os.path.exists(os.path.join(self.path, f'{self.type}_erp.npy')) and \
                os.path.exists(os.path.join(self.path, f'{self.type}_labels.npy')):
            return (np.load(os.path.join(self.path, f'{self.type}_erp.npy')),
                    np.load(os.path.join(self.path, f'{self.type}_labels.npy')))

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

    def save(self):
        np.save(os.path.join(self.path, f'{self.type}_erp.npy'), self.data)
        np.save(os.path.join(
            self.path, f'{self.type}_labels.npy'), self.labels)
