import os
import numpy as np
import tensorflow as tf

from src.utils.helper import get_file_names
from src.utils.preprocess import read_erp_file, read_csv_file, normalize


class TrainTestSplit:
    def __init__(self, data_path: str, test_size: float = 0.1, type: str = 'Safe') -> None:
        self.test_size = test_size
        self.path = data_path
        self.type = type

        self.data, self.labels = self.load()
        self.clean_data()
        self.save()

    def load(self):
        if os.path.exists(os.path.join(self.path, f'{self.type}_erp.npy')) and \
                os.path.exists(os.path.join(self.path, f'{self.type}_labels.npy')):
            return (np.load(os.path.join(self.path, f'{self.type}_erp.npy')),
                    np.load(os.path.join(self.path, f'{self.type}_labels.npy')))
            
        erps, labels = [], []
        erp_paths = get_file_names(os.path.join(self.path, 'erp'), ext='.mat', keyword=self.type)

        for erp_path in erp_paths:
            csv_path = erp_path.replace('ERP', 'Trial').replace('mat', 'csv').replace('erp', 'csv')
            
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

    def split(self):
        test_size = int(self.test_size * self.data.shape[0])
        train_size = self.data.shape[0] - test_size

        train_data, train_labels = self.data[:
                                             train_size], self.labels[:train_size]
        test_data, test_labels = self.data[train_size:], self.labels[train_size:]

        return (tf.data.Dataset.from_tensor_slices((train_data, train_labels)),
                tf.data.Dataset.from_tensor_slices((test_data, test_labels)))

    def save(self):
        np.save(os.path.join(self.path, f'{self.type}_erp.npy'), self.data)
        np.save(os.path.join(
            self.path, f'{self.type}_labels.npy'), self.labels)
