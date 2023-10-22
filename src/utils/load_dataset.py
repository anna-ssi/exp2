import os
import numpy as np

from src.utils.helper import get_file_names
from src.utils.preprocess import read_eeg_file, read_csv_file

class EEGDataset:
    def __init__(self, data_path: str) -> None:
        self.path = data_path
        
        self.data = self.load_eeg()
        self.labels = self.load_label()
        self.permute()
    
    def load_eeg(self):
        eegs = []
        eeg_paths = get_file_names(os.path.join(self.path, 'eeg'), ext='.mat')
        
        for eeg_path in eeg_paths:
            if "ERP" not in eeg_path:
                continue
            eeg = read_eeg_file(eeg_path)
            eegs.append(eeg)
        
        return np.concatenate(eegs, axis=2)
        
    def load_label(self):
        labels = []
        label_paths = get_file_names(os.path.join(self.path, 'csv'), ext='.csv')
        
        for label_path in label_paths:
            label = read_csv_file(label_path)
            labels.append(label)
        
        return np.concatenate(labels).reshape(1, -1)
    
    def __len__(self):
        return self.labels.shape[1]
    
    def get_batch(self, batch_size: int):
        random_indices = np.random.choice(self.labels.shape[1], batch_size)
        return self.data[:, :, random_indices], self.labels[:, random_indices]
    
    def permute(self):
        permuted_indices = np.random.permutation(self.labels.shape[1])
        self.data = self.data[:, :, permuted_indices]
        self.labels = self.labels[:, permuted_indices]
    
    
    