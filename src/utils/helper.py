from os import listdir
from os.path import isfile, join
from typing import List
from collections import defaultdict

import numpy as np


def get_file_names(path: str, ext: str, keyword: str = None) -> List[str]:
    """
    Get all file names in a directory
        :param path: path to the directory
        :return: list of file names
    """

    paths = []
    for f in listdir(path):
        if isfile(join(path, f)) and f.endswith(ext):
            if keyword is not None and keyword not in f:
                continue
            paths.append(join(path, f))
    return paths

def dict_to_string(d):
    return ', '.join([f'{k}: {v:.3f}' for k, v in d.items()])

def get_data_stats(dataset):
    label_counter = defaultdict(int)

    for _, labels in dataset:
        for label in labels:
            label_counter[int(label.cpu().numpy())] += 1
    
    return f'Label {0}: {label_counter[0]/ len(dataset) * 100:.2f}%, Label {1}: {label_counter[1]/ len(dataset) * 100:.2f}%'


def combine_datasets():
    import numpy as np
    
    safe_erp = np.load('./data/Safe_erp_balanced.npy')
    safe_labels = np.load('./data/Safe_label_balanced.npy')
    
    risk_erp = np.load('./data/Risk_erp_balanced.npy')
    risk_labels = np.load('./data/Risk_label_balanced.npy')
    
    all_erp = np.concatenate((safe_erp, risk_erp), axis=0)
    all_labels = np.concatenate((safe_labels, risk_labels), axis=0)
    
    
    np.save('./data/All_erp_balanced.npy', all_erp)
    np.save('./data/All_label_balanced.npy', all_labels)    
    
def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

if __name__ == '__main__':
    combine_datasets()