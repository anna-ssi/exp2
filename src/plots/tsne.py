import os

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch
from torch import nn
from torcheeg.models import DGCNN
from src.utils.preprocess import *


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

def plot_tsne(data_tsne, labels, title, save_path=None):
    plt.figure(figsize=(10, 10))
    plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=labels)
    plt.title(title)
    plt.colorbar()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':
    tsne = TSNE(n_components=2, random_state=42)
    model = DGCNN(in_channels=601, num_electrodes=61, num_classes=4)
    model.fc2 = Identity()
    
    device = torch.device('cuda')
    print("Device: ", device)

    test_participants = [2]
    env_type = 'Safe'
    
    model.to(device)

    erps, labels = [], []
    for idx in test_participants:
        idx = str(idx).zfill(3)
        erp_path = os.path.join('./data/erp', f'{env_type}_1B_ERP_{idx}.mat')
        csv_path = os.path.join('./data/csv', f'{env_type}_1B_Trial_{idx}.csv')
        
        erp_data = read_erp_file(erp_path)
        label = read_csv_file(csv_path)
        erp = normalize(erp_data)
        
        with torch.no_grad():
            erp = torch.from_numpy(erp).to(device)
            erp = model(erp)
            print(erp.shape)
            exit()
            erps.append(erp.squeeze().numpy())
            
        labels.append(label)
    
    erps = np.concatenate(erps, axis=0)
    labels = np.concatenate(labels, axis=0)
        
        