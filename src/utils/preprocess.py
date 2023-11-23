import scipy.io as sio
import pandas as pd
import numpy as np


def read_erp_file(path: str):
    """
    Read ERP data from a .mat file
        :param path: path to the file
        :return: ERP data
    """
    data = sio.loadmat(path)
    if 'ERP' in data.keys():
        data = data['ERP']
    elif 'WAVT' in data.keys():
        data = data['WAVT']
    return data


def read_csv_file(path: str):
    """
    Read CSV data from a .csv file
        :param path: path to the file
        :return: CSV data
    """
    data = pd.read_csv(path, header=None, names=['label'])
    return data


def normalize(data):
    return (data - np.mean(data)) / np.std(data)


def get_waveform(data, type='delta'):
    if type == 'delta':
        data = data[:, :4, :]
    elif type == 'theta':
        data = data[:, 4:8, :]
    elif type == 'alpha':
        data = data[:, 8:12, :]
    elif type == 'beta':
        data = data[:, 12:, :]

    return data.transpose(3, 1, 0, 2)


if __name__ == '__main__':
    erp = read_erp_file('./data/wav/Risk_1B_WAV_002.mat')
    erp = get_waveform(erp, 'delta')
    print(erp.shape)
