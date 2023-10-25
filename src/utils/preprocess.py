import scipy.io as sio
import pandas as pd


def read_eeg_file(path: str):
    """
    Read EEG data from a .mat file
        :param path: path to the file
        :return: EEG data
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
