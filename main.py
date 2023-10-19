from src.utils.preprocess import read_eeg_file, read_csv_file

if __name__ == '__main__':
    data_path = './data/eeg/SAFE_EE_ERP_002.mat'
    label_path = './data/csv/SAFE_EE_EEG_002.csv'
    
    data = read_eeg_file(data_path)
    labels = read_csv_file(label_path)
    
    # key = random.PRNGKey(1)