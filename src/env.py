import os
import numpy as np

from src.utils.helper import get_file_names
from src.utils.preprocess import read_erp_file, read_csv_file, normalize


class EEGEnv:
    def __init__(self, data_path: str, trial_type: str = 'Safe', net_type: str = 'eeg') -> None:
        self.path = data_path
        self.trial_type = trial_type
        self.net_type = net_type

        self.load()

    def load(self):
        behav_paths = get_file_names(os.path.join(
            self.path, 'behavior'), ext='.csv', keyword=self.trial_type)
        print(len(behav_paths))
        
        for behav_path in behav_paths[7:]:
            participant_number = behav_path.split('/')[-1].split('_')[-1].replace('.csv', '')

            beh_df = read_csv_file(behav_path, header=0, names=None)
            sorted_df = beh_df.sort_values(by=["BlockNum", "Trial"], ascending=True)
            
            erp_path = os.path.join(self.path, 'erp', f'{self.trial_type}_1B_ERP_{participant_number}.mat')
            erp = read_erp_file(erp_path)
            
            print(erp.shape, sorted_df[sorted_df.ForcedChoice == 0].shape)     
            print(sorted_df[(sorted_df.ForcedChoice == 0) & (sorted_df.Trial == 9)].shape)     
            print(sorted_df[sorted_df.ForcedChoice == 0].head())
            break


        
        
if __name__ == '__main__':
    eegenv = EEGEnv('./data')
    