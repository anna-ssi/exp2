import os
import numpy as np

from src.utils.preprocess import read_erp_file, read_behavior_data

TEST_PARTICIPANT_NUMBERS = [3, 21, 10, 48]
TRAIN_PARTICIPANT_NUMBERS = np.setdiff1d(
    np.arange(2, 53), TEST_PARTICIPANT_NUMBERS)


class EEGEnv:
    def __init__(self, data_path: str, train: bool, trial_type: str = 'Safe', net_type: str = 'eeg') -> None:
        self.path = data_path
        self.trial_type = trial_type
        self.net_type = net_type
        self.train = train

        self.erps, self.rewards, self.choices = self.load()

    def load(self):
        par_numbers = TRAIN_PARTICIPANT_NUMBERS if self.train else TEST_PARTICIPANT_NUMBERS
        erps, rewards, choices = []
        
        for number in par_numbers:
            number = str(number).zfill(3)
            erp_path = os.path.join(
                self.path, 'erp', f'{self.trial_type}_1B_ERP_{number}.mat')
            csv_path = os.path.join(
                self.path, 'behavior', f'{self.trial_type}_Exp2_{number}.csv')

            if not (os.path.exists(csv_path) and os.path.exists(erp_path)):
                continue

            erp = read_erp_file(erp_path)
            beh_df = read_behavior_data(csv_path)
            beh_df = beh_df.reset_index()
            
            assert erp.shape[2] == beh_df.shape[0]

            # remove trials with no response
            index_to_drop = beh_df[beh_df['Response'] == 0].index
            erp = np.delete(erp, index_to_drop, axis=2)
            beh_df = beh_df.drop(index_to_drop)

            # getting the rewards
            rewards = beh_df[['Sq1', 'Sq2', 'Sq3', 'Sq4']].to_numpy()
            human_choice = beh_df['SqChs'].to_numpy()
            
            erps.append(erp)
            rewards.append(rewards)
            choices.append(human_choice)
        
        return erps, rewards, choices
        
    def start(self):
        pass
    
    def step(self, action):
        pass
    
    def reset(self):
        pass
    


if __name__ == '__main__':
    eegenv = EEGEnv('./data', train=True)
