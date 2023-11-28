#!/bin/bash
#SBATCH --job-name=actions_repeat_eeg
#SBATCH --account=rrg-whitem
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --time=0-02:55
#SBATCH --mem=16G
#SBATCH -o ./results/aug/actions_repeat_eeg.out

module load python/3.10.2
source ./venv/bin/activate
python train_actions.py --gpu --net_type=eeg --balance

