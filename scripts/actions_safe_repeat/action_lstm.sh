#!/bin/bash
#SBATCH --job-name=actions_repeat_lstm
#SBATCH --account=rrg-whitem
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --time=0-02:55
#SBATCH --mem=16G
#SBATCH -o ./results/actions_repeat_lstm.out

module load python/3.10.2
source ./venv/bin/activate
python train_actions.py --gpu --net_type=lstm --balance

