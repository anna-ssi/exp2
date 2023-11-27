#!/bin/bash
#SBATCH --job-name=exp2
#SBATCH --account=rrg-whitem
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --time=0-02:55
#SBATCH --mem=16G
#SBATCH -o ./exp2.out

module load python/3.10.2
source ./venv/bin/activate

python train_exp2.py --gpu --net_type="eeg"

