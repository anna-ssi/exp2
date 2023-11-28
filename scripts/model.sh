#!/bin/bash
#SBATCH --job-name=model
#SBATCH --account=rrg-whitem
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --time=0-02:55
#SBATCH --mem=32G
#SBATCH -o ./model.out

module load python/3.10.2
source ./venv/bin/activate

python train_model.py --gpu 


