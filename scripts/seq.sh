#!/bin/bash
#SBATCH --job-name=seq
#SBATCH --account=rrg-whitem
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --time=0-02:55
#SBATCH --mem=16G
#SBATCH -o ./seq.out

module load python/3.10.2
source ./venv/bin/activate

python train_seq.py --gpu 


