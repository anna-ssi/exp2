#!/bin/bash
#SBATCH --job-name=actions
#SBATCH --account=rrg-whitem
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --time=0-02:55
#SBATCH --mem=6G
#SBATCH -o ./actions.out

module load python/3.10.2
source ./venv/bin/activate

declare -a arr=("eeg" "lstm" "dcgnn" "vit")
for i in "${arr[@]}"
do
   python train_actions.py --gpu --net_type=$i
done

