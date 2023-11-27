declare -a arr=("eeg" "lstm" "dcgnn" "vit")
for i in "${arr[@]}"
do
   echo "python train_exp2.py --gpu --net_type $i"
done