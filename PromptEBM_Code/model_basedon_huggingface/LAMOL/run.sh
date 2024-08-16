cuda=${1}
data_dir=${2}
model_dir_root=${3}
seq_train_type=${4}
tasks=${5}
model_name=${6}


echo $cuda
echo $data_dir
echo $model_dir_root
echo $seq_train_type
echo $tasks
echo $model_name

#array_pre_data_dir=(`echo $eval_data_dirs | sed 's/,/\n/g'`)
#spo='one;two;three'
#OIFS=$IFS
#IFS=';'
#spo_array=($eval_data_dirs)
#IFS=$OIFS
#echo ${spo_array[*]}
#
#for pre_data_dir in "${spo_array[@]}"
#   do
#   echo "$pre_data_dir"
#done

arr=(`echo $tasks | tr ',' ' '`)
echo ${arr[*]}
echo ${arr[0]}
echo ${arr[1]}
echo ${arr[2]}

CUDA_VISIBLE_DEVICES=$cuda bash train.sh --data_dir $data_dir --model_dir_root $model_dir_root --seq_train_type $seq_train_type --tasks ${arr[*]} --model_name $model_name
CUDA_VISIBLE_DEVICES=$cuda bash test.sh --data_dir $data_dir --model_dir_root $model_dir_root --seq_train_type $seq_train_type --tasks ${arr[*]} --model_name $model_name

#CUDA_VISIBLE_DEVICES=0 bash train.sh --data_dir ./data --model_dir_root ./models --seq_train_type pn_calibration --tasks sst srl woz.en --model_name openai-gpts
#CUDA_VISIBLE_DEVICES=0 bash test.sh --data_dir ./data --model_dir_root ./models --seq_train_type pn_calibration --tasks sst srl woz.en --model_name openai-gpt