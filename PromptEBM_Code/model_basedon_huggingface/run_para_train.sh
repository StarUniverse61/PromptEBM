#!/usr/bin/env bash

learning_rate=$1
max_seq_length=$2
cuda_device=$3
optimizer=$4 #RecAdam
model_name_or_path=$5 #facebook/bart-large
num_train_epochs=$6 #30
data_dir=$7 #./data/quora/data4fairseq/
output_dir=$8 #./data/quora/output_ppll_recadam_lr4e05_msl64
per_gpu_train_batch_size=$9 #96
encoder_norm_self=${10} #power
decoder_norm_self=${11} #power
encoder_norm_ff=${12} #layer
decoder_norm_ff=${13} #layer
NUM=${14}
eval_data_dirs=${15}

python para_train.py --learning_rate $learning_rate \
 --max_seq_length $max_seq_length \
 --cuda_device $cuda_device \
 --optimizer $optimizer \
 --model_name_or_path $model_name_or_path \
 --num_train_epochs $num_train_epochs \
 --data_dir $data_dir \
 --output_dir $output_dir \
 --per_gpu_train_batch_size $per_gpu_train_batch_size \
 --encoder_norm_self $encoder_norm_self \
 --decoder_norm_self $decoder_norm_self \
 --encoder_norm_ff $encoder_norm_ff \
 --decoder_norm_ff $decoder_norm_ff

#python scripts/average_checkpoints.py --inputs $output_dir --num-epoch-checkpoints $NUM --output $output_dir/averaged_model.pt

python para_eval.py --model_name_or_path $output_dir --data_dir $data_dir --cuda_device $cuda_device

if [ -z "$eval_data_dirs" ]
then
      echo "\$my_var is NULL"
      ##we need to evaluate previous datasets
      array_pre_data_dir=(`echo $eval_data_dirs | sed 's/,/\n/g'`)
     for pre_data_dir in "${array_pre_data_dir[@]}"
     do
         echo "$pre_data_dir"
         python para_eval.py --model_name_or_path $output_dir --data_dir $pre_data_dir
    done

else
      echo "\$my_var is NOT NULL"
fi

