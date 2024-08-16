cuda=${1}
seq_train_type=${2}
tasks=${3}
models=${4}
gamma=${5}
pretrain_cof=${6}
encoder_norm_self=${7}
decoder_norm_self=${8}
encoder_norm_ff=${9}
decoder_norm_ff=${10}
save_name=${11}
process_number=${12}

task_group=(`echo $tasks | tr '-' ' '`)
echo ${task_group[*]}
model_arr=(`echo $models | tr ',' ' '`)
echo ${model_arr[*]}

#arr=(`echo $tasks | tr ',' ' '`)

#FOO=( a b c )     # initialize the array
#BAR=${FOO[@]}     # create a space delimited string from array
#BAZ=${BAR// /,}   # use parameter expansion to substitute spaces with comma
#echo $BAZ

i=0
while [ $i -ne $process_number ]
do
        echo "$i"
        echo task\_group==${task_group[$i]}
        task_group_i=${task_group[$i]}
        SAVE=./models\_${task_group_i//,/_}\_${model_arr[$i]}\_$save_name
        echo SAVE\_NAME=$SAVE
        bash run_full_parameters.sh $cuda ./data $SAVE $seq_train_type ${task_group[$i]} ${model_arr[$i]} $gamma $pretrain_cof $encoder_norm_self $decoder_norm_self $encoder_norm_ff $decoder_norm_ff
        i=$(($i+1))
done

#bash run.sh $cuda ./data ./models pn_calibration sst,srl,woz.en,squad2,wikisql openai-gpt
#bash run.sh $cuda ./data ./models pn_calibration srl,sst,woz.en,wikisql,squad2 openai-gpt
#bash run.sh $cuda ./data ./models pn_calibration woz.en,sst,srl,wikisql,squad2 gpt2