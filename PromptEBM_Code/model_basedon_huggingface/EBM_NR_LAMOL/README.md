In order to make training with high efficiency, we use a series of bash scritps. They are chained together. We only need to run bash run_five_task.sh for example, they will run in all gpu resources, say 8 gpus. Each gpu will run a few task training one after another. 

Specifically speaking, run_five_task.sh will call experiments_five_tasks_gpu.sh, which in turn call run_full_parameters.sh, which in turn runs train_pn_calibration.sh, which in turn run train_pn_calibration.py.

In parameter's perspective, run_five_task.sh means we will run five tasks continously. In this bash, you may define 8 bash scripts, calling 8 gpus respectively as following example,  

nohup bash ./experiments_five_tasks_gpu.sh 0 pn_calibration sst,srl,woz.en,squad2,wikisql-srl,sst,woz.en,wikisql,squad2-woz.en,sst,srl,wikisql,squad2  openai-gpt,openai-gpt,gpt2 0.2 500 power power layer layer  3 >& experiments_five_tasks_gamma02_pretraincof500_gpu0.nohup &

in above example, experiments_five_tasks_gpu.sh requires 11 arguments, they are,
1. cuda=${1}              ### 0
2. seq_train_type=${2}    ### pn_calibration
3. tasks=${3}             ### sst,srl,woz.en,squad2,wikisql-srl,sst,woz.en,wikisql,squad2-woz.en,sst,srl,wikisql,squad2 
4. models=${4}            ### openai-gpt,openai-gpt,gpt2
5. gamma=${5}             ### 0.2
6. pretrain_cof=${6}      ### 500
7. encoder_norm_self=${7} ### power 
8. decoder_norm_self=${8} ### power
9. encoder_norm_ff=${9}   ### layer
10. decoder_norm_ff=${10} ### layer 
11. process_number=${11}  ### 3

In experiments_five_tasks_gpu.sh, will go through a loop as,

i=0
while [ $i -ne $process_number ]
do
        #i=$(($i+1))
        echo "$i"
        bash run_full_parameters.sh $cuda ./data ./models $seq_train_type ${task_group[$i]} ${model_arr[$i]} $gamma $pretrain_cof $encoder_norm_self $decoder_norm_self $encoder_norm_ff $decoder_norm_ff
done

The above loop means how many task group (namely, how many process_number, argument 11) will be run in a specific gpu, 0 for example. In our above example, three groups are set. The three group are separated by dash "-", sst,srl,woz.en,squad2,wikisql-srl,sst,woz.en,wikisql,squad2-woz.en,sst,srl,wikisql,squad2. Namely, we have five-task 1: sst,srl,woz.en,squad2,wikisql; five-task 2: srl,sst,woz.en,wikisql,squad2; and five-task 3: woz.en,sst,srl,wikisql,squad2 (they are the same five task with different orders or the same order with different model types, such as gpt1 or openai-gpt). The argument 5 refers to the sample proportion for replay for real data or for synthetic data. Argument 6 refers to the fisher information hyperparemeter. The arguments from 7 to 10 refer to the normalization types.

run_full_parameters.sh will call train_pn_calibration.sh and test_pn_calibration.sh. For simplicity, both train and test have the same number of arguments. Besides, both regularized memory recall and LAMO have the same arguments as well although LAMO does not need pretrain_cof at all.





