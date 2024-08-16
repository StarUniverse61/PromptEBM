#!/bin/bash
nohup bash ./experiments_five_tasks_gpu.sh 0 lll sst,srl,woz.en-srl,sst,woz.en  openai-gpt,openai-gpt 0.05 500 power power layer layer ppll 2 >& experiments_three_tasks_gamma02_pretraincof500_adamw_openaigpt_ppll_gpu0.nohup &
#nohup bash ./experiments_five_tasks_gpu.sh 1 lll sst,srl,woz.en,squad2,wikisql-srl,sst,woz.en,wikisql,squad2-woz.en,sst,srl,wikisql,squad2  openai-gpt,openai-gpt,openai-gpt 0.05 500 layer layer layer layer llll 3 >& experiments_five_tasks_gamma02_pretraincof500_adamw_openaigpt_llll_gpu1.nohup &
#nohup bash ./experiments_five_tasks_gpu.sh 2 lll sst,srl,woz.en,squad2,wikisql-srl,sst,woz.en,wikisql,squad2-woz.en,sst,srl,wikisql,squad2  gpt2,gpt2,gpt2 0.05 500 power power layer layer ppll 3 >& experiments_five_tasks_gamma02_pretraincof500_adamw_gpt2_ppll_gpu2.nohup &
#nohup bash ./experiments_five_tasks_gpu.sh 3 lll sst,srl,woz.en,squad2,wikisql-srl,sst,woz.en,wikisql,squad2-woz.en,sst,srl,wikisql,squad2  gpt2,gpt2,gpt2 0.05 500 layer layer layer layer llll 3 >& experiments_five_tasks_gamma02_pretraincof500_adamw_gpt2_llll_gpu3.nohup &
#nohup bash ./experiments_five_tasks_gpu.sh 4 lll sst,srl,woz.en,squad2,wikisql-srl,sst,woz.en,wikisql,squad2-woz.en,sst,srl,wikisql,squad2  openai-gpt,openai-gpt,gpt2 0.05 500 layer layer layer layer llll 3 >& experiments_five_tasks_gamma02_pretraincof500_adamw_llll_gpu4.nohup &
#nohup bash ./experiments_five_tasks_gpu.sh 0 lll sst,srl,woz.en,squad2,wikisql-srl,sst,woz.en,wikisql,squad2-woz.en,sst,srl,wikisql,squad2  openai-gpt,openai-gpt,gpt2 0.05 500 power power layer layer ppll 3 >& experiments_five_tasks_gamma02_pretraincof500_gpu0.nohup &
#nohup bash ./experiments_five_tasks_gpu.sh 1 lll sst,woz.en,srl,squad2,wikisql-srl,woz.en,sst,wikisql,squad2-woz.en,srl,sst,wikisql,squad2  openai-gpt,openai-gpt,gpt2 0.05 500 power power layer layer ppll 3 >& experiments_five_tasks_gamma02_pretraincof500_gpu1.nohup &
#nohup bash ./experiments_five_tasks_gpu.sh 2 lll srl,sst,woz.en,squad2,wikisql-woz.en,sst,srl,wikisql,squad2-sst,srl,woz.en,squad2,wikisql  openai-gpt,openai-gpt,gpt2 0.05 500 power power layer layer ppll 3 >& experiments_five_tasks_gamma02_pretraincof500_gpu2.nohup &
#nohup bash ./experiments_five_tasks_gpu.sh 3 lll srl,woz.en,sst,squad2,wikisql-woz.en,srl,sst,wikisql,squad2-sst,woz.en,srl,squad2,wikisql  openai-gpt,gpt2,gpt2 0.05 500 power power layer layer ppll 3 >& experiments_five_tasks_gamma02_pretraincof500_gpu3.nohup &
#nohup bash ./experiments_five_tasks_gpu.sh 4 lll woz.en,sst,srl,squad2,wikisql-sst,srl,woz.en,wikisql,squad2-srl,sst,woz.en,squad2,wikisql  openai-gpt,gpt2,gpt2 0.05 500 power power layer layer ppll 3 >& experiments_five_tasks_gamma02_pretraincof500_gpu4.nohup &
#nohup bash ./experiments_five_tasks_gpu.sh 5 lll woz.en,srl,sst,squad2,wikisql-sst,woz.en,srl,wikisql,squad2-srl,woz.en,sst,squad2,wikisql  openai-gpt,gpt2,gpt2 0.05 500 power power layer layer ppll 3 >& experiments_five_tasks_gamma02_pretraincof500_gpu5.nohup &
#nohup bash ./experiments_five_tasks_gpu.sh 6 lll sst,srl,woz.en,wikisql,squad2-srl,sst,woz.en,wikisql,squad2-woz.en,sst,srl,squad2,wikisql  openai-gpt,gpt2,gpt2 0.05 500 power power layer layer ppll 3 >& experiments_five_tasks_gamma02_pretraincof500_gpu6.nohup &
#nohup bash ./experiments_five_tasks_gpu.sh 7 lll sst,woz.en,srl,wikisql,squad2-srl,woz.en,sst,wikisql,squad2  openai-gpt,gpt2 0.05 500 power power layer layer ppll 2 >& experiments_five_tasks_gamma02_pretraincof500_gpu7.nohup &

#nohup bash ./experiments_five_tasks_gpu1.sh >& experiments_five_tasks_gpu1 &
#nohup bash ./experiments_five_tasks_gpu2.sh >& experiments_five_tasks_gpu2 &
#nohup bash ./experiments_five_tasks_gpu3.sh >& experiments_five_tasks_gpu3 &
#nohup bash ./experiments_five_tasks_gpu4.sh >& experiments_five_tasks_gpu4 &
#nohup bash ./experiments_five_tasks_gpu5.sh >& experiments_five_tasks_gpu5 &
#nohup bash ./experiments_five_tasks_gpu6.sh >& experiments_five_tasks_gpu6 &
#nohup bash ./experiments_five_tasks_gpu7.sh >& experiments_five_tasks_gpu7 &
#nohup bash ./experiments_five_tasks_gpu8.sh >& experiments_five_tasks_gpu8 &
# declare an array called array and define 3 vales
#array=( 1 2 3 )
#for i in "${array[@]}"
#do
#	echo "$i"
#	echo experiments_five_tasks_$i.sh
#done