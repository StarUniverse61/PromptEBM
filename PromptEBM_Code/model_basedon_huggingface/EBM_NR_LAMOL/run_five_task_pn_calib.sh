#!/bin/bash
nohup bash ./experiments_five_tasks_gpu.sh 0 pn_calibration sst,srl,woz.en,squad2,wikisql-sst,woz.en,srl,squad2,wikisql-srl,sst,woz.en,squad2,wikisql gpt2,gpt2,gpt2 0.05 500 power power layer layer ppll 8 3 >& experiments_five_tasks_gamma005_pretraincof500_gpt2_ppll_gpu0.nohup &
nohup bash ./experiments_five_tasks_gpu.sh 1 pn_calibration srl,woz.en,sst,squad2,wikisql-woz.en,sst,srl,squad2,wikisql-woz.en,srl,sst,squad2,wikisql gpt2,gpt2,gpt2 0.05 500 power power layer layer ppll 8 3 >& experiments_five_tasks_gamma005_pretraincof500_gpt2_ppll_gpu1.nohup &
nohup bash ./experiments_five_tasks_gpu.sh 2 pn_calibration sst,srl,woz.en,squad2,wikisql-sst,woz.en,srl,squad2,wikisql-srl,sst,woz.en,squad2,wikisql openai-gpt,openai-gpt,openai-gpt 0.05 500 layer layer layer layer llll 16 3 >& experiments_five_tasks_gamma005_pretraincof500_openaigpt_llll_gpu2.nohup &
nohup bash ./experiments_five_tasks_gpu.sh 3 pn_calibration srl,woz.en,sst,squad2,wikisql-woz.en,sst,srl,squad2,wikisql-woz.en,srl,sst,squad2,wikisql openai-gpt,openai-gpt,openai-gpt 0.05 500 layer layer layer layer llll 16 3 >& experiments_five_tasks_gamma005_pretraincof500_openaigpt_llll_gpu3.nohup &
nohup bash ./experiments_five_tasks_gpu.sh 4 pn_calibration sst,srl,woz.en,squad2,wikisql-sst,woz.en,srl,squad2,wikisql-srl,sst,woz.en,squad2,wikisql openai-gpt,openai-gpt,openai-gpt 0.05 500 power power layer layer ppll 8 3 >& experiments_five_tasks_gamma005_pretraincof500_gpt2_ppll_gpu4.nohup &
nohup bash ./experiments_five_tasks_gpu.sh 5 pn_calibration srl,woz.en,sst,squad2,wikisql-woz.en,sst,srl,squad2,wikisql-woz.en,srl,sst,squad2,wikisql openai-gpt,openai-gpt,openai-gpt 0.05 500 power power layer layer ppll 8 3 >& experiments_five_tasks_gamma005_pretraincof500_gpt2_ppll_gpu5.nohup &
nohup bash ./experiments_five_tasks_gpu.sh 6 pn_calibration sst,srl,woz.en,squad2,wikisql-sst,woz.en,srl,squad2,wikisql-srl,sst,woz.en,squad2,wikisql gpt2,gpt2,gpt2 0.05 500 layer layer layer layer llll 16 3 >& experiments_five_tasks_gamma005_pretraincof500_openaigpt_llll_gpu6.nohup &
nohup bash ./experiments_five_tasks_gpu.sh 7 pn_calibration srl,woz.en,sst,squad2,wikisql-woz.en,sst,srl,squad2,wikisql-woz.en,srl,sst,squad2,wikisql gpt2,gpt2,gpt2  0.05 500 layer layer layer layer llll 16 3 >& experiments_five_tasks_gamma005_pretraincof500_openaigpt_llll_gpu7.nohup &


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