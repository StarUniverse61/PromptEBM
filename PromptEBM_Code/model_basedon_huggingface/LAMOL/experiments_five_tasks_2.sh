#!/bin/bash
nohup bash run.sh 0 ./data ./models pn_calibration srl,sst,woz.en,wikisql,squad2 openai-gpt > train_pn_calib_srl_sst_woz_wi_sq_openai-gpt.nohup 2>&1 &
nohup bash run.sh 1 ./data ./models pn_calibration srl,woz.en,sst,wikisql,squad2 openai-gpt > train_pn_calib_srl_woz_sst_wi_sq_openai-gpt.nohup 2>&1 &
nohup bash run.sh 2 ./data ./models pn_calibration woz.en,sst,srl,wikisql,squad2 openai-gpt > train_pn_calib_woz_sst_srl_wi_sq_openai-gpt.nohup 2>&1 &
nohup bash run.sh 3 ./data ./models pn_calibration woz.en,srl,sst,wikisql,squad2 openai-gpt > train_pn_calib_woz_srl_sst_wi_sq_openai-gpt.nohup 2>&1 &
nohup bash run.sh 4 ./data ./models pn_calibration sst,srl,woz.en,wikisql,squad2 gpt2 > train_pn_calib_sst_srl_woz_wi_sq_gpt2.nohup 2>&1 &
nohup bash run.sh 5 ./data ./models pn_calibration sst,woz.en,srl,wikisql,squad2 gpt2 > train_pn_calib_sst_woz_srl_wi_sq_gpt2.nohup 2>&1 &
nohup bash run.sh 6 ./data ./models pn_calibration srl,sst,woz.en,wikisql,squad2 gpt2 > train_pn_calib_srl_sst_woz_wi_sq_gpt2.nohup 2>&1 &
nohup bash run.sh 7 ./data ./models pn_calibration srl,woz.en,sst,wikisql,squad2 gpt2 > train_pn_calib_srl_woz_sst_wi_sq_gpt2.nohup 2>&1 &
