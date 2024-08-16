#CUDA_VISIBLE_DEVICES=0 nohup bash train.sh --data_dir ./data --model_dir_root ./models --seq_train_type pn_calibration --tasks sst srl woz.en --model_name openai-gpt >& train_pn_calib_sst_srl_woz.nohup &
#CUDA_VISIBLE_DEVICES=0 nohup bash train.sh --data_dir ./data --model_dir_root ./models --seq_train_type pn_calibration --tasks sst srl woz.en --model_name gpt2 >& train_pn_calib_sst_srl_woz.nohup &
#CUDA_VISIBLE_DEVICES=0 nohup bash train.sh --data_dir ./data --model_dir_root ./models --seq_train_type pn_calibration --tasks sst woz.en srl --model_name openai-gpt >& train_pn_calib_sst_woz_srl.nohup &
#CUDA_VISIBLE_DEVICES=0 nohup bash train.sh --data_dir ./data --model_dir_root ./models --seq_train_type pn_calibration --tasks sst woz.en srl --model_name gpt2 >& train_pn_calib_sst_woz_srl.nohup &
#CUDA_VISIBLE_DEVICES=0 nohup bash train.sh --data_dir ./data --model_dir_root ./models --seq_train_type pn_calibration --tasks srl sst woz.en --model_name openai-gpt >& train_pn_calib_srl_sst_woz.nohup &
#CUDA_VISIBLE_DEVICES=0 nohup bash train.sh --data_dir ./data --model_dir_root ./models --seq_train_type pn_calibration --tasks srl sst woz.en --model_name gpt2 >& train_pn_calib_srl_sst_woz.nohup &
#CUDA_VISIBLE_DEVICES=0 nohup bash train.sh --data_dir ./data --model_dir_root ./models --seq_train_type pn_calibration --tasks srl woz.en sst --model_name openai-gpt >& train_pn_calib_srl_woz_sst.nohup &
#CUDA_VISIBLE_DEVICES=1 nohup bash train.sh --data_dir ./data --model_dir_root ./models --seq_train_type pn_calibration --tasks srl woz.en sst --model_name gpt2 >& train_pn_calib_srl_woz_sst.nohup &
#CUDA_VISIBLE_DEVICES=1 nohup bash train.sh --data_dir ./data --model_dir_root ./models --seq_train_type pn_calibration --tasks woz.en sst srl --model_name openai-gpt >& train_pn_calib_woz_sst_srl.nohup &
#CUDA_VISIBLE_DEVICES=1 nohup bash train.sh --data_dir ./data --model_dir_root ./models --seq_train_type pn_calibration --tasks woz.en sst srl --model_name gpt2 >& train_pn_calib_woz_sst_srl.nohup &
#CUDA_VISIBLE_DEVICES=1 nohup bash train.sh --data_dir ./data --model_dir_root ./models --seq_train_type pn_calibration --tasks woz.en srl sst --model_name openai-gpt >& train_pn_calib_woz_srl_sst.nohup &
#CUDA_VISIBLE_DEVICES=1 nohup bash train.sh --data_dir ./data --model_dir_root ./models --seq_train_type pn_calibration --tasks woz.en srl sst --model_name gpt2 >& train_pn_calib_woz_srl_sst.nohup &
#CUDA_VISIBLE_DEVICES=0 nohup bash test.sh --data_dir ./data --model_dir_root ./models --seq_train_type pn_calibration --tasks sst srl woz.en --model_name openai-gpt >& train_pn_calib_sst_srl_woz.nohup &
#CUDA_VISIBLE_DEVICES=0 nohup bash test.sh --data_dir ./data --model_dir_root ./models --seq_train_type pn_calibration --tasks sst srl woz.en --model_name gpt2 >& train_pn_calib_sst_srl_woz.nohup &
#CUDA_VISIBLE_DEVICES=0 nohup bash test.sh --data_dir ./data --model_dir_root ./models --seq_train_type pn_calibration --tasks sst woz.en srl --model_name openai-gpt >& train_pn_calib_sst_woz_srl.nohup &
#CUDA_VISIBLE_DEVICES=0 nohup bash test.sh --data_dir ./data --model_dir_root ./models --seq_train_type pn_calibration --tasks sst woz.en srl --model_name gpt2 >& train_pn_calib_sst_woz_srl.nohup &
#CUDA_VISIBLE_DEVICES=0 nohup bash test.sh --data_dir ./data --model_dir_root ./models --seq_train_type pn_calibration --tasks srl sst woz.en --model_name openai-gpt >& train_pn_calib_srl_sst_woz.nohup &
#CUDA_VISIBLE_DEVICES=0 nohup bash test.sh --data_dir ./data --model_dir_root ./models --seq_train_type pn_calibration --tasks srl sst woz.en --model_name gpt2 >& train_pn_calib_srl_sst_woz.nohup &
#CUDA_VISIBLE_DEVICES=0 nohup bash test.sh --data_dir ./data --model_dir_root ./models --seq_train_type pn_calibration --tasks srl woz.en sst --model_name openai-gpt >& train_pn_calib_srl_woz_sst.nohup &
#CUDA_VISIBLE_DEVICES=1 nohup bash test.sh --data_dir ./data --model_dir_root ./models --seq_train_type pn_calibration --tasks srl woz.en sst --model_name gpt2 >& train_pn_calib_srl_woz_sst.nohup &
#CUDA_VISIBLE_DEVICES=1 nohup bash test.sh --data_dir ./data --model_dir_root ./models --seq_train_type pn_calibration --tasks woz.en sst srl --model_name openai-gpt >& train_pn_calib_woz_sst_srl.nohup &
#CUDA_VISIBLE_DEVICES=1 nohup bash test.sh --data_dir ./data --model_dir_root ./models --seq_train_type pn_calibration --tasks woz.en sst srl --model_name gpt2 >& train_pn_calib_woz_sst_srl.nohup &
#CUDA_VISIBLE_DEVICES=1 nohup bash test.sh --data_dir ./data --model_dir_root ./models --seq_train_type pn_calibration --tasks woz.en srl sst --model_name openai-gpt >& train_pn_calib_woz_srl_sst.nohup &
#CUDA_VISIBLE_DEVICES=1 nohup bash test.sh --data_dir ./data --model_dir_root ./models --seq_train_type pn_calibration --tasks woz.en srl sst --model_name gpt2 >& train_pn_calib_woz_srl_sst.nohup &
CUDA_VISIBLE_DEVICES=2 nohup bash train.sh --data_dir ./data --model_dir_root ./models --seq_train_type pn_calibration --tasks sst srl woz.en --model_name openai-gpt --add_task_tokens >& train_pn_calib_sst_srl_woz.nohup &
CUDA_VISIBLE_DEVICES=2 nohup bash train.sh --data_dir ./data --model_dir_root ./models --seq_train_type pn_calibration --tasks sst srl woz.en --model_name gpt2 --add_task_tokens >& train_pn_calib_sst_srl_woz.nohup &
CUDA_VISIBLE_DEVICES=2 nohup bash train.sh --data_dir ./data --model_dir_root ./models --seq_train_type pn_calibration --tasks sst woz.en srl --model_name openai-gpt --add_task_tokens >& train_pn_calib_sst_woz_srl.nohup &
CUDA_VISIBLE_DEVICES=2 nohup bash train.sh --data_dir ./data --model_dir_root ./models --seq_train_type pn_calibration --tasks sst woz.en srl --model_name gpt2 --add_task_tokens >& train_pn_calib_sst_woz_srl.nohup &
CUDA_VISIBLE_DEVICES=2 nohup bash train.sh --data_dir ./data --model_dir_root ./models --seq_train_type pn_calibration --tasks srl sst woz.en --model_name openai-gpt --add_task_tokens >& train_pn_calib_srl_sst_woz.nohup &
CUDA_VISIBLE_DEVICES=2 nohup bash train.sh --data_dir ./data --model_dir_root ./models --seq_train_type pn_calibration --tasks srl sst woz.en --model_name gpt2 --add_task_tokens >& train_pn_calib_srl_sst_woz.nohup &
CUDA_VISIBLE_DEVICES=3 nohup bash train.sh --data_dir ./data --model_dir_root ./models --seq_train_type pn_calibration --tasks srl woz.en sst --model_name openai-gpt --add_task_tokens >& train_pn_calib_srl_woz_sst.nohup &
CUDA_VISIBLE_DEVICES=3 nohup bash train.sh --data_dir ./data --model_dir_root ./models --seq_train_type pn_calibration --tasks srl woz.en sst --model_name gpt2 --add_task_tokens >& train_pn_calib_srl_woz_sst.nohup &
CUDA_VISIBLE_DEVICES=3 nohup bash train.sh --data_dir ./data --model_dir_root ./models --seq_train_type pn_calibration --tasks woz.en sst srl --model_name openai-gpt --add_task_tokens >& train_pn_calib_woz_sst_srl.nohup &
CUDA_VISIBLE_DEVICES=3 nohup bash train.sh --data_dir ./data --model_dir_root ./models --seq_train_type pn_calibration --tasks woz.en sst srl --model_name gpt2 --add_task_tokens >& train_pn_calib_woz_sst_srl.nohup &
CUDA_VISIBLE_DEVICES=3 nohup bash train.sh --data_dir ./data --model_dir_root ./models --seq_train_type pn_calibration --tasks woz.en srl sst --model_name openai-gpt --add_task_tokens >& train_pn_calib_woz_srl_sst.nohup &
CUDA_VISIBLE_DEVICES=3 nohup bash train.sh --data_dir ./data --model_dir_root ./models --seq_train_type pn_calibration --tasks woz.en srl sst --model_name gpt2 --add_task_tokens >& train_pn_calib_woz_srl_sst.nohup &
CUDA_VISIBLE_DEVICES=2 nohup bash test.sh --data_dir ./data --model_dir_root ./models --seq_train_type pn_calibration --tasks sst srl woz.en --model_name openai-gpt --add_task_tokens >& train_pn_calib_sst_srl_woz.nohup &
CUDA_VISIBLE_DEVICES=2 nohup bash test.sh --data_dir ./data --model_dir_root ./models --seq_train_type pn_calibration --tasks sst srl woz.en --model_name gpt2 --add_task_tokens >& train_pn_calib_sst_srl_woz.nohup &
CUDA_VISIBLE_DEVICES=2 nohup bash test.sh --data_dir ./data --model_dir_root ./models --seq_train_type pn_calibration --tasks sst woz.en srl --model_name openai-gpt --add_task_tokens >& train_pn_calib_sst_woz_srl.nohup &
CUDA_VISIBLE_DEVICES=2 nohup bash test.sh --data_dir ./data --model_dir_root ./models --seq_train_type pn_calibration --tasks sst woz.en srl --model_name gpt2 --add_task_tokens >& train_pn_calib_sst_woz_srl.nohup &
CUDA_VISIBLE_DEVICES=2 nohup bash test.sh --data_dir ./data --model_dir_root ./models --seq_train_type pn_calibration --tasks srl sst woz.en --model_name openai-gpt --add_task_tokens >& train_pn_calib_srl_sst_woz.nohup &
CUDA_VISIBLE_DEVICES=2 nohup bash test.sh --data_dir ./data --model_dir_root ./models --seq_train_type pn_calibration --tasks srl sst woz.en --model_name gpt2 --add_task_tokens >& train_pn_calib_srl_sst_woz.nohup &
CUDA_VISIBLE_DEVICES=3 nohup bash test.sh --data_dir ./data --model_dir_root ./models --seq_train_type pn_calibration --tasks srl woz.en sst --model_name openai-gpt --add_task_tokens >& train_pn_calib_srl_woz_sst.nohup &
CUDA_VISIBLE_DEVICES=3 nohup bash test.sh --data_dir ./data --model_dir_root ./models --seq_train_type pn_calibration --tasks srl woz.en sst --model_name gpt2 --add_task_tokens >& train_pn_calib_srl_woz_sst.nohup &
CUDA_VISIBLE_DEVICES=3 nohup bash test.sh --data_dir ./data --model_dir_root ./models --seq_train_type pn_calibration --tasks woz.en sst srl --model_name openai-gpt --add_task_tokens >& train_pn_calib_woz_sst_srl.nohup &
CUDA_VISIBLE_DEVICES=3 nohup bash test.sh --data_dir ./data --model_dir_root ./models --seq_train_type pn_calibration --tasks woz.en sst srl --model_name gpt2 --add_task_tokens >& train_pn_calib_woz_sst_srl.nohup &
CUDA_VISIBLE_DEVICES=3 nohup bash test.sh --data_dir ./data --model_dir_root ./models --seq_train_type pn_calibration --tasks woz.en srl sst --model_name openai-gpt --add_task_tokens >& train_pn_calib_woz_srl_sst.nohup &
CUDA_VISIBLE_DEVICES=3 nohup bash test.sh --data_dir ./data --model_dir_root ./models --seq_train_type pn_calibration --tasks woz.en srl sst --model_name gpt2 --add_task_tokens >& train_pn_calib_woz_srl_sst.nohup &
