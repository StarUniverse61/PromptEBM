export PYTHONPATH=../../../:../../:../:`pwd`/:$PYTHONPATH
nohup python text_generation_from_pretrained.py  --forward_only False --data_dir ../data --model_dir ./ebm_models --tasks sst srl woz.en --ebm_data lamol --gpu_idx 0 --model_name gpt2 --latent_size 80 --max_utt_len 80 --max_dec_len 80 >text_generation_from_pretrained_gpt2.nohup &