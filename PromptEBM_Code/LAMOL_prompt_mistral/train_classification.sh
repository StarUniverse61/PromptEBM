export PYTHONPATH=:../:../:../:../:../:../:`pwd`/:$PYTHONPATH
nohup python train.py  --data_dir /home/xiaodi/NLP/LAMOL/data  --model_dir_root /home/xiaodi/NLP/Lifelonglearning-main/Lifelonglearning_For_NLG_v3/LAMOL_prompt_mistral/model --model_name mistralai/Mistral-7B-v0.1 --seq_train_type ebm --max_vocab_cnt 32000 --tasks yelp ag dbpedia amazon yahoo --ebm_data lamol --gen_lm_sample_percentage 0.05 >train_classification4_prompt_mistral_0.05.nohup &