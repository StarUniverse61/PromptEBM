export PYTHONPATH=../../../:../../:../:`pwd`/:$PYTHONPATH
nohup python text_generation_from_pretrained.py --forward_only False --data_dir ../data --model_dir ./ebm_models --seq_train_type ebm --tasks ag yelp amazon yahoo dbpedia  --ebm_data lamol --gpu_idx 0 --model_name mistralai/Mistral-7B-v0.1 >text_generation_from_pretrained_mistral_classification.nohup &