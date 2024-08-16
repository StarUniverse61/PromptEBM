1. Go to model_basedon_huggingface -> EBM_NR_LAMOL -> ebm_models -> bash train_text_generation_mistral_new.sh
2. get the directory name of the trained EBM model in ibebm -> logs -> ptb -> dgmvae
3. copy the directory name of the trained EBM model and paste it to LAMOL_prompt_mistral -> settings.py -> 
modify the following arguments:
parser.add_argument('--ebm_log_dir', type=str, default='parent directory of the trained EBM model')
parser.add_argument('--ebm_model_file', type=str, default='dir of the trained EBM model')
4. Go to LAMOL_prompt_mistral -> bash train_3tasks.sh -> bash test_3tasks.sh