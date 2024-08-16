
def get_para_argparser():
    import warnings

    import pandas as pd
    import argparse
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default='/data/dingcheng/workspace/baidu/ccl/Lifelonglearning_For_NLG/data/twitter/data4fairseq/',
        # default='/home/ec2-user/workspaces/hoverboard-workspaces/src/data/quora_duplicate_questions.tsv',
        type=str,
        required=False,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default='bart',
        type=str,
        required=False,
        help="Model type selected in the list",
    )
    parser.add_argument(
        "--encoder_norm_self",
        default='power',
        type=str,
        required=False,
        help="encoder norm type selected in the list",
    )
    parser.add_argument(
        "--decoder_norm_self",
        default='power',
        type=str,
        required=False,
        help="decoder norm type selected in the list",
    )
    parser.add_argument(
        "--encoder_norm_ff",
        default='layer',
        type=str,
        required=False,
        help="encoder norm ff type selected in the list",
    )
    parser.add_argument(
        "--decoder_norm_ff",
        default='layer',
        type=str,
        required=False,
        help="decoder norm ff type selected in the list",
    )
    parser.add_argument(
        "--num_return_sequences",
        default=5,
        type=int,
        required=False,
        help="number of return sequencess: ",
    )
    parser.add_argument(
        "--model_name_or_path",
        default='/data/dingcheng/workspace/baidu/ccl/Lifelonglearning_For_NLG/data/Quora/output_recadam_batch48/best_model',
        type=str,
        required=False,
        help="Path to pre-trained model or shortcut name selected in the list: ",
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=False,
        help="The name of the task to train selected in the list: ",
    )
    parser.add_argument(
        "--output_dir",
        default='/data/dingcheng/workspace/baidu/ccl/Lifelonglearning_For_NLG/data/twitter/output_crladam',
        type=str,
        required=False,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--log_path",
        default='log/bart_qqp_log',
        type=str,
        required=False,
        help="Path to the logging file.", )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="/data/dingcheng/workspace/baidu/ccl/Lifelonglearning_For_NLG/data/twitter/cache_dir",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=64, ## 128 is the original one, looks too large
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--n_gpu", default=8, type=int, help="the nubmer of gpu.",
    )

    parser.add_argument(
        "--cuda_device", default=0, type=int,
        help="cuda number, default is -1, refers to cuda:0 if use_gpu is True, use cpu if use_gpu is False.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=1, type=int, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    #warmup_updates: float = 1000
    parser.add_argument("--warmup_updates", default=1000, type=float, help="warm up updates")
    parser.add_argument("--train_logging_steps", type=int, default=100, help="Log training info every X updates steps.")
    parser.add_argument("--eval_logging_steps", type=int, default=500, help="Evaluate every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=5000000, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")

    # RecAdam parameters
    parser.add_argument("--optimizer", type=str, default="RecAdam", choices=["AdamW", "RecAdam", "CrlAdam"],
                        help="Choose the optimizer to use. Default RecAdam.")
    parser.add_argument("--recadam_anneal_fun", type=str, default='sigmoid', choices=["sigmoid", "linear", 'cWonstant'],
                        help="the type of annealing function in RecAdam. Default sigmoid")
    parser.add_argument("--recadam_anneal_k", type=float, default=0.5, help="k for the annealing function in RecAdam.")
    parser.add_argument("--recadam_anneal_t0", type=int, default=1000, help="t0 for the annealing function in RecAdam.")
    parser.add_argument("--recadam_anneal_w", type=float, default=1.0,
                        help="Weight for the annealing function in RecAdam. Default 1.0.")
    parser.add_argument("--recadam_pretrain_cof", type=float, default=5000.0,
                        help="Coefficient of the quadratic penalty in RecAdam. Default 5000.0.")

    parser.add_argument("--use_external", action="store_true",
                        help="Whether to use external resources")

    parser.add_argument("--logging_Euclid_dist", action="store_true",
                        help="Whether to log the Euclidean distance between the pretrained model and fine-tuning model")
    parser.add_argument("--start_from_pretrain", type=bool, default=True,
                        help="Whether to initialize the model with pretrained parameters")

    parser.add_argument("--reprocess_input_data", action="store_true",
                        help="Whether to initialize the model with pretrained parameters")

    parser.add_argument("--albert_dropout", default=0.0, type=float,
                        help="The dropout rate for the ALBERT model")
    args = parser.parse_args()
    return args