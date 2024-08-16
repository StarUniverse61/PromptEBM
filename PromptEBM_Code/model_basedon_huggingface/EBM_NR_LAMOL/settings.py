import os
import json
import argparse
import logging
import datetime
logger = logging.getLogger(__name__)

import GPUtil
#from pytorch_transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, OpenAIGPTConfig
#from pytorch_transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, CONFIG_NAME
from pytorch_transformers import GPT2Tokenizer, CONFIG_NAME
from pytorch_transformers import OpenAIGPTTokenizer, OpenAIGPTConfig
from model_basedon_huggingface.EBM_NR_LAMOL.modeling_gpt2 import GPT2LMHeadModel, GPT2Config
from model_basedon_huggingface.EBM_NR_LAMOL.modeling_openai import OpenAIGPTLMHeadModel
from model_basedon_huggingface.EBM_NR_LAMOL.ebm_models.dgmvae.models.sent_models import GMVAE, GMVAE4Lamol
from transformers import AutoModelForCausalLM, AutoTokenizer, MistralConfig, MISTRAL_PRETRAINED_CONFIG_ARCHIVE_MAP
import torch
from model_basedon_huggingface.EBM_NR_LAMOL.ebm_models.dgmvae.utils import str2bool

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILL_VAL = -1
LEN_FACTOR = 1.163
MEMORY_FACTOR = {
    "finetune": 0.58,
    "multitask": 0.58,
    "lll": 0.35,
    "ebm": 0.35,
    "ewc": 0.30,
    "mas": 0.18,
    "gem": 0.50,
    "pn_calibration": 0.35,
    "pn_calib_ebm": 0.35
}
TURING_ARCHS = {'Tesla V100', '2080 Ti'}
MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer, GPT2Config),
    'openai-gpt': (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, OpenAIGPTConfig),
    'mistralai/Mistral-7B-v0.1': (AutoModelForCausalLM, AutoTokenizer, MistralConfig)
    #'gmvae': ([GPT2LMHeadModel,GMVAE], OpenAIGPTTokenizer, OpenAIGPTConfig)
}
SAVE_NAME = 'model-'
FINAL_SAVE_NAME = 'model-finish'

def add_default_training_parser(parser):
    parser.add_argument('--op', type=str, default='adam')
    parser.add_argument('--backward_size', type=int, default=5)
    parser.add_argument('--step_size', type=int, default=1)
    parser.add_argument('--grad_clip', type=float, default=0.5)
    parser.add_argument('--prior_grad_clip', type=float, default=1)
    parser.add_argument('--init_w', type=float, default=0.08)
    parser.add_argument('--init_lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--lr_hold', type=int, default=3)
    parser.add_argument('--lr_decay', type=str2bool, default=True)
    parser.add_argument('--lr_decay_rate', type=float, default=0.8)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--improve_threshold', type=float, default=0.996)
    parser.add_argument('--patient_increase', type=float, default=2.0)
    parser.add_argument('--early_stop', type=str2bool, default=True)
    # parser.add_argument('--max_epoch', type=int, default=60)
    parser.add_argument('--max_epoch', type=int, default=20)
    parser.add_argument('--save_model', type=str2bool, default=True)
    parser.add_argument('--use_gpu', type=str2bool, default=True)
    parser.add_argument('--print_step', type=int, default=100)
    parser.add_argument('--fix_batch', type=str2bool, default=False)
    parser.add_argument('--ckpt_step', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=60)
    parser.add_argument('--preview_batch_num', type=int, default=1)
    parser.add_argument('--gen_type', type=str, default='greedy')
    parser.add_argument('--avg_type', type=str, default='seq')
    parser.add_argument('--beam_size', type=int, default=10)
    parser.add_argument('--forward_only', type=str2bool, default=True)
    parser.add_argument('--load_sess', type=str, default="", help="Load model directory.")
    parser.add_argument('--debug', type=bool, default=False)
    #parser.add_argument('--seed', type=int, default=300)
    #parser.add_argument('--model_file', type=str, default='ckpts/ptb/model_ckpt.pt')
    return parser

def add_default_variational_training_parser(parser):
    # KL-annealing
    parser.add_argument('--anneal', type=str2bool, default=True)
    parser.add_argument('--anneal_function', type=str, default='logistic')
    parser.add_argument('--anneal_k', type=float, default=0.0025)
    parser.add_argument('--anneal_x0', type=int, default=2500)
    parser.add_argument('--anneal_warm_up_step', type=int, default=0)
    parser.add_argument('--anneal_warm_up_value', type=float, default=0.000)

    # Word dropout & posterior sampling number
    parser.add_argument('--word_dropout_rate', type=float, default=0.0)
    parser.add_argument('--post_sample_num', type=int, default=1)
    parser.add_argument('--sel_metric', type=str, default="elbo", help="select best checkpoint base on what metric.",
                        choices=['elbo', 'obj'],)

    # Other:
    parser.add_argument('--aggressive', type=str2bool, default=False)
    return parser

def add_default_ebm_data_parser(parser):
    # Data & logging path
    parser.add_argument('--ebm_data', type=str, default='ptb')
    parser.add_argument('--ebm_data_dir', type=str, default='/home/xiaodi/NLP/Lifelonglearning-main/Lifelonglearning_For_NLG_v3/ibebm/data/ptb')
    parser.add_argument('--ebm_log_dir', type=str, default='/home/xiaodi/NLP/Lifelonglearning-main/Lifelonglearning_For_NLG_v3/ibebm/logs/ptb/dgmvae')
    # Draw points
    parser.add_argument('--ebm_fig_dir', type=str, default='/home/xiaodi/NLP/Lifelonglearning-main/Lifelonglearning_For_NLG_v3/ibebm/figs')
    parser.add_argument('--ebm_draw_points', type=str2bool, default=False)
    return parser

def get_parser(model_class="sent_models"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--adam_epsilon", default=1e-4, type=float)
    parser.add_argument("--add_task_tokens", action="store_true")
    parser.add_argument("--data_dir", type=str, required=True)
    #parser.add_argument("--debug", action="store_true")
    parser.add_argument('--gpu_idx', type=int, default=1)
    parser.add_argument("--decay_style", type=str, default="linear")
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--real_sample", action="store_true")
    parser.add_argument("--unbound", type=int, default=0)
    parser.add_argument("--gen_lm_sample_percentage", type=float, default=0.05)
    parser.add_argument("--learning_rate", type=float, default=6.25e-5)
    parser.add_argument("--logging_steps", type=int, default=1000)
    parser.add_argument("--lm_lambda", type=float, default=0.25)
    parser.add_argument("--lr_schedule", type=str, default="warmup_linear")
    parser.add_argument("--max_grad_norm", type=int, default=1)
    parser.add_argument("--max_n_epochs", type=int, default=9)
    parser.add_argument("--min_batch_size", type=int, default=16)
    parser.add_argument("--min_n_steps", type=int, default=1500)
    parser.add_argument("--model_dir_root", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="gpt2", choices=["gpt2", "openai-gpt", "mistralai/Mistral-7B-v0.1"])
    #parser.add_argument('--ebm_model_file', type=str, default='/home/dingcheng/workspace/baidu/ccl/Lifelonglearning/Lifelonglearning_For_NLG_v3/ebm_models/ckpts/ptb/model_ckpt.pt')
    parser.add_argument('--ebm_model_file', type=str,
                        default='/home/xiaodi//NLP/Lifelonglearning-main/Lifelonglearning_For_NLG_v3/ibebm/logs/ptb/dgmvae/2022-05-11T20-43-13-text_generation.py/model_ckpt.pt')
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--n_train_epochs", type=int, default=1)
    parser.add_argument("--dynamic_epochs", action="store_true")
    parser.add_argument("--n_warmup_ratio", type=float, default=0.005)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--use_sep", action="store_true")
    parser.add_argument("--reg_lambda", type=float, default=1.)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq_train_type", type=str, default="lll", choices=["lll","ebm","finetune","multitask","mas","ewc","gem","pn_calibration","pn_calib_ebm"])
    parser.add_argument("--tasks", nargs='+', default=["squad2"])
    parser.add_argument("--skip_tasks", nargs='+')
    parser.add_argument("--temperature_lm", type=float, default=1.0)
    parser.add_argument("--temperature_qa", type=float, default=1.0)
    parser.add_argument("--test_batch_size", type=int, default=0)
    parser.add_argument("--tokens_weight", type=float, default=5)
    parser.add_argument("--top_k_lm", type=int, default=20)
    parser.add_argument("--top_k_qa", type=int, default=20)
    parser.add_argument("--top_p_lm", type=float, default=0.)
    parser.add_argument("--top_p_qa", type=float, default=0.)
    parser.add_argument("--train_batch_size", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--qp_margin", type=float, default=0.5)
    parser.add_argument("--recadam_anneal_fun", type=str, default='sigmoid', choices=["sigmoid", "linear", 'constant'],
                        help="the type of annealing function in RecAdam. Default sigmoid")
    parser.add_argument("--recadam_anneal_k", type=float, default=0.5, help="k for the annealing function in RecAdam.")
    parser.add_argument("--recadam_anneal_t0", type=int, default=1000, help="t0 for the annealing function in RecAdam.")
    parser.add_argument("--recadam_anneal_w", type=float, default=1.0,
                        help="Weight for the annealing function in RecAdam. Default 1.0.")
    parser.add_argument("--recadam_pretrain_cof", type=float, default=5000.0,
                        help="Coefficient of the quadratic penalty in RecAdam. Default 5000.0.")
    parser.add_argument("--encoder_norm_self", type=str, default="layer", choices=["layer", "power"])
    parser.add_argument("--decoder_norm_self", type=str, default="layer", choices=["layer", "power"])
    parser.add_argument("--encoder_norm_ff", type=str, default="layer", choices=["layer", "power"])
    parser.add_argument("--decoder_norm_ff", type=str, default="layer", choices=["layer", "power"])
    parser.add_argument("--warmup_updates", default=1000, type=float, help="warm up updates")
    parser.add_argument('--ebm_model', type=str, default="GMVAE4Lamol")
    parser = add_default_ebm_data_parser(parser)
    parser = add_default_training_parser(parser)
    parser = add_default_variational_training_parser(parser)
    config, unparsed = parser.parse_known_args()
    try:
        model_name = config.ebm_model
        model_class = eval(model_name)
        parser = model_class.add_args(parser)
    except Exception as e:
        raise ValueError("Wrong model" + config.ebm_model)

    args, _ = parser.parse_known_args()
    print(args)
    # config = process_config(config)
    #args = parser.parse_args()
    #print(args)
    if args.debug:
        args.logging_steps = 1
        torch.manual_seed(0)
        torch.backends.cudnn.deterministric = True

    args.model_dir_root = os.path.join(args.model_dir_root, args.model_name,
            args.seq_train_type, "{}_{}".format("_".join(args.tasks),
                args.gen_lm_sample_percentage) if "lll" in args.seq_train_type else "_".join(args.tasks))

    #args.device_ids = GPUtil.getAvailable(maxLoad=0.05, maxMemory=0.05, limit=args.n_gpus)
    #print(f'device_ids[0] by calling GPUtil.getAvailable in settings.py ={args.device_ids[0]}')
    args.device_ids = [args.gpu_idx]
    if len(args.device_ids) == 0:
        print(f'length of device ids in settings.py = {len(args.device_ids)}')
        logger.error('No available GPUs!')
        raise NotImplementedError("No CPU mode available!")

    if len(args.device_ids) < args.n_gpus:
        logger.warning('Available number of GPU = {} < n_gpus = {}'.format(len(args.device_ids), args.n_gpus))
        args.n_gpus = len(args.device_ids)
        logger.warning('Continue training with {} GPUs'.format(args.n_gpus))
    print(f'device_ids[0]={args.device_ids[0]}')
    torch.cuda.set_device(args.device_ids[0])

    gpus = GPUtil.getGPUs()
    gpu_names = [gpus[device_id].name for device_id in args.device_ids]
    if not all(any(turing_arch in gpu_name for turing_arch in TURING_ARCHS) for gpu_name in gpu_names):
        logger.warning('Not all gpus support fp16 training! Will use fp32 instead.')
        args.fp32 = True
    if args.model_name == "openai-gpt":
        args.fp32 = True  # openai-gpt currently doesn't support fp16
    if not args.fp32:
        global MEMORY_FACTOR
        MEMORY_FACTOR = dict([k, v*1.4] for k, v in MEMORY_FACTOR.items())
    args.memory_sizes = [gpus[device_id].memoryTotal for device_id in args.device_ids]
    args.memory_sizes[0] = args.memory_sizes[0] * (1 - 0.04 * (args.n_gpus-1))
    for i in range(1, args.n_gpus):
        args.memory_sizes[i] = args.memory_sizes[i] * 1.04
    if args.train_batch_size <= 0:
        args.train_batch_size = [int(memory_size * MEMORY_FACTOR[args.seq_train_type]) for memory_size in args.memory_sizes]
    if args.test_batch_size <= 0:
        args.test_batch_size = [int(memory_size * MEMORY_FACTOR[args.seq_train_type]) for memory_size in args.memory_sizes]

    special_tokens = {"ans_token":'__ans__', "pad_token":'__pad__', "unk_token":'__unk__', "eos_token": '<|endoftext|>'}
    if args.use_sep:
        special_tokens["sep_token"] = '__sep__'
    model_class, tokenizer_class, config_class = MODEL_CLASSES[args.model_name]
    # config_class.pretrained_config_archive_map['encoder_norm_self'] = args.encoder_norm_self
    # config_class.pretrained_config_archive_map['decoder_norm_self'] = args.decoder_norm_self
    # config_class.pretrained_config_archive_map['encoder_norm_ff'] = args.encoder_norm_ff
    # config_class.pretrained_config_archive_map['decoder_norm_ff'] = args.decoder_norm_ff
    # config_class.pretrained_config_archive_map['warmup_updates'] = args.warmup_updates

    # MISTRAL_PRETRAINED_CONFIG_ARCHIVE_MAP['encoder_norm_self'] = args.encoder_norm_self
    # MISTRAL_PRETRAINED_CONFIG_ARCHIVE_MAP['decoder_norm_self'] = args.decoder_norm_self
    # MISTRAL_PRETRAINED_CONFIG_ARCHIVE_MAP['encoder_norm_ff'] = args.encoder_norm_ff
    # MISTRAL_PRETRAINED_CONFIG_ARCHIVE_MAP['decoder_norm_ff'] = args.decoder_norm_ff
    # MISTRAL_PRETRAINED_CONFIG_ARCHIVE_MAP['warmup_updates'] = args.warmup_updates
    tokenizer = tokenizer_class.from_pretrained(args.model_name)
    model_config = config_class.from_pretrained(args.model_name)
    tokenizer.add_tokens(list(special_tokens.values()))
    special_token_ids = {k:tokenizer.convert_tokens_to_ids(v) for k,v in special_tokens.items()}

    model_config.vocab_size = len(tokenizer)
    tokens_weight = torch.ones([model_config.vocab_size], dtype=torch.float).cuda()
    tokens_weight[special_token_ids["ans_token"]] = args.tokens_weight
    if args.use_sep:
        tokens_weight[special_token_ids["sep_token"]] = args.tokens_weight

    # args.max_len = model_config.n_positions
    args.max_len = 512

    data_attrs_path = os.path.join(BASE_DIR,"data_attrs.json")
    assert os.path.exists(data_attrs_path)
    with open(data_attrs_path, "r") as f:
        data_attrs = json.load(f)

    if args.seq_train_type == "multitask":
        args.n_train_epochs = {'_'.join(args.tasks): args.n_train_epochs}
    elif args.unbound:
        pass
    else:
        if "gem" in args.seq_train_type:
            args.memory_data = []
        if args.dynamic_epochs:
            data_sizes = {task: data_attrs[task]["train"]["data_size"] for task in args.tasks}
            max_total_data_size = max(data_sizes.values()) * args.n_train_epochs
            args.n_train_epochs = {d[0]: min(args.max_n_epochs, max_total_data_size//d[1]) for d in data_sizes.items()}
        else:
            args.n_train_epochs = {task: args.n_train_epochs for task in args.tasks}

    return args, model_config, model_class, tokenizer, config_class, special_token_ids, special_tokens, data_attrs, tokens_weight, parser

    #return config

# def parse_args():
#
#     # # Data & logging path
#     # parser.add_argument('--ebm_data', type=str, default='ptb')
#     # parser.add_argument('--ebm_data_dir', type=str, default='data/ptb')
#     # parser.add_argument('--ebm_log_dir', type=str, default='logs/ptb/dgmvae')
#     # # Draw points
#     # parser.add_argument('--ebm_fig_dir', type=str, default='figs')
#     # parser.add_argument('--ebm_draw_points', type=str2bool, default=False)
#     args = parser.parse_args()
#     return args


class TimeFilter(logging.Filter):
    def filter(self, record):
        try:
          last = self.last
        except AttributeError:
          last = record.relativeCreated

        delta = record.relativeCreated/1000 - last/1000
        record.relative = "{:.1f}".format(delta)
        record.uptime = str(datetime.timedelta(seconds=record.relativeCreated//1000))
        self.last = record.relativeCreated
        return True


def init_logging(filename):
    logging_format = "%(asctime)s - %(uptime)s - %(relative)ss - %(levelname)s - %(name)s - %(message)s"
    logging.basicConfig(format=logging_format, filename=filename, filemode='a', level=logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(logging_format))
    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)
    for handler in root_logger.handlers:
        handler.addFilter(TimeFilter())

##args, model_config, model_class, tokenizer, config_class, special_token_ids, special_tokens, data_attrs, tokens_weight
args, MODEL_CONFIG, MODEL_CLASS, TOKENIZER, CONFIG_CLASS, SPECIAL_TOKEN_IDS, SPECIAL_TOKENS, DATA_ATTRS, TOKENS_WEIGHT, parser = get_parser()


TASK_DICT = {
    "squad1": {
               "train":os.path.join(args.data_dir,"squad-train-v1.1.json"),
               "eval":os.path.join(args.data_dir,"squad-dev-v1.1.json"),
               "test":os.path.join(args.data_dir,"squad-dev-v1.1.json"),
               "n_train_epochs": args.n_train_epochs
    },
    "squad2": {
               "train":os.path.join(args.data_dir,"squad2-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"squad2-dev-v2.0.json"),
               "test":os.path.join(args.data_dir,"squad2-dev-v2.0.json"),
               "n_train_epochs": args.n_train_epochs
    },
    "covid": {
               "train":os.path.join(args.data_dir,"covid_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"covid_to_squad-dev-v2.0.json"),
               "test":os.path.join(args.data_dir,"covid_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs
    },
    "tweet": {
               "train":os.path.join(args.data_dir,"tweet_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"tweet_to_squad-dev-v2.0.json"),
               "test":os.path.join(args.data_dir,"tweet_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs
    },
    "selfrc": {
               "train":os.path.join(args.data_dir,"selfrc_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"selfrc_to_squad-dev-v2.0.json"),
               "test":os.path.join(args.data_dir,"selfrc_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs
    },
    "webmd": {
               "train":os.path.join(args.data_dir,"webmd_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"webmd_to_squad-dev-v2.0.json"),
               "test":os.path.join(args.data_dir,"webmd_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs
    },
    "iwslt.en.de": {
               "train":os.path.join(args.data_dir,"iwslt.en.de_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"iwslt.en.de_to_squad-dev-v2.0.json"),
               "test":os.path.join(args.data_dir,"iwslt.en.de_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs
    },
    "cnn_dailymail": {
               "train":os.path.join(args.data_dir,"cnn_dailymail_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"cnn_dailymail_to_squad-dev-v2.0.json"),
               "test":os.path.join(args.data_dir,"cnn_dailymail_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs
    },
    "multinli.in.out": {
               "train":os.path.join(args.data_dir,"multinli.in.out_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"multinli.in.out_to_squad-dev-v2.0.json"),
               "test":os.path.join(args.data_dir,"multinli.in.out_to_squad-dev-v2.0.json"),
               "n_train_epochs": args.n_train_epochs
    },
    "sst": {
               "train":os.path.join(args.data_dir,"sst_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"sst_to_squad-dev-v2.0.json"),
               "test":os.path.join(args.data_dir,"sst_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs
    },
    "srl": {
               "train":os.path.join(args.data_dir,"srl_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"srl_to_squad-dev-v2.0.json"),
               "test":os.path.join(args.data_dir,"srl_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs
    },
    "zre": {
               "train":os.path.join(args.data_dir,"zre_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"zre_to_squad-dev-v2.0.json"),
               "test":os.path.join(args.data_dir,"zre_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs
    },
    "woz.en": {
               "train":os.path.join(args.data_dir,"woz.en_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"woz.en_to_squad-dev-v2.0.json"),
               "test":os.path.join(args.data_dir,"woz.en_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs
    },
    "wikisql": {
               "train":os.path.join(args.data_dir,"wikisql_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"wikisql_to_squad-dev-v2.0.json"),
               "test":os.path.join(args.data_dir,"wikisql_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs
    },
    "schema": {
               "train":os.path.join(args.data_dir,"schema_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"schema_to_squad-dev-v2.0.json"),
               "test":os.path.join(args.data_dir,"schema_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs
    },
    "ag": {
               "train":os.path.join(args.data_dir,"ag_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"ag_to_squad-test-v2.0.json"),
               "test":os.path.join(args.data_dir,"ag_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs
    },
    "dbpedia": {
               "train":os.path.join(args.data_dir,"dbpedia_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"dbpedia_to_squad-test-v2.0.json"),
               "test":os.path.join(args.data_dir,"dbpedia_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs
    },
    "yahoo": {
               "train":os.path.join(args.data_dir,"yahoo_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"yahoo_to_squad-test-v2.0.json"),
               "test":os.path.join(args.data_dir,"yahoo_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs
    },
    "amazon": {
               "train":os.path.join(args.data_dir,"amazon_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"amazon_to_squad-test-v2.0.json"),
               "test":os.path.join(args.data_dir,"amazon_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs
    },
    "yelp": {
               "train":os.path.join(args.data_dir,"yelp_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"yelp_to_squad-test-v2.0.json"),
               "test":os.path.join(args.data_dir,"yelp_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs
    },
}
