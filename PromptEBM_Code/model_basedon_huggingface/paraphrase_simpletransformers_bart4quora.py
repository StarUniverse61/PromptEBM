'''
https://towardsdatascience.com/bart-for-paraphrasing-with-simple-transformers-7c9ea3dfdd8c
'''

#wget http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv -P data
import warnings

import argparse
parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument(
    "--train_data_filename",
    #default='/efs-storage/jieha/share/hackson/rephrase/training_10-11-testing_12_100k',
    default='/efs-storage/data/quora/quora_train.tsv',
    type=str,
    required=False,
    help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
)
parser.add_argument(
    "--test_data_filename",
    #default='/efs-storage/jieha/share/hackson/rephrase/training_10-11-testing_12_100k',
    default='/efs-storage/data/quora/quora_test.tsv',
    type=str,
    required=False,
    help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
)
parser.add_argument(
    "--external_data_dir",
    #default='/efs-storage/jieha/share/hackson/rephrase/training_10-11-testing_12_100k',
    default='/home/ec2-user/workspaces/hoverboard-workspaces/src/data',
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
# parser.add_argument(
#     "--model_name_or_path",
#     default=None,
#     type=str,
#     required=False,
#     help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
# )
parser.add_argument(
    "--model_name_or_path",
    #default='/home/ec2-user/workspaces/hoverboard-workspaces/src/facebook/bart-large',
    default='output/output_bart/quora_qr_lll_v1/checkpoint-23340-epoch-30',
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
    #default='output/output_bart/alexa_qr',
    #default='output/output_bart/quora_qr_lll_1',
    default='output/output_bart/quora_qr_lll',
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
    default="/efs-storage/jieha/share/hackson/rephrase/training_10-11-testing_12_100k_cache_dir",
    type=str,
    help="Where do you want to store the pre-trained models downloaded from s3",
)
parser.add_argument(
    "--max_seq_length",
    default=64,
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
    "--cuda_device", default=0, type=int, help="cuda number, default is -1, refers to cuda:0 if use_gpu is True, use cpu if use_gpu is False.",
)

parser.add_argument(
    "--train_batch_size", default=48, type=int, help="Batch size per GPU/CPU for training.",
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
    "--num_train_epochs", default=30.0, type=float, help="Total number of training epochs to perform.",
)
parser.add_argument(
    "--max_steps",
    default=-1,
    type=int,
    help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
)
parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

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
parser.add_argument("--local_rank", type=int, default=0, help="For distributed training: local_rank")
parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")

# RecAdam parameters
parser.add_argument("--optimizer", type=str, default="RecAdam", choices=["Adam", "AdamW", "Adafactor", "RecAdam"],
                    help="Choose the optimizer to use. Default RecAdam.")
parser.add_argument("--recadam_anneal_fun", type=str, default='sigmoid', choices=["sigmoid", "linear", 'constant'],
                    help="the type of annealing function in RecAdam. Default sigmoid")
parser.add_argument("--recadam_anneal_k", type=float, default=0.5, help="k for the annealing function in RecAdam.")
parser.add_argument("--recadam_anneal_t0", type=int, default=1000, help="t0 for the annealing function in RecAdam.")
parser.add_argument("--recadam_anneal_w", type=float, default=1.0,
                    help="Weight for the annealing function in RecAdam. Default 1.0.")
parser.add_argument("--recadam_pretrain_cof", type=float, default=5000.0,
                    help="Coefficient of the quadratic penalty in RecAdam. Default 5000.0.")

parser.add_argument("--use_external", action="store_true",
                    help="Whether to use external resources，such as google or msr data")

parser.add_argument("--logging_Euclid_dist", action="store_true",
                    help="Whether to log the Euclidean distance between the pretrained model and fine-tuning model")
parser.add_argument("--start_from_pretrain", type=bool, default=True,
                    help="Whether to initialize the model with pretrained parameters")

parser.add_argument("--reprocess_input_data", action="store_true",
                    help="Whether to initialize the model with pretrained parameters")

parser.add_argument("--albert_dropout", default=0.0, type=float,
                    help="The dropout rate for the ALBERT model")
args = parser.parse_args()

def load_data(
    file_path, input_text_column, target_text_column, label_column, keep_label=1
):
    df = pd.read_csv(file_path, sep="\t", error_bad_lines=False)
    df = df.loc[df[label_column] == keep_label]
    df = df.rename(
        columns={input_text_column: "input_text", target_text_column: "target_text"}
    )
    df = df[["input_text", "target_text"]]
    df["prefix"] = "paraphrase"

    return df


def clean_unnecessary_spaces(out_string):
    if not isinstance(out_string, str):
        warnings.warn(f">>> {out_string} <<< is not a string.")
        out_string = str(out_string)
    out_string = (
        out_string.replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ,", ",")
        .replace(" ' ", "'")
        .replace(" n't", "n't")
        .replace(" 'm", "'m")
        .replace(" 's", "'s")
        .replace(" 've", "'ve")
        .replace(" 're", "'re")
    )
    return out_string
import os
from datetime import datetime
import logging

import pandas as pd
from sklearn.model_selection import train_test_split
#from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs
from simpletransformers.seq2seq import Seq2SeqArgs
from model_basedon_huggingface.seq2seq_model import Seq2SeqModel
import wandb
wandb.init
model_args = Seq2SeqArgs()
model_args.do_sample = True
model_args.eval_batch_size = 8
model_args.evaluate_during_training = args.evaluate_during_training
model_args.evaluate_during_training_steps = 2500
model_args.evaluate_during_training_verbose = True
model_args.fp16 = False
model_args.learning_rate = 5e-5
model_args.max_length = 128
model_args.max_seq_length = args.max_seq_length
model_args.num_beams = None
model_args.num_return_sequences = 3
model_args.num_train_epochs = args.num_train_epochs
model_args.overwrite_output_dir = True
model_args.reprocess_input_data = args.reprocess_input_data
model_args.save_eval_checkpoints = False
model_args.save_steps = -1
model_args.top_k = 50
model_args.top_p = 0.95
model_args.train_batch_size = args.train_batch_size
model_args.optimizer = args.optimizer
model_args.output_dir = args.output_dir
model_args.best_model_dir = args.output_dir +'/best_model'
model_args.cuda_device = args.cuda_device
model_args.n_gpu = args.n_gpu
model_args.local_rank = args.local_rank
model_args.cache_dir = args.cache_dir
model_args.use_multiprocessing = False
#model_args.wandb_project = "Paraphrasing with BART"
model_args.start_from_pretrain = args.start_from_pretrain
model_args.external_data_dir = args.external_data_dir
model_args.model_name_or_path = args.model_name_or_path
model_args.train_data_filename = args.train_data_filename
model_args.test_data_filename = args.test_data_filename
# model = Seq2SeqModel(
#     encoder_decoder_type="bart",
#     encoder_decoder_name="facebook/bart-large",
#     args=model_args,
#     use_cuda=False,
#     cuda_device=args.cuda_device ##if use_cuda=False, cuda_device=-1, yet, use_cuda=True, cuda_device=-1, means we use cuda:0
# )
# model = Seq2SeqModel(
#     encoder_decoder_type="bart",
#     encoder_decoder_name="/efs-storage/intDFS_allenv12/DeepQueryRewritingTools/query_rewriting/push_button_recipe/outputs",
#     args=model_args,
#     use_cuda=True
# )
#from utils import load_data, clean_unnecessary_spaces
# Google Data
# train_df = pd.read_csv("/home/ec2-user/workspaces/hoverboard-workspaces/src/data/train.tsv", sep="\t").astype(str)
# eval_df = pd.read_csv("/home/ec2-user/workspaces/hoverboard-workspaces/src/data/dev.tsv", sep="\t").astype(str)
#train_df = pd.read_csv("/Users/lidingch/Documents/DFS_works/data/paws_wiki_labeled_final/train.tsv", sep="\t").astype(str)
#eval_df = pd.read_csv("/Users/lidingch/Documents/DFS_works/data/paws_wiki_labeled_final/dev.tsv", sep="\t").astype(str)
# # MSRP Data
# train_df = pd.concat(
#     [
#         train_df,
#         load_data("data/msr_paraphrase_train.txt", "#1 String", "#2 String", "Quality"),
#     ]
# )
# eval_df = pd.concat(
#     [
#         eval_df,
#         load_data("data/msr_paraphrase_test.txt", "#1 String", "#2 String", "Quality"),
#     ]
# )

# Quora Data
# The Quora Dataset is not separated into train/test, so we do it manually the first time.

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.ERROR)
model_args.use_external = False

if model_args.use_external:
    # (will add them later)
    # Google Data
    train_df = pd.read_csv(model_args.external_data_dir+"/train.tsv", sep="\t").astype(str)
    eval_df = pd.read_csv(model_args.external_data_dir+"/dev.tsv", sep="\t").astype(str)
    # train_df = pd.read_csv("/Users/lidingch/Documents/DFS_works/data/paws_wiki_labeled_final/train.tsv", sep="\t").astype(str)
    # eval_df = pd.read_csv("/Users/lidingch/Documents/DFS_works/data/paws_wiki_labeled_final/dev.tsv", sep="\t").astype(str)
    train_df = train_df.loc[train_df["label"] == "1"]
    eval_df = eval_df.loc[eval_df["label"] == "1"]

    train_df = train_df.rename(
        columns={"sentence1": "input_text", "sentence2": "target_text"}
    )
    eval_df = eval_df.rename(
        columns={"sentence1": "input_text", "sentence2": "target_text"}
    )

    train_df = train_df[["input_text", "target_text"]]
    eval_df = eval_df[["input_text", "target_text"]]

    train_df["prefix"] = "paraphrase"
    eval_df["prefix"] = "paraphrase"
    # MSR Data

if not os.path.exists("/efs-storage/data/quora/quora_train.tsv"):
    df = load_data(
        model_args.external_data_dir+"/quora_duplicate_questions.tsv", "question1", "question2", "is_duplicate"
    )
    # df = load_data(
    #     "/Users/lidingch/Documents/DFS_works/data/Quora/sample_test_data/quora_duplicate_questions.tsv", "question1", "question2", "is_duplicate"
    # )

    q_train, q_test = train_test_split(df)

    q_train.to_csv("/efs-storage/data/quora/quora_train.tsv", sep="\t")
    q_test.to_csv("/efs-storage/data/quora/quora_test.tsv", sep="\t")
    #q_train.to_csv("/Users/lidingch/Documents/DFS_works/data/Quora/sample_test_data/quora_train.tsv", sep="\t")
    #q_test.to_csv("/Users/lidingch/Documents/DFS_works/data/Quora/sample_test_data/quora_test.tsv", sep="\t")
    # The code block above only needs to be run once.
    # After that, the two lines below are sufficient to load the Quora dataset.

    #q_train.to_csv(args.data_dir + "/train.tsv", sep="\t")
    #q_test.to_csv(args.data_dir + "/test.tsv", sep="\t")

    # q_train = pd.read_csv("data/quora_train.tsv", sep="\t")
    # q_test = pd.read_csv("data/quora_test.tsv", sep="\t")
    #q_test = pd.read_csv("/efs-storage/data/quora_test.tsv", sep="\t")
#q_train = pd.read_csv(args.data_dir + "/train.tsv", sep="\t")
#q_train = pd.read_csv("/efs-storage/jieha/share/hackson/rephrase/training_10-11-testing_12_1k/train.tsv", sep="\t")
#q_valid = pd.read_csv(args.data_dir + "/valid.tsv", sep="\t")
q_valid = pd.read_csv(args.external_data_dir + "/dev.tsv", sep="\t")
#q_test = pd.read_csv(args.data_dir + "/test.tsv", sep="\t")

# q_train = pd.read_csv("/efs-storage/data/quora/quora_train_1.tsv", sep="\t")
# q_test = pd.read_csv("/efs-storage/data/quora/quora_test_1.tsv", sep="\t")
q_train = pd.read_csv(model_args.train_data_filename, sep="\t")
q_test = pd.read_csv(model_args.test_data_filename, sep="\t")

# train_df = q_train
valid_df = q_valid
# eval_df = q_test

##in fact, all labels are 1. So, the following lines just for references, not really functional
#train_df = q_train.loc[q_train["label"] == 1]
valid_df = q_valid.loc[q_valid["label"] == 1]
#eval_df = q_test.loc[q_test["label"] == 1]


valid_df = valid_df.rename(
    columns={"sentence1": "input_text", "sentence2": "target_text"}
)

if model_args.use_external:
    train_df = pd.concat([train_df, q_train])
    eval_df = pd.concat([eval_df, q_test])
else:
    train_df = q_train
    eval_df = q_test


train_df = train_df[["input_text", "target_text"]]
valid_df = valid_df[["input_text", "target_text"]]
eval_df = eval_df[["input_text", "target_text"]]

train_df["prefix"] = "paraphrase"
valid_df["prefix"] = "paraphrase"
eval_df["prefix"] = "paraphrase"



train_df = train_df[["prefix", "input_text", "target_text"]]
valid_df = valid_df[["prefix", "input_text", "target_text"]]
eval_df = eval_df[["prefix", "input_text", "target_text"]]

train_df = train_df.dropna()
valid_df = valid_df.dropna()
eval_df = eval_df.dropna()

train_df["input_text"] = train_df["input_text"].apply(clean_unnecessary_spaces)
train_df["target_text"] = train_df["target_text"].apply(clean_unnecessary_spaces)
valid_df["input_text"] = valid_df["input_text"].apply(clean_unnecessary_spaces)
valid_df["target_text"] = valid_df["target_text"].apply(clean_unnecessary_spaces)
eval_df["input_text"] = eval_df["input_text"].apply(clean_unnecessary_spaces)
eval_df["target_text"] = eval_df["target_text"].apply(clean_unnecessary_spaces)

print(train_df)
print(train_df.size)
print(train_df.columns)
print(valid_df)
print(valid_df.size)
print(eval_df)
print(eval_df.size)




# from transformers import AutoTokenizer, AutoModel
# tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
# model = AutoModel.from_pretrained("facebook/bart-large")
#
#
#

model = Seq2SeqModel(
     encoder_decoder_type="bart",
     encoder_decoder_name="/home/ec2-user/workspaces/hoverboard-workspaces/src/facebook/bart-large",
     #encoder_decoder_name="/efs-storage/intDFS_allenv12/DeepQueryRewritingTools/query_rewriting/push_button_recipe/paraphrase_bart_transformer/output/output_bart/quora_qr_lll_v1/",
     #encoder_decoder_name=model_args.model_name_or_path,
     args=model_args,
     use_cuda=True,
     #cuda_device=args.cuda_device ##if use_cuda=False, cuda_device=-1, yet, use_cuda=True, cuda_device=-1, means we use cuda:0
)
model.train_model(train_df, eval_data=valid_df, args=args)
'''
if model trained, we can use the following two lines
model_dir='outputs/best_model/'
model._load_model_args(model_dir)
'''

to_predict = [
    prefix + ": " + str(input_text) for prefix, input_text in zip(eval_df["prefix"].tolist(), eval_df["input_text"].tolist())
]
truth = eval_df["target_text"].tolist()

prefix = 'paraphrase'
print('model training done and start to do final evaluations')
for line in q_test:
   print(line)

'''
id
setence1
sentence2
label
'''
# test_source = q_test['setence1']
# test_target = q_test['sentence2']
#
# to_predict = [prefix +": " + str(line) for line in test_source.tolist()]
# truth = test_target.tolist()

#model.args.eval_batch_size=16 ##avoid OOM

'''
note sure why we need the following, eval data from google rather than from alexa

to_predict = [
    prefix + ": " + str(input_text) for prefix, input_text in zip(eval_df["prefix"].tolist(), eval_df["input_text"].tolist())
]
truth = eval_df["target_text"].tolist()
'''
preds = model.predict(to_predict)

# Saving the predictions if needed
#os.makedirs("/efs-storage/data/quora_predictions", exist_ok=True)
if not os.path.exists(f"{args.output_dir}/predictions/"):
    os.makedirs(f"{args.output_dir}/predictions/")
with open(f"{args.output_dir}/predictions/predictions_{datetime.now()}.txt", "w") as f:
    for i, text in enumerate(eval_df["input_text"].tolist()):
        f.write(str(text) + "\n\n")

        f.write("Truth:\n")
        f.write(truth[i] + "\n\n")

        f.write("Prediction:\n")
        for pred in preds[i]:
            f.write(str(pred) + "\n")
        f.write(
            "________________________________________________________________________________\n"
        )

'''
let's write another output file only hypo in order to keep consistent with Jie's eval
'''
#output_dir='output/output_bart/alexa_qr_data_training_10-11-testing_12_100k/predictions'
with open(f"{args.output_dir}/predictions/test_{datetime.now()}.hypo", "w") as f:
    for i, text in enumerate(eval_df["input_text"].tolist()):
        for j, pred in enumerate(preds[i]):
            if j == 0:
                pred = str(pred)[10:]
                if ':' in pred:
                    pred=pred[1:].strip()
                else:
                    pred=pred.strip()
                f.write(str(pred) + "\n")

with open(f"{args.output_dir}/test.target_", "w") as f:
    for line in truth:
        f.write(line+'\n')

'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
The following is copied from seq2seq_model.py for debugging purpose
'''
# use_cuda = True
# #cuda_device = -1
# silent=True
# encoder_decoder_type = 'bart'
# encoder_decoder_name="/home/ec2-user/workspaces/hoverboard-workspaces/src/facebook/bart-large"
# model.args = model._load_model_args(encoder_decoder_name)
# additional_special_tokens_encoder = None
# additional_special_tokens_decoder = None
# config = None,
# if isinstance(args, dict):
#     model.args.update_from_dict(args)
# elif isinstance(args, Seq2SeqArgs):
#     model.args = args
#
# # if "sweep_config" in kwargs:
# #     model.is_sweeping = True
# #     sweep_config = kwargs.pop("sweep_config")
# #     sweep_values = sweep_config_to_sweep_values(sweep_config)
# #     model.args.update_from_dict(sweep_values)
# # else:
# model.is_sweeping = False
#
# if model.args.manual_seed:
#     random.seed(model.args.manual_seed)
#     np.random.seed(model.args.manual_seed)
#     torch.manual_seed(model.args.manual_seed)
#     if model.args.n_gpu > 0:
#         torch.cuda.manual_seed_all(model.args.manual_seed)
#
#
# if use_cuda:
#     if torch.cuda.is_available():
#         if cuda_device == -1:
#             model.device = torch.device("cuda")
#         else:
#             model.device = torch.device(f"cuda:{cuda_device}")
#     else:
#         raise ValueError(
#             "'use_cuda' set to True when cuda is unavailable."
#             "Make sure CUDA is available or set `use_cuda=False`."
#         )
# else:
#     model.device = "cpu"
#
# model.results = {}
#
# if not use_cuda:
#     model.args.fp16 = False
#
# # config = EncoderDecoderConfig.from_encoder_decoder_configs(config, config)
# if encoder_decoder_type:
#     config_class, model_class, tokenizer_class = MODEL_CLASSES[encoder_decoder_type]
# else:
#     config_class, model_class, tokenizer_class = MODEL_CLASSES[encoder_type]
#
# if encoder_decoder_type in ["bart", "mbart", "marian"]:
#     #model.model = model_class.from_pretrained(encoder_decoder_name)
#     if encoder_decoder_type in ["bart", "mbart"]:
#         model.pretrained_model = model_class.from_pretrained(encoder_decoder_name)
#         if encoder_decoder_type in ["bart", "mbart"]:
#             model.encoder_tokenizer = tokenizer_class.from_pretrained(encoder_decoder_name)
#         elif encoder_decoder_type == "marian":
#             if model.args.base_marian_model_name:
#                 model.encoder_tokenizer = tokenizer_class.from_pretrained(model.args.base_marian_model_name)
#             else:
#                 model.encoder_tokenizer = tokenizer_class.from_pretrained(encoder_decoder_name)
#
#         if args.start_from_pretrain:
#             model.new_model = model_class.from_pretrained(encoder_decoder_name)
#         else:
#             #model.new_model = model_class(config=config)
#             model.new_model = model_class(config=config)
#
#     elif encoder_decoder_type == "marian":
#         if model.args.base_marian_model_name:
#             model.encoder_tokenizer = tokenizer_class.from_pretrained(model.args.base_marian_model_name)
#         else:
#             model.encoder_tokenizer = tokenizer_class.from_pretrained(encoder_decoder_name)
#     model.decoder_tokenizer = model.encoder_tokenizer
#     model.config = model.new_model.config
# else:
#     if encoder_decoder_name:
#         # model.model = EncoderDecoderModel.from_pretrained(encoder_decoder_name)
#         # model.model = EncoderDecoderModel.from_encoder_decoder_pretrained(
#         #     os.path.join(encoder_decoder_name, "encoder"), os.path.join(encoder_decoder_name, "decoder")
#         # )
#
#         model.pretrained_model = EncoderDecoderModel.from_encoder_decoder_pretrained(
#             os.path.join(encoder_decoder_name, "encoder"), os.path.join(encoder_decoder_name, "decoder"),
#             # from_tf=bool(".ckpt" in args.model_name_or_path),
#             config=config,
#             # cache_dir=args.cache_dir if args.cache_dir else None,
#         )
#
#         if args.start_from_pretrain:
#             model.new_model = EncoderDecoderModel.from_encoder_decoder_pretrained(
#                 os.path.join(encoder_decoder_name, "encoder"), os.path.join(encoder_decoder_name, "decoder"),
#                 from_tf=bool(".ckpt" in args.model_name_or_path),
#                 config=config,
#                 cache_dir=args.cache_dir if args.cache_dir else None,
#             )
#         else:
#             model.new_model = model_class(config=config)
#
#         # model.pretrained_model = model_class.from_pretrained(
#         #     args.model_name_or_path,
#         #     # from_tf=bool(".ckpt" in args.model_name_or_path),
#         #     config=config,
#         #     # cache_dir=args.cache_dir if args.cache_dir else None,
#         # )
#         # if args.start_from_pretrain:
#         #     model.new_model = model_class.from_pretrained(
#         #         args.model_name_or_path,
#         #         from_tf=bool(".ckpt" in args.model_name_or_path),
#         #         config=config,
#         #         cache_dir=args.cache_dir if args.cache_dir else None,
#         #     )
#         # else:
#         #     model.new_model = model_class(config=config)
#
#         model.encoder_tokenizer = tokenizer_class.from_pretrained(os.path.join(encoder_decoder_name, "encoder"))
#         model.decoder_tokenizer = AutoTokenizer.from_pretrained(os.path.join(encoder_decoder_name, "decoder"))
#     else:
#         model.new_model = EncoderDecoderModel.from_encoder_decoder_pretrained(
#             encoder_name, decoder_name, config=config
#         )
#         model.encoder_tokenizer = tokenizer_class.from_pretrained(encoder_name)
#         model.decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_name)
#     model.encoder_config = model.new_model.config.encoder
#     model.decoder_config = model.new_model.config.decoder
#
# if additional_special_tokens_encoder is not None:
#     model.encoder_tokenizer.add_special_tokens(additional_special_tokens_encoder)
#
# if additional_special_tokens_decoder is not None:
#     model.decoder_tokenizer.add_special_tokens(additional_special_tokens_decoder)
#
# if model.args.wandb_project and not wandb_available:
#     warnings.warn("wandb_project specified but wandb is not available. Wandb disabled.")
#     model.args.wandb_project = None
#
# # `model_name` could be provided in args
# if model.args.model_name is None:
#     if encoder_decoder_name:
#         model.args.model_name = encoder_decoder_name
#
#         # # Checking if we are loading from a saved model or using a pre-trained model
#         # if not saved_model_args and encoder_decoder_type == "marian":
#         # Need to store base pre-trained model name to get the tokenizer when loading a saved model
#         model.args.base_marian_model_name = encoder_decoder_name
#
#     elif encoder_name and decoder_name:
#         model.args.model_name = encoder_name + "-" + decoder_name
#     else:
#         model.args.model_name = "encoder-decoder"
#
#     if encoder_decoder_type:
#         model.args.model_type = encoder_decoder_type
#     elif encoder_type:
#         model.args.model_type = encoder_type + "-bert"
#     else:
#         model.args.model_type = "encoder-decoder"
#
#
# '''
# the following is for training
# '''
#
# new_args = {}
# eval_data=valid_df
# train_data=train_df
# output_dir=None
# show_running_loss=True
# args=args
# verbose=True
# #for arg in args:
# #    new_args[arg] = args.arg
# new_args['model_type'] = args.model_type
# new_args['recadam_anneal_fun']=args.recadam_anneal_fun
# new_args['recadam_anneal_k']=args.recadam_anneal_k
# new_args['recadam_anneal_t0']=args.recadam_anneal_t0
# new_args['recadam_pretrain_cof']=args.recadam_pretrain_cof
# new_args['recadam_anneal_w']=args.recadam_anneal_w
# new_args['start_from_pretrain'] = args.start_from_pretrain
#
# if args:
#     model.args.update_from_dict(new_args)
#
# # if model.args.silent:
# #     show_running_loss = False
#
# if model.args.evaluate_during_training and eval_data is None:
#     raise ValueError(
#         "evaluate_during_training is enabled but eval_data is not specified."
#         " Pass eval_data to model.train_model() if using evaluate_during_training."
#     )
#
# if not output_dir:
#     output_dir = model.args.output_dir
#
# if os.path.exists(output_dir) and os.listdir(output_dir) and not model.args.overwrite_output_dir:
#     raise ValueError(
#         "Output directory ({}) already exists and is not empty."
#         " Set args.overwrite_output_dir = True to overcome.".format(output_dir)
#     )
#
# model._move_model_to_device()
#
# train_dataset = model.load_and_cache_examples(train_data, verbose=verbose)
# #eval_dataset = model.load_and_cache_examples(eval_data, evaluate=True, verbose=verbose, silent=silent)
# '''
# for simplcity, let's use train_data for eval_dataset since train_data I am using now only has 1000 instances
# '''
# eval_dataset = model.load_and_cache_examples(train_data, evaluate=True, verbose=verbose, silent=silent)
# os.makedirs(output_dir, exist_ok=True)
#
# '''
# the following is copied from train of seq2seq_model.py
# '''
# show_running_loss=True
# verbose=True,
#
# new_model = model.new_model
# pretrained_model = model.pretrained_model
# args = model.args
#
# tb_writer = SummaryWriter(logdir=args.tensorboard_dir)
# train_sampler = RandomSampler(train_dataset)
# train_dataloader = DataLoader(
#     train_dataset,
#     sampler=train_sampler,
#     batch_size=args.train_batch_size,
#     num_workers=model.args.dataloader_num_workers,
# )
#
# if args.max_steps > 0:
#     t_total = args.max_steps
#     args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
# else:
#     t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
#
# no_decay = ["bias", "LayerNorm.weight"]
#
# optimizer_grouped_parameters = []
# custom_parameter_names = set()
# for group in model.args.custom_parameter_groups:
#     params = group.pop("params")
#     custom_parameter_names.update(params)
#     param_group = {**group}
#     param_group["params"] = [p for n, p in new_model.named_parameters() if n in params]
#     optimizer_grouped_parameters.append(param_group)
#
# for group in model.args.custom_layer_parameters:
#     layer_number = group.pop("layer")
#     layer = f"layer.{layer_number}."
#     group_d = {**group}
#     group_nd = {**group}
#     group_nd["weight_decay"] = 0.0
#     params_d = []
#     params_nd = []
#     for n, p in new_model.named_parameters():
#         if n not in custom_parameter_names and layer in n:
#             if any(nd in n for nd in no_decay):
#                 params_nd.append(p)
#             else:
#                 params_d.append(p)
#             custom_parameter_names.add(n)
#     group_d["params"] = params_d
#     group_nd["params"] = params_nd
#
#     optimizer_grouped_parameters.append(group_d)
#     optimizer_grouped_parameters.append(group_nd)
#
# if not model.args.train_custom_parameters_only:
#     optimizer_grouped_parameters.extend(
#         [
#             {
#                 "params": [
#                     p
#                     for n, p in new_model.named_parameters()
#                     if n not in custom_parameter_names and not any(nd in n for nd in no_decay)
#                 ],
#                 "weight_decay": args.weight_decay,
#             },
#             {
#                 "params": [
#                     p
#                     for n, p in new_model.named_parameters()
#                     if n not in custom_parameter_names and any(nd in n for nd in no_decay)
#                 ],
#                 "weight_decay": 0.0,
#             },
#         ]
#     )
#
# warmup_steps = math.ceil(t_total * args.warmup_ratio)
# args.warmup_steps = warmup_steps if args.warmup_steps == 0 else args.warmup_steps
#
# # TODO: Use custom optimizer like with BertSum?
# if args.optimizer == "AdamW":
#     optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
# elif args.optimizer == "Adafactor":
#     optimizer = Adafactor(
#         optimizer_grouped_parameters,
#         lr=args.learning_rate,
#         eps=args.adafactor_eps,
#         clip_threshold=args.adafactor_clip_threshold,
#         decay_rate=args.adafactor_decay_rate,
#         beta1=args.adafactor_beta1,
#         weight_decay=args.weight_decay,
#         scale_parameter=args.adafactor_scale_parameter,
#         relative_step=args.adafactor_relative_step,
#         warmup_init=args.adafactor_warmup_init,
#     )
#     print("Using Adafactor for T5")
# elif args.optimizer == 'RecAdam':
#     # Prepare for the grouped parameters for RecAdam optimizer.
#     # Since the classifier layer is not pretrained, it is not penalized during optimization.
#     optimizer_grouped_parameters = [
#         {
#             "params": [p for n, p in new_model.named_parameters() if
#                        not any(nd in n for nd in no_decay) and args.model_type in n],
#             "weight_decay": args.weight_decay,
#             "anneal_w": args.recadam_anneal_w,
#             "pretrain_params": [p_p for p_n, p_p in pretrained_model.named_parameters() if
#                                 not any(nd in p_n for nd in no_decay) and args.model_type in p_n]
#         },
#         {
#             "params": [p for n, p in new_model.named_parameters() if
#                        not any(nd in n for nd in no_decay) and args.model_type not in n],
#             "weight_decay": args.weight_decay,
#             "anneal_w": 0.0,
#             "pretrain_params": [p_p for p_n, p_p in pretrained_model.named_parameters() if
#                                 not any(nd in p_n for nd in no_decay) and args.model_type not in p_n]
#         },
#         {
#             "params": [p for n, p in model.named_parameters() if
#                        any(nd in n for nd in no_decay) and args.model_type in n],
#             "weight_decay": 0.0,
#             "anneal_w": args.recadam_anneal_w,
#             "pretrain_params": [p_p for p_n, p_p in pretrained_model.named_parameters() if
#                                 any(nd in p_n for nd in no_decay) and args.model_type in p_n]
#         },
#         {
#             "params": [p for n, p in new_model.named_parameters() if
#                        any(nd in n for nd in no_decay) and args.model_type not in n],
#             "weight_decay": 0.0,
#             "anneal_w": 0.0,
#             "pretrain_params": [p_p for p_n, p_p in pretrained_model.named_parameters() if
#                                 any(nd in p_n for nd in no_decay) and args.model_type not in p_n]
#         }
#     ]
#     optimizer = RecAdam(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon,
#                         anneal_fun=args.recadam_anneal_fun, anneal_k=args.recadam_anneal_k,
#                         anneal_t0=args.recadam_anneal_t0, pretrain_cof=args.recadam_pretrain_cof)
# else:
#     raise ValueError(
#         "{} is not a valid optimizer class. Please use one of ('AdamW', 'Adafactor') instead.".format(
#             args.optimizer
#         )
#     )
#
#
# '''
# the following is for evaluations
# '''
# new_model = model.new_model
# args = model.args
# eval_output_dir = output_dir
#
# results = {}
#
# eval_sampler = SequentialSampler(eval_dataset)
# eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
#
# if args.n_gpu > 1:
#     new_model = torch.nn.DataParallel(new_model)
#
# eval_loss = 0.0
# nb_eval_steps = 0
# new_model.eval()
#
# #batch_group = list(tqdm(eval_dataloader, disable=args.silent or silent, desc="Running Evaluation"))
#
# # for batch in batch_group:
# #     inputs = model._get_inputs_dict(batch)
# #     outputs = new_model(**inputs)
#
# for batch in tqdm(eval_dataloader, disable=args.silent or silent, desc="Running Evaluation"):
#     # batch = tuple(t.to(device) for t in batch)
#
#     inputs = model._get_inputs_dict(batch)
#     with torch.no_grad():
#         if model.args.fp16:
#             with amp.autocast():
#                 outputs = model(**inputs)
#                 tmp_eval_loss = outputs[0]
#         else:
#             outputs = model(**inputs)
#             tmp_eval_loss = outputs[0]
#         if model.args.n_gpu > 1:
#             tmp_eval_loss = tmp_eval_loss.mean()
#         eval_loss += tmp_eval_loss.item()
#     nb_eval_steps += 1
#
