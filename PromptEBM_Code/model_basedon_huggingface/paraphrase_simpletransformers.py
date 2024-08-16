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
    default='output/output_bart/quora_adamw/checkpoint-27750-epoch-30',
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
    #default='output/output_bart/quora_qr_lll_v1',
    default='output/output_bart/quora_adamw',
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
parser.add_argument("--optimizer", type=str, default="AdamW", choices=["Adam", "AdamW", "Adafactor", "RecAdam"],
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
from model_basedon_huggingface.seq2seq_bart_model import Seq2SeqModel
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

#from utils import load_data, clean_unnecessary_spaces


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.ERROR)
model_args.use_external = False

if model_args.use_external:
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

#model_args.wandb_project = "Paraphrasing with BART"
# model = Seq2SeqModel(
#     encoder_decoder_type="bart",
#     encoder_decoder_name="facebook/bart-large",
#     args=model_args,
#     use_cuda=False
# )
#model_args.output_dir = 'output/output_bart/quora_adamw_v2/'
encoder_decoder_name = "/home/ec2-user/workspaces/hoverboard-workspaces/src/facebook/bart-large"
#encoder_decoder_name= 'output/output_bart/quora_adamw/checkpoint-27750-epoch-30/'
model = Seq2SeqModel(
    encoder_decoder_type="bart",
    #encoder_decoder_name=encoder_decoder_name,
    encoder_decoder_name=model_args.model_name_or_path,
    args=model_args,
    use_cuda=True
)
# model = Seq2SeqModel(
#     encoder_decoder_type="bart",
#     encoder_decoder_name="/efs-storage/intDFS_allenv12/DeepQueryRewritingTools/query_rewriting/push_button_recipe/outputs",
#     args=model_args,
#     use_cuda=True
# )

# from transformers import AutoTokenizer, AutoModel
# tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
# model = AutoModel.from_pretrained("facebook/bart-large")
model.train_model(train_df, eval_data=eval_df)


to_predict = [
    prefix + ": " + str(input_text)
    for prefix, input_text in zip(eval_df["prefix"].tolist(), eval_df["input_text"].tolist())
]
truth = eval_df["target_text"].tolist()

preds = model.predict(to_predict)

# Saving the predictions if needed
#os.makedirs("/efs-storage/data/quora_predictions", exist_ok=True)

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

