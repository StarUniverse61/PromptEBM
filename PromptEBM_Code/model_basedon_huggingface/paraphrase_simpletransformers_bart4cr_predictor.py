'''
https://towardsdatascience.com/bart-for-paraphrasing-with-simple-transformers-7c9ea3dfdd8c
'''

#wget http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv -P data
import warnings

import argparse
parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument(
    "--data_dir",
    default='/efs-storage/jieha/share/hackson/rephrase/training_06-07-testing_12_100k',
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
    default='/home/ec2-user/workspaces/hoverboard-workspaces/src/facebook/bart-large',
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
    default='output/output_bart/alexa_qr',
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
    default="",
    type=str,
    help="Where do you want to store the pre-trained models downloaded from s3",
)
parser.add_argument(
    "--max_seq_length",
    default=10,
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
    "--n_gpu", default=1, type=int, help="the nubmer of gpu.",
)

parser.add_argument(
    "--cuda_device", default=-1, type=int, help="cuda number, default is -1, refers to cuda:0 if use_gpu is True, use cpu if use_gpu is False.",
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
    "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
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
parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")

# RecAdam parameters
parser.add_argument("--optimizer", type=str, default="RecAdam", choices=["Adam", "RecAdam","CrlAdam"],
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
#from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs
from simpletransformers.seq2seq import Seq2SeqArgs
from model_basedon_huggingface.seq2seq_model import Seq2SeqModel
import wandb
wandb.init
model_args = Seq2SeqArgs()
model_args.do_sample = True
model_args.eval_batch_size = 16
model_args.evaluate_during_training = True
model_args.evaluate_during_training_steps = 2500
model_args.evaluate_during_training_verbose = True
model_args.fp16 = False
model_args.learning_rate = 5e-5
model_args.max_length = 10
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
model_args.train_batch_size = 8
model_args.use_multiprocessing = False
model_args.optimizer = args.optimizer
model_args.output_dir = args.output_dir
model_args.best_model_dir = args.output_dir +'/best_model'
model_args.cuda_device = args.cuda_device
model_args.n_gpu = args.n_gpu
model_args.cache_dir = args.cache_dir
model_args.use_multiprocessing = False
#model_args.wandb_project = "Paraphrasing with BART"
model_args.start_from_pretrain = args.start_from_pretrain
model = Seq2SeqModel(
    encoder_decoder_type="bart",
    encoder_decoder_name="facebook/bart-large",
    args=model_args,
    use_cuda=False,
    cuda_device=args.cuda_device ##if use_cuda=False, cuda_device=-1, yet, use_cuda=True, cuda_device=-1, means we use cuda:0
)

'''
model = Seq2SeqModel(
     encoder_decoder_type="bart",
     encoder_decoder_name="/home/ec2-user/workspaces/hoverboard-workspaces/src/facebook/bart-large",
     args=model_args,
     use_cuda=True,
     cuda_device=args.cuda_device ##if use_cuda=False, cuda_device=-1, yet, use_cuda=True, cuda_device=-1, means we use cuda:0
)
'''
# model = Seq2SeqModel(
#     encoder_decoder_type="bart",
#     encoder_decoder_name="/efs-storage/intDFS_allenv12/DeepQueryRewritingTools/query_rewriting/push_button_recipe/outputs",
#     args=model_args,
#     use_cuda=True
# )

#from utils import load_data, clean_unnecessary_spaces

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.ERROR)

q_test = pd.read_csv(args.data_dir + "/test.tsv", sep="\t")

eval_df = q_test
eval_df = q_test.loc[q_test["label"] == 1]

eval_df = eval_df.rename(
    columns={"sentence1": "input_text", "sentence2": "target_text"}
)

eval_df = eval_df[["input_text", "target_text"]]

eval_df["prefix"] = "paraphrase"

eval_df = eval_df[["prefix", "input_text", "target_text"]]

eval_df["input_text"] = eval_df["input_text"].apply(clean_unnecessary_spaces)
eval_df["target_text"] = eval_df["target_text"].apply(clean_unnecessary_spaces)
print(eval_df)
print(eval_df.size)

# from transformers import AutoTokenizer, AutoModel
# tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
# model = AutoModel.from_pretrained("facebook/bart-large")
#
#
#
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    BartConfig,
    BartForConditionalGeneration,
    BartTokenizer,
    MBartConfig,
    MBartForConditionalGeneration,
    MBartTokenizer,
    BertConfig,
    BertModel,
    BertTokenizer,
    CamembertConfig,
    CamembertModel,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertModel,
    DistilBertTokenizer,
    ElectraConfig,
    ElectraModel,
    ElectraTokenizer,
    LongformerConfig,
    LongformerModel,
    LongformerTokenizer,
    MarianConfig,
    MarianMTModel,
    MarianTokenizer,
    MobileBertConfig,
    MobileBertModel,
    MobileBertTokenizer,
    RobertaConfig,
    RobertaModel,
    RobertaTokenizer,
)
MODEL_CLASSES = {
    "auto": (AutoConfig, AutoModel, AutoTokenizer),
    "bart": (BartConfig, BartForConditionalGeneration, BartTokenizer),
    "mbart": (MBartConfig, MBartForConditionalGeneration, MBartTokenizer),
    "bert": (BertConfig, BertModel, BertTokenizer),
    "camembert": (CamembertConfig, CamembertModel, CamembertTokenizer),
    "distilbert": (DistilBertConfig, DistilBertModel, DistilBertTokenizer),
    "electra": (ElectraConfig, ElectraModel, ElectraTokenizer),
    "longformer": (LongformerConfig, LongformerModel, LongformerTokenizer),
    "mobilebert": (MobileBertConfig, MobileBertModel, MobileBertTokenizer),
    "marian": (MarianConfig, MarianMTModel, MarianTokenizer),
    "roberta": (RobertaConfig, RobertaModel, RobertaTokenizer),
}

if args.model_type:
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
else:
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

trained_model_path = 'output/output_bart/alexa_qr_data_training_10-11-testing_12_100k/best_model/'
model.new_model = model_class.from_pretrained(trained_model_path)
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
    ...:         if i%3==1: 
    ...:             line = line[10:] 
    ...:             if ':' in line: 
    ...:                 line=line[1:].strip() 
    ...:             else: 
    ...:                 line = line.strip() 

'''
output_dir='output/output_bart/alexa_qr_data_training_10-11-testing_12_100k_10epoch'
output_dir='output/output_bart/alexa_qr_data_training_10-11-testing_12_100k_15epoch'
output_dir='output/output_bart/alexa_qr_data_training_10-11-testing_12_100k_20epoch'
with open(f"{output_dir}/predictions/test_{datetime.now()}.hypo", "w") as f:
    for i, text in enumerate(eval_df["input_text"].tolist()):
        for j, pred in enumerate(preds[i]):
            if j == 0:
                pred = str(pred)[10:]
                if ':' in pred:
                    pred=pred[1:].strip()
                else:
                    pred=pred.strip()
                f.write(str(pred) + "\n")
