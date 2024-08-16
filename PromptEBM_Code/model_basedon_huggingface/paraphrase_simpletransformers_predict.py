'''
https://towardsdatascience.com/bart-for-paraphrasing-with-simple-transformers-7c9ea3dfdd8c
'''
import os
from datetime import datetime
import logging

import pandas as pd
from sklearn.model_selection import train_test_split
#wget http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv -P data
import warnings
def load_data(
    file_path, input_text_column, target_text_column, label_column, keep_label=1
):
    df = pd.read_csv(file_path, sep="\t", error_bad_lines=False)
    df = df.loc[df[label_column] == keep_label]
    df = df.rename(
        columns={input_text_column: "input_text", target_text_column: "target_text"}
    )
    #df = df[["input_text", "target_text"]]
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
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs
#import wandb
#wandb.init
model_args = Seq2SeqArgs()
model_args.do_sample = True
model_args.eval_batch_size = 64
model_args.evaluate_during_training = True
model_args.evaluate_during_training_steps = 2500
model_args.evaluate_during_training_verbose = True
model_args.fp16 = False
model_args.learning_rate = 5e-5
model_args.max_length = 10
model_args.max_seq_length = 10
model_args.num_beams = None
model_args.num_return_sequences = 3
model_args.num_train_epochs = 2
model_args.overwrite_output_dir = True
model_args.reprocess_input_data = True
model_args.save_eval_checkpoints = False
model_args.save_steps = -1
model_args.top_k = 50
model_args.top_p = 0.95
model_args.train_batch_size = 64
model_args.use_multiprocessing = False
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.ERROR)
from transformers.optimization import AdamW, Adafactor
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
    BertForMaskedLM,
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
    EncoderDecoderConfig,
    EncoderDecoderModel,
    LongformerConfig,
    LongformerModel,
    LongformerTokenizer,
    MarianConfig,
    MarianMTModel,
    MarianTokenizer,
    MobileBertConfig,
    MobileBertModel,
    MobileBertTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
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

config_class, model_class, tokenizer_class = MODEL_CLASSES['bart']
model = Seq2SeqModel(
    encoder_decoder_type="bart",
    #encoder_decoder_name="/home/ec2-user/workspaces/hoverboard-workspaces/src/facebook/bart-large",
    encoder_decoder_name='output/output_bart/quora_adamw_v2/checkpoint-5256-epoch-9/',
    args=model_args,
    use_cuda=True
)
#trained_model_path = '/efs-storage/models/bart_quora_paraphrase_models/checkpoint-250830-epoch-30/'
#trained_model_path = 'output/output_bart/quora_adamw/checkpoint-27750-epoch-30/'
#model.new_model = model_class.from_pretrained(trained_model_path)
#model._load_model_args(trained_model_path)


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.ERROR)
# Quora Data
# The Quora Dataset is not separated into train/test, so we do it manually the first time.
eval_df = pd.read_csv("/efs-storage/data/quora_test.tsv", sep="\t")
eval_df = eval_df[["prefix", "input_text", "target_text"]]
eval_df = eval_df.dropna()
eval_df["input_text"] = eval_df["input_text"].apply(clean_unnecessary_spaces)
eval_df["target_text"] = eval_df["target_text"].apply(clean_unnecessary_spaces)

to_predict = [
    prefix + ": " + str(input_text)
    for prefix, input_text in zip(eval_df["prefix"].tolist(), eval_df["input_text"].tolist())
]
query = eval_df["input_text"].tolist()
truth = eval_df["target_text"].tolist()
preds = model.predict(to_predict)

# Saving the predictions if needed
#os.makedirs("/efs-storage/data/quora_predictions", exist_ok=True)
if not os.path.exists(f"{model_args.output_dir}/predictions/"):
    os.makedirs(f"{model_args.output_dir}/predictions/")
with open(f"{model_args.output_dir}/predictions/predictions_{datetime.now()}.txt", "w") as f:
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
with open(f"{model_args.output_dir}/predictions/test_{datetime.now()}.hypo", "w") as f:
    for i, text in enumerate(eval_df["input_text"].tolist()):
        for j, pred in enumerate(preds[i]):
            if j == 0:
                #pred = str(pred)[10:]
                if ':' in pred:
                    colon_ind = pred.find(':')
                    pred=pred[colon_ind+1:].strip()
                else:
                    pred=pred.strip()
                f.write(str(pred) + "\n")

with open(f"{model_args.output_dir}/test.target_quora", "w") as f:
    for line in truth:
        f.write(line+'\n')

with open(f"{model_args.output_dir}/test.source_quora", "w") as f:
    for line in query:
        f.write(line + '\n')