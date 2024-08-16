#https://towardsdatascience.com/paraphrase-any-question-with-t5-text-to-text-transfer-transformer-pretrained-model-and-cbb9e35f1555
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
import time
import json
import re
from datetime import datetime, timedelta, date
from pyspark.sql import Window
import nltk
from pyspark import SparkConf, SparkContext
from pyspark.sql.session import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import udf, struct
from pyspark.sql.types import StructType, StringType, StructField, ArrayType, IntegerType, FloatType

def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_paraphraser')
# tokenizer = T5Tokenizer.from_pretrained('t5-base')
# save_dir = '/Users/lidingch/Documents/DFS_works/ramsrigouthamg/'
# tokenizer.save_pretrained(save_dir)
# model.save_pretrained(save_dir)

model = T5ForConditionalGeneration.from_pretrained('/home/ec2-user/workspaces/hoverboard-workspaces/src/models/ramsrigouthamg')
tokenizer = T5Tokenizer.from_pretrained('/home/ec2-user/workspaces/hoverboard-workspaces/src/models/ramsrigouthamg')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print ("device ",device)
model = model.to(device)

source_target_dict = {}
eval_filename = '/Users/lidingch/Documents/workspaces/WeaklySupervisedTextGeneration/PG_RL_Retrieve/data/quora/test.csv'
eval_filename = '/efs-storage/WeaklySupervisedTextGeneration/PG_RL_Retrieve/data/quora/test.csv'
with open(eval_filename, 'r') as fin:
    eva_data = fin.readlines()
    for cnt, line in enumerate(eva_data):
        if cnt == 0:
            continue
        line = line.strip().split('\t')
        source_target_dict[line[0]] = line[1]

results_schema = StructType([
    StructField('source', StringType(), False),
    StructField('target', StringType(), False),
    StructField('paraphrase', ArrayType(StringType()), False),
])


## https://stackoverflow.com/questions/59838563/append-to-pyspark-array-column


#sentence = "Which course should I take to get started in data science?"
# sentence = "What are the ingredients required to bake a perfect cake?"
# sentence = "What is the best possible approach to learn aeronautical engineering?"
# sentence = "Do apples taste better than oranges in general?"
paraphrase_json_list = []
for source, target in source_target_dict.items():
    json_line = {}
    text =  "paraphrase: " + source + " </s>"
    max_len = 256

    encoding = tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)


    # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
    beam_outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        do_sample=True,
        max_length=256,
        top_k=120,
        top_p=0.98,
        early_stopping=True,
        num_return_sequences=10
    )

    print ("\nOriginal Question ::")
    print (source)
    json_line['source'] = source
    json_line['target'] = target
    print ("\n")
    print ("Paraphrased Questions :: ")
    final_outputs =[]
    for beam_output in beam_outputs:
        sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
        if sent.lower() != source.lower() and sent not in final_outputs:
            final_outputs.append(sent)
    json_line['paraphrase'] = final_outputs
    # for i, final_output in enumerate(final_outputs):
    #     print("{}: {}".format(i, final_outputs))
    paraphrase_json_list.append(json_line)
df = spark.createDataFrame(paraphrase_json_list, results_schema)
df.coalesce(1).write.json('/efs-storage/WeaklySupervisedTextGeneration/PG_RL_Retrieve/data/quora/paraphrase_output', mode='overwrite')
