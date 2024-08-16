
import os
import sys
from datetime import datetime
import logging

import pandas as pd
from sklearn.model_selection import train_test_split
#from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs

from utils import load_data, clean_unnecessary_spaces

##let's load the following data format
##/data/dingcheng/workspace/baidu/ccl/Lifelonglearning_For_NLG/data/quora/data4fairseq_v2/test.source
##/data/dingcheng/workspace/baidu/ccl/Lifelonglearning_For_NLG/data/quora/data4fairseq_v2/test.target
def load_test_df(source_path, target_path):
    with open(source_path, 'r') as s, open(target_path, 'r') as t:
        source = [line.rstrip() for line in s]
        target = [line.rstrip() for line in t]
        eval_df = pd.DataFrame(list(zip(source, target)), columns=['input_text', 'target_text'])
        eval_df = eval_df[["input_text", "target_text"]]
        eval_df["prefix"] = "paraphrase"
        eval_df = eval_df[["prefix", "input_text", "target_text"]]
        eval_df = eval_df.dropna()
        eval_df["input_text"] = eval_df["input_text"].apply(clean_unnecessary_spaces)
        eval_df["target_text"] = eval_df["target_text"].apply(clean_unnecessary_spaces)
        return eval_df

def load_df_generic(source_path, target_path):
    with open(source_path, 'r') as s, open(target_path, 'r') as t:
        source = [line.rstrip() for line in s]
        target = [line.rstrip() for line in t]
        df = pd.DataFrame(list(zip(source, target)), columns=['input_text', 'target_text'])
        df = df[["input_text", "target_text"]]
        df["prefix"] = "paraphrase"
        df = df[["prefix", "input_text", "target_text"]]
        df = df.dropna()
        df["input_text"] = df["input_text"].apply(clean_unnecessary_spaces)
        df["target_text"] = df["target_text"].apply(clean_unnecessary_spaces)
        return df

def load_df():
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.ERROR)
    data_path = '/home/xiaodi/NLP/Lifelonglearning-main/data/quora/augmented_data/data/train.tsv' ##'/data/dingcheng/workspace/baidu/ccl/Lifelonglearning_For_NLG/data/quora/augmented_data/data/train.tsv'
    if os.path.exists(data_path):
        # Google Data
        #train_df = pd.read_csv("/data/dingcheng/workspace/baidu/ccl/Lifelonglearning_For_NLG/data/quora/augmented_data/data/train.tsv", sep="\t").astype(str)
        #eval_df = pd.read_csv("/data/dingcheng/workspace/baidu/ccl/Lifelonglearning_For_NLG/data/quora/augmented_data/data/dev.tsv", sep="\t").astype(str)
        train_df = pd.read_csv(
            "/home/xiaodi/NLP/Lifelonglearning-main/data/quora/augmented_data/data/train.tsv",
            sep="\t").astype(str)
        eval_df = pd.read_csv(
            "/home/xiaodi/NLP/Lifelonglearning-main/data/quora/augmented_data/data/dev.tsv",
            sep="\t").astype(str)
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
        train_df = pd.concat(
            [
                train_df,
                load_data("/home/xiaodi/NLP/Lifelonglearning-main/data/quora/augmented_data/data/msr_paraphrase_train.txt", "#1 String", "#2 String", "Quality"),
            ]
        )
        eval_df = pd.concat(
            [
                eval_df,
                load_data("/home/xiaodi/NLP/Lifelonglearning-main/data/quora/augmented_data/data/msr_paraphrase_test.txt", "#1 String", "#2 String", "Quality"),
            ]
        )

        # quora Data

        # The quora Dataset is not separated into train/test, so we do it manually the first time.
        df = load_data(
            "/home/xiaodi/NLP/Lifelonglearning-main/data/quora/augmented_data/data/quora_duplicate_questions.tsv", "question1", "question2", "is_duplicate"
        )
        q_train, q_test = train_test_split(df)

        q_train.to_csv("/home/xiaodi/NLP/Lifelonglearning-main/data/quora/augmented_data/data/quora_train.tsv", sep="\t")
        q_test.to_csv("/home/xiaodi/NLP/Lifelonglearning-main/data/quora/augmented_data/data/quora_test.tsv", sep="\t")

        # The code block above only needs to be run once.
        # After that, the two lines below are sufficient to load the quora dataset.
    # else:
    #     q_train = pd.read_csv("/data/dingcheng/workspace/baidu/ccl/Lifelonglearning_For_NLG/data/quora/augmented_data/data/quora_train.tsv", sep="\t")
    #     q_test = pd.read_csv("/data/dingcheng/workspace/baidu/ccl/Lifelonglearning_For_NLG/data/quora/augmented_data/data/quora_test.tsv", sep="\t")

    train_df = pd.concat([train_df, q_train])
    # eval_df = pd.concat([eval_df, q_test])
    train_df = train_df[["prefix", "input_text", "target_text"]]
    eval_df = eval_df[["prefix", "input_text", "target_text"]]
    train_df = train_df.dropna()
    eval_df = eval_df.dropna()
    q_test = q_test.dropna()
    train_df["input_text"] = train_df["input_text"].apply(clean_unnecessary_spaces)
    train_df["target_text"] = train_df["target_text"].apply(clean_unnecessary_spaces)
    eval_df["input_text"] = eval_df["input_text"].apply(clean_unnecessary_spaces)
    eval_df["target_text"] = eval_df["target_text"].apply(clean_unnecessary_spaces)
    q_test["input_text"] = q_test["input_text"].apply(clean_unnecessary_spaces)

    convert4fairseq = False
    if convert4fairseq:
        #aug_data_path = '/data/dingcheng/workspace/baidu/ccl/Lifelonglearning_For_NLG/data/quora/data4fairseq_v2'
        aug_data_path = '/home/xiaodi/NLP/Lifelonglearning-main/data/quora/data4fairseq_v2'
        train_source = aug_data_path + '/train.source'
        valid_source = aug_data_path + '/valid.source'
        test_source = aug_data_path + '/test.source'
        train_target = aug_data_path + '/train.target'
        valid_target = aug_data_path + '/valid.target'
        test_target = aug_data_path + '/test.target'
        with open(train_source, 'w') as ts,open(train_target, 'w') as tt, open(valid_source, 'w') as vs, \
            open(valid_target, 'w') as vt,open(test_source, 'w') as tes,open(test_target, 'w') as tet:

            #train_df.values[1][1]
            train_values = train_df.values  ##136422 x 3 [1==source, 2==target]
            eval_values = eval_df.values  ##8000 x 3 [1==source, 2==target]
            test_values = q_test.values ##37316 x 3 [0==source, 1==target]
            train_source_arr = []
            train_target_arr = []
            for line in train_values:
                train_source_arr.append(clean_unnecessary_spaces(line[1]))
                train_target_arr.append(clean_unnecessary_spaces(line[2]))
                ts.write(line[1].strip()+ '\n')
                tt.write(line[2].strip() + '\n')
            #train_source_df = pd.DataFrame(train_source_arr)
            #train_target_df = pd.DataFrame(train_target_arr)
            #train_source_df.to_csv(train_source)
            #train_df.to_csv(train_source)
            #train_target_df.to_csv(train_target)
            # for source, target in zip(*(train_source_arr, train_target_arr)):
            #     ts.write(source.strip()+'\n')
            #     tt.write(target.strip() + '\n')
            for line in eval_values:
                vs.write(line[1].strip()+'\n')
                vt.write(line[2].strip()+ '\n')
            for line in test_values:
                tes.write(line[0].strip()+'\n')
                tet.write(line[1].strip() + '\n')
    print(train_df)
    return train_df, eval_df