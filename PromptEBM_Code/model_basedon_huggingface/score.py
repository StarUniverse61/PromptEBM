# -*- coding: utf-8 -*
from __future__ import print_function
import sys, subprocess
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
import nltk
import string
from rouge import Rouge
import sacrebleu
import transformers as tfs
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import datasets


smoothing_function = SmoothingFunction().method4

class Metrics(object):
    def __init__(self):
        pass

    @staticmethod
    def bleu_score(references, candidates):
        """
        计算bleu值
        :param references: 实际值, list of string
        :param candidates: 验证值, list of string
        :return:
        """
        # 遍历计算bleu
        bleu1s = []
        bleu2s = []
        bleu3s = []
        bleu4s = []
        for ref, cand in zip(references, candidates):
            #ref_list = list(ref)
            #cand_list = list(cand)
            ref_list = [ref]
            cand_list = cand
            if len(cand)<2:
               continue
           
            bleu1 = sentence_bleu(ref_list, cand_list, weights=(1, 0, 0, 0), smoothing_function=smoothing_function)
            bleu2 = sentence_bleu(ref_list, cand_list, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing_function)
            bleu3 = sentence_bleu(ref_list, cand_list, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing_function)
            bleu4 = sentence_bleu(ref_list, cand_list, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_function)
            # print ("ref: %s, cand: %s, bleus: %.3f, %.3f, %.3f, %.3f"
            #        % (ref, cand, bleu1, bleu2, bleu3, bleu4))
            bleu1s.append(bleu1)
            bleu2s.append(bleu2)
            bleu3s.append(bleu3)
            bleu4s.append(bleu4)

        # 计算平均值
        bleu1_average = sum(bleu1s)*100.0 / len(bleu1s)
        bleu2_average = sum(bleu2s)*100.0 / len(bleu2s)
        bleu3_average = sum(bleu3s)*100.0 / len(bleu3s)
        bleu4_average = sum(bleu4s)*100.0 / len(bleu4s)
        print(len(candidates), len(bleu1s), len(bleu2s), len(bleu3s),  len(bleu4s))
        print('bleu1', min(bleu1s), max(bleu1s))
        print('bleu2', min(bleu2s), max(bleu2s))
        print('bleu3', min(bleu3s), max(bleu3s))
        print('bleu4', min(bleu4s), max(bleu4s))
        # 输出
        print("average bleus: bleu1: %.3f, bleu2: %.3f, bleu3: %.3f, bleu4: %.3f" % (bleu1_average, bleu2_average, bleu3_average, bleu4_average))
        return (bleu1_average, bleu2_average, bleu3_average, bleu4_average)

    @staticmethod
    def bleu_score_perl(references_file, candidates_file):    
        script_line = "perl multi-bleu.perl -lc " + references_file + " < " + candidates_file
        perl_script = subprocess.Popen([script_line], shell=True, stdout=subprocess.PIPE)

        bleu_perl_result = perl_script.communicate()[0][:-1].decode('utf8').split('/')
        #print(bleu_perl_result)

        bleu_perl_result = [float(i) for i in bleu_perl_result]

        w1 = [1, 0, 0, 0]
        w2 = [0.5, 0.5, 0, 0]
        w3 = [0.33, 0.33, 0.33, 0]
        w4 = [0.25, 0.25, 0.25, 0.25]
        s1 = weighted_sum(bleu_perl_result,w1)
        s2 = weighted_sum(bleu_perl_result,w2)
        s3 = weighted_sum(bleu_perl_result,w3)
        s4 = weighted_sum(bleu_perl_result,w4)

        print("average bleus by perl: bleu1: %.3f, bleu2: %.3f, bleu3: %.3f, bleu4: %.3f" % (s1, s2, s3, s4))
        return (s1, s2, s3, s4)


    @staticmethod
    def sacrebleu_score(references_file, candidates_file):    
        script_line = "sacrebleu --input " + candidates_file + " " + references_file + " --quiet --short --force"
        python_script = subprocess.Popen([script_line], shell=True, stdout=subprocess.PIPE)

        python_perl_result = python_script.communicate()[0][:-1].decode('utf8').split(' ')

        print(python_perl_result)

        bleu_perl_result = [float(i) for i in python_perl_result[3].split('/')]

        w1 = [1, 0, 0, 0]
        w2 = [0.5, 0.5, 0, 0]
        w3 = [0.33, 0.33, 0.33, 0]
        w4 = [0.25, 0.25, 0.25, 0.25]
        s1 = weighted_sum(bleu_perl_result,w1)
        s2 = weighted_sum(bleu_perl_result,w2)
        s3 = weighted_sum(bleu_perl_result,w3)
        s4 = weighted_sum(bleu_perl_result,w4)

        print("average bleus by sacrebleu: bleu1: %.3f, bleu2: %.3f, bleu3: %.3f, bleu4: %.3f" % (s1, s2, s3, s4))
        return (s1, s2, s3, s4)
    
    @staticmethod
    def sacrebleu_score2(references, candidates):
        metric = datasets.load_metric("sacrebleu")
        #metric = datasets.load_metric('/home/ec2-user/workspaces/hoverboard-workspaces/src/whl_collections/datasets/metrics/sacrebleu')
        #metric = datasets.load_metric('./dataset/metrics/sacrebleu')
        ref_list = []
        cand_list = []
        for ref, cand in zip(references, candidates):
            ref_list.append([' '.join(list(ref))])
            cand_list.append(' '.join(list(cand)))
        final_score = metric.compute(predictions=cand_list, references=ref_list)

        sacrebleu_score = final_score['score']
        bleu_precisions = final_score['precisions']
        w1 = [1, 0, 0, 0]
        w2 = [0.5, 0.5, 0, 0]
        w3 = [0.33, 0.33, 0.33, 0]
        w4 = [0.25, 0.25, 0.25, 0.25]
        s1 = weighted_sum(bleu_precisions, w1)
        s2 = weighted_sum(bleu_precisions, w2)
        s3 = weighted_sum(bleu_precisions, w3)
        s4 = weighted_sum(bleu_precisions, w4)

        print("SACREBLEU score: {}, BLEU1: {}, BLEU2:{}, BLEU3: {}, BLEU4: {}".format(sacrebleu_score, s1, s2, s3, s4))

    @staticmethod
    def rouge_score2(references, candidates):
        #metric = datasets.load_metric("/home/ec2-user/workspaces/hoverboard-workspaces/src/whl_collections/datasets/metrics/rouge")
        #metric = datasets.load_metric('./dataset/metrics/rouge')
        metric = datasets.load_metric('rouge')
        ref_list = []
        cand_list = []
        for ref, cand in zip(references, candidates):
            #ref, cand  = [ref], [cand]
            ref_list.append(' '.join(list(ref)))
            cand_list.append(' '.join(list(cand)))
        final_score = metric.compute(predictions=cand_list, references=ref_list)
        print(final_score)

    @staticmethod
    def meteor_score2(references, candidates):
        metric = datasets.load_metric("meteor")
        #metric = datasets.load_metric("/home/ec2-user/workspaces/hoverboard-workspaces/src/whl_collections/datasets/metrics/meteor")
        #metric = datasets.load_metric('./dataset/metrics/meteor')
        ref_list = []
        cand_list = []
        for ref, cand in zip(references, candidates):
            #ref, cand = [ref], [cand]
            #print(list(ref),list(cand))
            ref_list.append(' '.join(list(ref)))
            cand_list.append(' '.join(list(cand)))

        final_score = metric.compute(predictions=cand_list, references=ref_list)
        print(final_score)

    @staticmethod
    def em_score(references, candidates):
        total_cnt = len(references)
        match_cnt = 0
        for ref, cand in zip(references, candidates):
            if ref == cand:
                match_cnt = match_cnt + 1

        em_score = match_cnt / (float)(total_cnt)
        print("em_score: %.3f, match_cnt: %d, total_cnt: %d" % (em_score, match_cnt, total_cnt))
        return em_score

    @staticmethod
    def meteor_score(references, candidates):
        """
        meteor_score计算
        :param references: 实际值, list of string
        :param candidates: 验证值, list of string
        :return:
        """
        # 遍历计算meteor
        meteors = []
        for ref, cand in zip(references, candidates):
            #ref = ' '.join(list(ref))
            #cand = ' '.join(list(cand))
            meteor = single_meteor_score(ref, cand)
            
            meteors.append(meteor)
            

        # 计算平均值
        meteor_average = sum(meteors) / len(meteors) * 100
        
        # 输出
        print("average meteors: %.2f" % (meteor_average))
        return (meteor_average)

    @staticmethod
    def rouge_score(references, candidates):
        """
        rouge计算，NLG任务语句生成，词语的recall
        https://github.com/pltrdy/rouge
        :param references: list string
        :param candidates: list string
        :return:
        """
        eval = Rouge()

        # 遍历计算rouge
        rouge1s = []
        rouge2s = []
        rougels = []
        for ref, cand in zip(references, candidates):
            ref = ' '.join(list(ref))
            cand = ' '.join(list(cand))
            #print(ref, cand)
            rouge_score = eval.get_scores(cand,ref)
            #print('rouge', rouge_score)
            rouge_1 = rouge_score[0]["rouge-1"]['f']
            rouge_2 = rouge_score[0]["rouge-2"]['f']
            rouge_l = rouge_score[0]["rouge-l"]['f']
            # print "ref: %s, cand: %s" % (ref, cand)
            # print 'rouge_score: %s' % rouge_score

            rouge1s.append(rouge_1)
            rouge2s.append(rouge_2)
            rougels.append(rouge_l)

        # 计算平均值
        rouge1_average = sum(rouge1s) / len(rouge1s) * 100
        rouge2_average = sum(rouge2s) / len(rouge2s) * 100
        rougel_average = sum(rougels) / len(rougels) * 100

        # 输出
        print("average rouges, rouge_1: %.2f, rouge_2: %.2f, rouge_l: %.2f" \
              % (rouge1_average, rouge2_average, rougel_average))
        return (rouge1_average, rouge2_average, rougel_average)

    @staticmethod
    def paraphrase_score(references, candidates, cuda_device):
        """
        rouge计算，NLG任务语句生成，词语的recall
        https://github.com/pltrdy/rouge
        :param references: list string
        :param candidates: list string
        :return:
        """
        model_class, tokenizer_class, pretrained_weights = (
            tfs.AutoModelForSequenceClassification, tfs.AutoTokenizer, 'textattack/bert-base-uncased-QQP')
        bert_tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        bert_model = model_class.from_pretrained(pretrained_weights)

        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # if torch.cuda.device_count() > 1:
        #     print("Multi-GPU")
        #     bert_model = nn.DataParallel(bert_model)

        if cuda_device > 0:
            device = torch.device(f"cuda:{cuda_device}")
        else:
            device = "cpu"
        bert_model.to(device)

        text_pair = zip(references, candidates)
        # tokenized_text_pair = bert_tokenizer.batch_encode_plus(text_pair, add_special_tokens=True,
        #                                                        max_length=66,
        #                                                        is_pretokenized=False, pad_to_max_length=True,
        #                                                        return_tensors='pt')
        tokenized_text_pair = bert_tokenizer.batch_encode_plus(list(text_pair), add_special_tokens=True,
                                                               max_length=66,
                                                               pad_to_max_length=True,
                                                               return_tensors='pt')
        text_pair_ids = tokenized_text_pair['input_ids']
        mask_bert = tokenized_text_pair['attention_mask']
        token_type_ids = tokenized_text_pair['token_type_ids']
        print("Finish Bert Tokenization!")

        # compute final data value
        data_size = len(references)
        idx_list = range(data_size)
        idx_batches = [idx_list[i:i + 1000] for i in range(0, len(idx_list), 1000)]
        data_values_all = []
        with torch.no_grad():
            for batch_idx in idx_batches:
                text_pair_ids_batch = text_pair_ids[batch_idx].to(device)
                mask_bert_batch = mask_bert[batch_idx].to(device)
                token_type_ids_batch = token_type_ids[batch_idx].to(device)
                print(f'device={device}')
                print(f'bert device={bert_model.device}')
                print(f'bert device={text_pair_ids_batch.device}')
                print(f'bert device={mask_bert_batch.device}')
                print(f'bert device={token_type_ids_batch.device}')
                outputs_batch = bert_model(text_pair_ids_batch, attention_mask=mask_bert_batch,
                                           token_type_ids=token_type_ids_batch)
                data_values = F.softmax(outputs_batch[0], dim=-1)
                data_values = data_values[:, 1].detach().cpu().numpy().tolist()
                data_values_all.extend(data_values)

        data_values_all = np.array(data_values_all)
        score = np.mean(data_values_all) * 100
        print("paraphrase score %.2f" % score)
        return score


def preprocess(s):
    
    #words = nltk.word_tokenize(s)

    #words=[word.lower() for word in words if word.isalpha()]

    #return s.translate(None, string.punctuation)

    table = str.maketrans(dict.fromkeys(string.punctuation))  # OR {key: None for key in string.punctuation}
    new_s = s.translate(table)                          # Output: string without punctuation

    new_s = new_s.split(' ')

    s = []
    for i in new_s:
        if len(i)>0:
            s.append(i)

    return s

def weighted_sum(s, w):
    re = 0
    for i in range(len(s)):
        re += s[i]*w[i]
    return re


def run_score(ref_file, hypo_file, cuda_device=0):
    import nltk
    nltk.download('omw-1.4')
    nltk.download('wordnet')
    # ref_file = sys.argv[1]
    # hypo_file = sys.argv[2]
    
    references_orig = []
    f = open(ref_file, 'r')
    for line in f:        
        #references.append(line.rstrip().split(' '))
        #references_original.append(line.rstrip())
        references_orig.append(preprocess(line.strip()))
        

    candidates_orig = []
    f = open(hypo_file, 'r')
    for line in f:        
        #candidates.append(line.rstrip().split(' '))
        #candidates_original.append(line.rstrip())
        candidates_orig.append(preprocess(line.strip()))
    if len(candidates_orig) > len(references_orig):
        candidates_orig = candidates_orig[1:]
    references = []
    candidates = []
    references4bert = []
    candidates4bert = []
    print(len(references_orig), len(candidates_orig))
    for i in range(len(references_orig)):
        if len(references_orig[i])>0 and len(candidates_orig[i])>0:
            references.append(references_orig[i])
            candidates.append(candidates_orig[i])
            references4bert.append(' '.join(references_orig[i]))
            candidates4bert.append(' '.join(candidates_orig[i]))

    print('Done preprocessing')


    # 计算metrics
    Metrics.bleu_score(references, candidates)
    #Metrics.bleu_score_perl(ref_file, hypo_file)
    #Metrics.sacrebleu_score(ref_file, hypo_file)
    #Metrics.sacrebleu_score2(references, candidates)
    Metrics.em_score(references, candidates)
    Metrics.meteor_score(references, candidates)
    Metrics.rouge_score(references, candidates)
    Metrics.rouge_score2(references, candidates)
    Metrics.meteor_score2(references, candidates)
    print(f'cuda_device in run_score of score.py ={cuda_device}')
    Metrics.paraphrase_score(references4bert, candidates4bert, cuda_device)

if __name__ == '__main__':
    ref_file = sys.argv[1]
    hypo_file = sys.argv[2]
    run_score(ref_file, hypo_file)

