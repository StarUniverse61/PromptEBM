from __future__ import unicode_literals  # at top of module
from collections import Counter
import numpy as np
import json
from LAMOL_prompt_mistral.ebm_models.dgmvae.utils import get_tokenize, get_chat_tokenize, missingdict, Pack
from LAMOL_prompt_mistral.settings import FILL_VAL
from LAMOL_prompt_mistral.settings import LEN_FACTOR, DATA_ATTRS, MEMORY_FACTOR, MODEL_CONFIG, MODEL_CLASS
import logging
import os
from multiprocessing import Pool
import itertools
from collections import defaultdict
import copy
import random
from transformers.optimization import AdamW, Adafactor
# from modeling_bert import BertConfig, BertForConditionalGeneration, BertEncoder, BertDecoder
import sys, io
#sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="UTF-8")
logger = logging.getLogger(__name__)

from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    #BartConfig,
    #BartForConditionalGeneration,
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
    RobertaConfig,
    RobertaModel,
    RobertaTokenizer,
)
from ibebm.configuration_bert import BertConfig
PAD = '__pad__'
UNK = '__unk__'
BOS = '<s>'
EOS = '</s>'
BOD = "<d>"
EOD = "</d>"
BOT = "<t>"
EOT = "</t>"
ME = "<me>"
OT = "<ot>"
SYS = "<sys>"
USR = "<usr>"
KB = "<kb>"
SEP = "|"
REQ = "<requestable>"
INF = "<informable>"
WILD = "%s"
GEN = '__gen__'
ANS = '__ans__'
'''
In order to deploy ebm models, let's create a special corpus for LAMOL.
'''


class QACorpus(object):
    logger = logging.getLogger()

    def __init__(self, config, gen_token, SPECIAL_TOKEN_IDS, data_paths_train, data_paths_valid, data_paths_test, TOKENIZER):
        self.config = config
        self._path = config.ebm_data_dir
        self.max_utt_len = config.max_utt_len
        self.config = config
        self.max_a_len = 0
        self.gen_token = gen_token
        if config.use_sep:
            self.sep_token = SPECIAL_TOKEN_IDS["sep_token"]
        self.ans_token = SPECIAL_TOKEN_IDS["ans_token"]
        self.eos_token = SPECIAL_TOKEN_IDS["eos_token"]
        self.pad_token = SPECIAL_TOKEN_IDS["pad_token"]
        self.tokenizer = TOKENIZER
        self._build_vocab()
        self.data_all_train = self.data_reading(data_paths_train)
        data_paths_valid_ = []
        for data_path_valid in data_paths_valid:
            if not os.path.exists(data_path_valid):
                continue
            else:
                data_paths_valid_.append(data_path_valid)

        if len(data_paths_valid_)>0:
            self.data_all_valid = self.data_reading(data_paths_valid)
        else:
            self.data_all_valid = self.data_reading(data_paths_test)
        self.data_all_test = self.data_reading(data_paths_test)


    def data_reading(self, data_paths):
        if not isinstance(data_paths, list):
            data_paths = [data_paths]

        data = []
        for data_path in data_paths:
            if not data_path:
                continue
            with open(data_path, "r") as f:
                raw_ds = json.load(f)
            raw_ds = map(lambda x: x["paragraphs"], raw_ds["data"])
            d = []
            for raw_d in raw_ds:
                d.extend(raw_d)
            data += d

        if len(data_paths) == 1 and data_paths[0] is not None and ('wiki' in data_paths[0] or 'woz' in data_paths[0]):
            # data = self._sort_by_index(data)
            # args.n_workers = 1
            if 'wiki' in data_paths[0]:
                answers_file = "wikisql_answers.json"
            elif 'woz' in data_paths[0]:
                answers_file = "woz.en_answers.json"
            with open(os.path.join(self.config.data_dir, answers_file), "r") as f:
                self.answers = json.load(f)
        if len(data) > 0:
            data_all, self.max_a_len = self.data_tokenization(data)
        print("Done loading corpus")
        return data_all

    def data_tokenization(self, data):
        if self.config.debug:
            data = data[:10]
            new_data = []
            for datum in data:
                new_data.append(self.parallel_tokenization(datum))
            data = new_data
        else:
            with Pool(self.config.n_workers) as pool:
                data = pool.map(self.parallel_tokenization, data)
        data_all = []
        for datum, max_a_len in data:
            data_all.extend(datum)
            max_a_len = max(self.max_a_len, max_a_len)
        return data_all, max_a_len

    def _process_data(self, data):
        all_text = []
        all_lens = []
        for line in data:
            tokens = [BOS] + line.strip().split() + [EOS]
            all_lens.append(len(tokens))
            all_text.append(Pack(utt=tokens, speaker=0))
        print("Max utt len %d, mean utt len %.2f" % (
            np.max(all_lens), float(np.mean(all_lens))))
        return all_text

    def parallel_tokenization(self, d):
        '''
        The following is the file reader for ebm_modesl, for LAMOL, we have more complex reading as defined in this
        function, DC notes.
        def _read_file(self, path):
            with open(path, 'r') as f:
                lines = f.readlines()

            return self._process_data(lines)

        def _process_data(self, data):
            all_text = []
            all_lens = []
            for line in data:
                tokens = [BOS] + line.strip().split() + [EOS]
                all_lens.append(len(tokens))
                all_text.append(Pack(utt=tokens, speaker=0))
            print("Max utt len %d, mean utt len %.2f" % (
                np.max(all_lens), float(np.mean(all_lens))))
            return all_text

        '''
        examples = []
        context = self.tokenizer.encode(d["context"])
        max_a_len = 0
        for qa in d["qas"]:
            question = self.tokenizer.encode(qa["question"])

            raw_answers = qa["answers"]
            if len(raw_answers) == 0:
                assert qa["is_impossible"]
                raw_answers.append({"text": ""})

            answer = []
            for i, raw_answer in enumerate(raw_answers):
                answer.extend(self.tokenizer.encode(raw_answer["text"]))
                if i != len(raw_answers) - 1:
                    answer.append(self.pad_token)
            max_a_len = max(max_a_len, len(answer))

            examples.append(self.parse_example(self.gen_token, context, question, answer, qa.get("id", 0)))
        return examples, max_a_len

    def concat_example(self, gen_token, c, sep_token, q, ans_token, a, eos_token):
        example = sep_token + q + ans_token + a
        if len(example) + 1 > self.config.max_len:
            logger.warning('an example with len {} is too long!'.format(len(example) + 1))
            return
        example = gen_token + c[:self.config.max_len-len(example)-1] + example + eos_token
        return example

    def parse_example(self, gen_token, context, question, answer, idx):
        if self.config.use_sep:
            cq_example = self.concat_example([], context, [self.sep_token], question, [self.ans_token], [], [])
            cqa_example = self.concat_example([], context, [self.sep_token], question, [self.ans_token], answer, [])
        else:
            cq_example = self.concat_example([], context, [], question, [self.ans_token], [], [])
            cqa_example = self.concat_example([], context, [], question, [self.ans_token], answer, [])
        Y_example = self.concat_example([], [], [], [], [], answer, [self.eos_token])
        Y_example = [FILL_VAL] * (len(cqa_example) - len(Y_example)) + Y_example
        if self.config.use_sep:
            gen_X_example = self.concat_example([gen_token], context, [self.sep_token], question, [self.ans_token], answer, [])
            gen_Y_example = self.concat_example([], context, [self.sep_token], question, [self.ans_token], answer, [self.eos_token])
        else:
            gen_X_example = self.concat_example([gen_token], context, [], question, [self.ans_token], answer, [])
            gen_Y_example = self.concat_example([], context, [], question, [self.ans_token], answer, [self.eos_token])
        return cq_example, len(cq_example), cqa_example, len(cqa_example), Y_example, gen_X_example, gen_Y_example, idx


    def sort(self):
        self.data.sort(key=lambda x: len(x[0]))
        return self

    def sort_by_index(self):
        self.data.sort(key=lambda x: x[-1])

    def get_indices(self):
        return [d[-1] for d in self.data]

    def _build_vocab(self):
        self.rev_vocab = self.tokenizer.vocab
        self.unk_id = self.tokenizer.added_tokens_encoder['__unk__']
        self.pad_id = self.tokenizer.added_tokens_encoder['__pad__']
        # self.sys_id = TOKENIZER.added_tokens_encoder[SYS]
        # self.usr_id = TOKENIZER.added_tokens_encoder[USR]
        self.gen_id = self.tokenizer.added_tokens_encoder['__gen__']
        self.ans_id = self.tokenizer.added_tokens_encoder['__ans__']

        self.vocab = ['__pad__', '__unk__', SYS, USR, '__gen__', '__ans__'] + [t for t in self.tokenizer.vocab]
        self.rev_vocab['__unk__'] = self.unk_id
        self.rev_vocab['<|endoftext|>'] = self.eos_token
        self.rev_vocab['__pad__'] = self.pad_id
        # self.rev_vocab[SYS] = self.sys_id
        # self.rev_vocab[USR] = self.usr_id
        self.rev_vocab['__gen__'] = self.gen_id
        self.rev_vocab['__ans__'] = self.ans_id

    def _sent2id(self, sent):
        return [self.rev_vocab.get(t, self.unk_id) for t in sent]

    def _to_id_corpus(self, data):
        results = []
        for line in data:
            id_turn = Pack(utt=line[6],
                           label=0,
                           meta=None)
            results.append(id_turn)
        return results

    def get_corpus(self):
        id_train = self._to_id_corpus(self.data_all_train)
        id_valid = self._to_id_corpus(self.data_all_valid)
        id_test = self._to_id_corpus(self.data_all_test)
        return Pack(train=id_train, valid=id_valid, test=id_test)

    # def _process_data(self, data, label=0):
    #     all_text = []
    #     all_lens = []
    #     for line in data:
    #         tokens = [BOS] + line.strip().split() + [EOS]
    #         all_lens.append(len(tokens))
    #         all_text.append(Pack(utt=tokens, speaker=0, label=label))
    #     print("Max utt len %d, mean utt len %.2f" % (
    #         np.max(all_lens), float(np.mean(all_lens))))
    #     return all_text

class StanfordCorpus(object):
    logger = logging.getLogger(__name__)

    def __init__(self, config):
        self.config = config
        self._path = config.ebm_data_dir
        self.max_utt_len = config.max_utt_len
        self.tokenize = get_chat_tokenize()
        self.train_corpus = self._read_file(os.path.join(self._path, 'kvret_train_public.json'))
        self.valid_corpus = self._read_file(os.path.join(self._path, 'kvret_dev_public.json'))
        self.test_corpus = self._read_file(os.path.join(self._path, 'kvret_test_public.json'))
        self._build_vocab(config.max_vocab_cnt)
        # self._output_hyps(os.path.join(self._path, 'kvret_test_public.hyp'))
        print("Done loading corpus")

    def _output_hyps(self, path):
        if not os.path.exists(path):
            f = open(path, "w", encoding="utf-8")
            for utts in self.test_corpus:
                for utt in utts:
                    if utt['speaker'] != 0:
                        f.write(' '.join(utt['utt_ori']) + "\n")
            f.close()

    def _read_file(self, path):
        with open(path, 'rb') as f:
            data = json.load(f)

        return self._process_dialog(data)

    def _process_dialog(self, data):
        new_dialog = []
        bod_utt = [BOS, BOD, EOS]
        eod_utt = [BOS, EOD, EOS]
        all_lens = []
        all_dialog_lens = []
        speaker_map = {'assistant': SYS, 'driver': USR}
        for raw_dialog in data:
            intent = raw_dialog['scenario']['task']['intent']
            dialog = [Pack(utt=bod_utt,
                           speaker=0,
                           meta={'intent': intent, "text": ' '.join(bod_utt[1:-1])})]
            for turn in raw_dialog['dialogue']:

                utt = turn['data']['utterance']
                utt_ori = self.tokenize(utt)
                utt = [BOS, speaker_map[turn['turn']]] + utt_ori + [EOS]
                all_lens.append(len(utt))
                # meta={"text": line.strip()}
                dialog.append(Pack(utt=utt, speaker=turn['turn'], utt_ori=utt_ori, meta={'intent': intent,
                                                                                         'text': ' '.join(utt[1:-1])}))

            if hasattr(self.config, 'include_eod') and self.config.include_eod:
                dialog.append(Pack(utt=eod_utt, speaker=0, meta={'intent': intent,
                                                                 'text': ' '.join(eod_utt[1:-1])}))

            all_dialog_lens.append(len(dialog))
            new_dialog.append(dialog)

        print("Max utt len %d, mean utt len %.2f" % (
            np.max(all_lens), float(np.mean(all_lens))))
        print("Max dialog len %d, mean dialog len %.2f" % (
            np.max(all_dialog_lens), float(np.mean(all_dialog_lens))))
        return new_dialog

    def _build_vocab(self, max_vocab_cnt):
        all_words = []
        for dialog in self.train_corpus:
            for turn in dialog:
                all_words.extend(turn.utt)

        vocab_count = Counter(all_words).most_common()
        raw_vocab_size = len(vocab_count)
        discard_wc = np.sum([c for t, c, in vocab_count[max_vocab_cnt:]])
        vocab_count = vocab_count[0:max_vocab_cnt]

        # create vocabulary list sorted by count
        print("Load corpus with train size %d, valid size %d, "
              "test size %d raw vocab size %d vocab size %d at cut_off %d OOV rate %f"
              % (len(self.train_corpus), len(self.valid_corpus),
                 len(self.test_corpus),
                 raw_vocab_size, len(vocab_count), vocab_count[-1][1],
                 float(discard_wc) / len(all_words)))

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.vocab = [PAD, UNK, SYS, USR] + [t for t, cnt in vocab_count]
        self.rev_vocab = {t: idx for idx, t in enumerate(self.vocab)}
        self.unk_id = self.rev_vocab[UNK]
        print("<d> index %d" % self.rev_vocab[BOD])

    def _sent2id(self, sent):
        return [self.rev_vocab.get(t, self.unk_id) for t in sent]

    def _to_id_corpus(self, data):
        results = []
        for dialog in data:
            temp = []
            # convert utterance and feature into numeric numbers
            for turn in dialog:
                id_turn = Pack(utt=self._sent2id(turn.utt),
                               speaker=turn.speaker,
                               meta=turn.get('meta'))
                temp.append(id_turn)
            results.append(temp)
        return results

    def get_corpus(self):
        id_train = self._to_id_corpus(self.train_corpus)
        id_valid = self._to_id_corpus(self.valid_corpus)
        id_test = self._to_id_corpus(self.test_corpus)
        return Pack(train=id_train, valid=id_valid, test=id_test)

class PTBCorpus(object):
    logger = logging.getLogger()

    def __init__(self, config):
        self.config = config
        self._path = config.ebm_data_dir
        self.max_utt_len = config.max_utt_len
        self.tokenize = get_chat_tokenize()
        self.train_corpus = self._read_file(os.path.join(self._path, 'ptb.train.txt'))
        self.valid_corpus = self._read_file(os.path.join(self._path, 'ptb.valid.txt'))
        self.test_corpus = self._read_file(os.path.join(self._path, 'ptb.test.txt'))
        self._build_vocab(config.max_vocab_cnt)
        self.unk = "<unk>"
        print("Done loading corpus")

    def _read_file(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()

        return self._process_data(lines)

    def _process_data(self, data):
        all_text = []
        all_lens = []
        for line in data:
            tokens = [BOS] + line.strip().split() + [EOS]
            all_lens.append(len(tokens))
            all_text.append(Pack(utt=tokens, speaker=0))
        print("Max utt len %d, mean utt len %.2f" % (
            np.max(all_lens), float(np.mean(all_lens))))
        return all_text

    def _build_vocab(self, max_vocab_cnt):
        all_words = []
        for turn in self.train_corpus:
            all_words.extend(turn.utt)

        vocab_count = Counter(all_words).most_common()
        raw_vocab_size = len(vocab_count)
        discard_wc = np.sum([c for t, c, in vocab_count[max_vocab_cnt:]])
        vocab_count = vocab_count[0:max_vocab_cnt]

        # create vocabulary list sorted by count
        print("Load corpus with train size %d, valid size %d, "
              "test size %d raw vocab size %d vocab size %d at cut_off %d OOV rate %f"
              % (len(self.train_corpus), len(self.valid_corpus),
                 len(self.test_corpus),
                 raw_vocab_size, len(vocab_count), vocab_count[-1][1],
                 float(discard_wc) / len(all_words)))

        self.vocab = [PAD] + [t for t, cnt in vocab_count]
        if UNK not in self.vocab:
            self.vocab = [PAD, UNK] + [t for t, cnt in vocab_count]
        self.rev_vocab = {t: idx for idx, t in enumerate(self.vocab)}
        self.unk_id = self.rev_vocab[UNK]

    def _sent2id(self, sent):
        return [self.rev_vocab.get(t, self.unk_id) for t in sent]

    def _to_id_corpus(self, data):
        results = []
        for line in data:
            id_turn = Pack(utt=self._sent2id(line.utt),
                           speaker=line.speaker,
                           meta=line.get('meta'))
            results.append(id_turn)
        return results

    def get_corpus(self):
        id_train = self._to_id_corpus(self.train_corpus)
        id_valid = self._to_id_corpus(self.valid_corpus)
        id_test = self._to_id_corpus(self.test_corpus)
        return Pack(train=id_train, valid=id_valid, test=id_test)

class DailyDialogCorpus(object):
    logger = logging.getLogger()

    def __init__(self, config):
        self.config = config
        self._path = config.ebm_data_dir
        self.max_utt_len = config.max_utt_len
        self.tokenize = get_chat_tokenize()
        self.train_corpus = self._read_file(os.path.join(self._path, 'train'))
        self.valid_corpus = self._read_file(os.path.join(self._path, 'validation'))
        self.test_corpus = self._read_file(os.path.join(self._path, 'test'))
        self._build_vocab(config.max_vocab_cnt)
        print("Done loading corpus")

    def _read_file(self, path):
        with open(os.path.join(path, 'dialogues.txt'), 'r') as f:
            txt_lines = f.readlines()

        with open(os.path.join(path, 'dialogues_act.txt'), 'r') as f:
            da_lines = f.readlines()

        with open(os.path.join(path, 'dialogues_emotion.txt'), 'r') as f:
            emotion_lines = f.readlines()

        combined_data = [(t, d, e) for t, d, e in zip(txt_lines, da_lines, emotion_lines)]

        return self._process_dialog(combined_data)

    def _process_dialog(self, data):
        new_dialog = []
        bod_utt = [BOS, BOD, EOS]
        eod_utt = [BOS, EOD, EOS]
        all_lens = []
        all_dialog_lens = []
        for raw_dialog, raw_act, raw_emotion in data:
            dialog = [Pack(utt=bod_utt,
                           speaker=0,
                           meta=None)]

            # raw_dialog = raw_dialog.decode('ascii', 'ignore').encode()
            raw_dialog = raw_dialog.split('__eou__')[0:-1]
            raw_act = raw_act.split()
            raw_emotion = raw_emotion.split()

            for t_id, turn in enumerate(raw_dialog):
                utt = turn
                utt = [BOS] + self.tokenize(utt.lower()) + [EOS]
                all_lens.append(len(utt))
                dialog.append(Pack(utt=utt, speaker=t_id%2,
                                   meta={'emotion': raw_emotion[t_id], 'act': raw_act[t_id]}))

            if not hasattr(self.config, 'include_eod') or self.config.include_eod:
                dialog.append(Pack(utt=eod_utt, speaker=0))

            all_dialog_lens.append(len(dialog))
            new_dialog.append(dialog)

        print("Max utt len %d, mean utt len %.2f" % (
            np.max(all_lens), float(np.mean(all_lens))))
        print("Max dialog len %d, mean dialog len %.2f" % (
            np.max(all_dialog_lens), float(np.mean(all_dialog_lens))))
        return new_dialog

    def _build_vocab(self, max_vocab_cnt):
        all_words = []
        for dialog in self.train_corpus:
            for turn in dialog:
                all_words.extend(turn.utt)

        vocab_count = Counter(all_words).most_common()
        raw_vocab_size = len(vocab_count)
        discard_wc = np.sum([c for t, c, in vocab_count[max_vocab_cnt:]])
        vocab_count = vocab_count[0:max_vocab_cnt]

        # create vocabulary list sorted by count
        print("Load corpus with train size %d, valid size %d, "
              "test size %d raw vocab size %d vocab size %d at cut_off %d OOV rate %f"
              % (len(self.train_corpus), len(self.valid_corpus),
                 len(self.test_corpus),
                 raw_vocab_size, len(vocab_count), vocab_count[-1][1],
                 float(discard_wc) / len(all_words)))

        self.vocab = [PAD, UNK, SYS, USR] + [t for t, cnt in vocab_count]
        self.rev_vocab = {t: idx for idx, t in enumerate(self.vocab)}
        self.unk_id = self.rev_vocab[UNK]

    def _sent2id(self, sent):
        return [self.rev_vocab.get(t, self.unk_id) for t in sent]

    def _to_id_corpus(self, data):
        results = []
        for dialog in data:
            temp = []
            # convert utterance and feature into numeric numbers
            for turn in dialog:
                id_turn = Pack(utt=self._sent2id(turn.utt),
                               speaker=turn.speaker,
                               meta=turn.get('meta'))
                temp.append(id_turn)
            results.append(temp)
        return results

    def get_corpus(self):
        id_train = self._to_id_corpus(self.train_corpus)
        id_valid = self._to_id_corpus(self.valid_corpus)
        id_test = self._to_id_corpus(self.test_corpus)
        return Pack(train=id_train, valid=id_valid, test=id_test)


class NewsCorpus(object):
    logger = logging.getLogger()

    def __init__(self, config):
        self.config = config
        self._path = config.ebm_data_dir
        self.max_utt_len = config.max_utt_len
        self.tokenize = get_chat_tokenize()
        self.train_corpus = self._read_file(os.path.join(self._path, 'news.train.txt'))
        self.valid_corpus = self._read_file(os.path.join(self._path, 'news.valid.txt'))
        self.test_corpus = self._read_file(os.path.join(self._path, 'news.test.txt'))
        self._build_vocab(config.max_vocab_cnt)
        self.unk = "<unk>"
        print("Done loading corpus")

    def _read_file(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()

        return self._process_data(lines)

    def _process_data(self, data):
        all_text = []
        all_lens = []
        for line in data:
            tokens = [BOS] + line.strip().split() + [EOS]
            all_lens.append(len(tokens))
            all_text.append(Pack(utt=tokens, speaker=0))
        print("Max utt len %d, mean utt len %.2f" % (
            np.max(all_lens), float(np.mean(all_lens))))
        return all_text

    def _build_vocab(self, max_vocab_cnt):
        all_words = []
        for turn in self.train_corpus:
            all_words.extend(turn.utt)

        vocab_count = Counter(all_words).most_common()
        raw_vocab_size = len(vocab_count)
        discard_wc = np.sum([c for t, c, in vocab_count[max_vocab_cnt:]])
        vocab_count = vocab_count[0:max_vocab_cnt]

        # create vocabulary list sorted by count
        print("Load corpus with train size %d, valid size %d, "
              "test size %d raw vocab size %d vocab size %d at cut_off %d OOV rate %f"
              % (len(self.train_corpus), len(self.valid_corpus),
                 len(self.test_corpus),
                 raw_vocab_size, len(vocab_count), vocab_count[-1][1],
                 float(discard_wc) / len(all_words)))

        self.vocab = [PAD] + [t for t, cnt in vocab_count]
        if UNK not in self.vocab:
            self.vocab = [PAD, UNK] + [t for t, cnt in vocab_count]
        self.rev_vocab = {t: idx for idx, t in enumerate(self.vocab)}
        self.unk_id = self.rev_vocab[UNK]

    def _sent2id(self, sent):
        return [self.rev_vocab.get(t, self.unk_id) for t in sent]

    def _to_id_corpus(self, data):
        results = []
        for line in data:
            id_turn = Pack(utt=self._sent2id(line.utt),
                           speaker=line.speaker,
                           meta=line.get('meta'))
            results.append(id_turn)
        return results

    def get_corpus(self):
        id_train = self._to_id_corpus(self.train_corpus)
        id_valid = self._to_id_corpus(self.valid_corpus)
        id_test = self._to_id_corpus(self.test_corpus)
        return Pack(train=id_train, valid=id_valid, test=id_test)


class YelpCorpus(object):
    logger = logging.getLogger()

    def __init__(self, config):
        self.config = config
        self._path = config.ebm_data_dir
        self.max_utt_len = config.max_utt_len
        self.tokenize = get_chat_tokenize()
        self.train_corpus0 = self._read_file(os.path.join(self._path, 'sentiment.train.0'), label=0)
        self.train_corpus1 = self._read_file(os.path.join(self._path, 'sentiment.train.1'), label=1)
        self.valid_corpus0 = self._read_file(os.path.join(self._path, 'sentiment.dev.0'), label=0)
        self.valid_corpus1 = self._read_file(os.path.join(self._path, 'sentiment.dev.1'), label=1)
        self.test_corpus0 = self._read_file(os.path.join(self._path, 'sentiment.test.0'), label=0)
        self.test_corpus1 = self._read_file(os.path.join(self._path, 'sentiment.test.1'), label=1)
        self._build_vocab(config.max_vocab_cnt)
        self.unk = "<unk>"
        print("Done loading corpus")

    def _read_file(self, path, label=0):
        with open(path, 'r') as f:
            lines = f.readlines()

        return self._process_data(lines, label=label)

    def _process_data(self, data, label=0):
        all_text = []
        all_lens = []
        for line in data:
            tokens = [BOS] + line.strip().split() + [EOS]
            all_lens.append(len(tokens))
            all_text.append(Pack(utt=tokens, speaker=0, label=label))
        print("Max utt len %d, mean utt len %.2f" % (
            np.max(all_lens), float(np.mean(all_lens))))
        return all_text

    def _build_vocab(self, max_vocab_cnt):
        all_words = []
        for turn in (self.train_corpus0 + self.train_corpus1):
            all_words.extend(turn.utt)

        vocab_count = Counter(all_words).most_common()
        raw_vocab_size = len(vocab_count)
        discard_wc = np.sum([c for t, c, in vocab_count[max_vocab_cnt:]])
        vocab_count = vocab_count[0:max_vocab_cnt]

        # create vocabulary list sorted by count
        print("Load corpus with train size %d, valid size %d, "
              "test size %d raw vocab size %d vocab size %d at cut_off %d OOV rate %f"
              % (len(self.train_corpus0 + self.train_corpus1), len(self.valid_corpus0 + self.valid_corpus1),
                 len(self.test_corpus0 + self.test_corpus1),
                 raw_vocab_size, len(vocab_count), vocab_count[-1][1],
                 float(discard_wc) / len(all_words)))

        self.vocab = [PAD] + [t for t, cnt in vocab_count]
        if UNK not in self.vocab:
            self.vocab = [PAD, UNK] + [t for t, cnt in vocab_count]
        self.rev_vocab = {t: idx for idx, t in enumerate(self.vocab)}
        self.unk_id = self.rev_vocab[UNK]

    def _sent2id(self, sent):
        return [self.rev_vocab.get(t, self.unk_id) for t in sent]

    def _to_id_corpus(self, data):
        results = []
        for line in data:
            id_turn = Pack(utt=self._sent2id(line.utt),
                           speaker=line.speaker,
                           label=line.label,
                           meta=line.get('meta'))
            results.append(id_turn)
        return results

    def get_corpus(self):
        id_train = self._to_id_corpus(self.train_corpus0 + self.train_corpus1)
        id_valid = self._to_id_corpus(self.valid_corpus0 + self.valid_corpus1)
        id_test = self._to_id_corpus(self.test_corpus0 + self.test_corpus1)
        return Pack(train=id_train, valid=id_valid, test=id_test)