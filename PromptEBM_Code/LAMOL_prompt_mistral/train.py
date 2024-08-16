import torch
from torch.utils.data import DataLoader
from torch import nn
from pytorch_transformers import AdamW, WEIGHTS_NAME, WarmupLinearSchedule
import csv
import numpy as np
import os, sys, json
import logging
from fp16 import FP16_Module, FP16_Optimizer
from parallel import DataParallelModel, DataParallelCriterion
from collections import OrderedDict
from utils import *
from settings import args, TASK_DICT, init_logging, MODEL_CONFIG, MODEL_CLASS, SPECIAL_TOKENS, CONFIG_CLASS
from settings import TOKENIZER, SPECIAL_TOKEN_IDS, FILL_VAL, SAVE_NAME, FINAL_SAVE_NAME, TOKENS_WEIGHT, CONFIG_NAME
from scheduler import AnnealingLR
from regularizers import REG_TYPES, REG_TYPE_KEYS, Weight_Regularized_AdamW, Weight_Regularized_SGD
from torch.nn import CrossEntropyLoss
logger = logging.getLogger(__name__)
import random
from ebm_models.dgmvae.dataset import corpora
from ebm_models.dgmvae.dataset import data_loaders
from ebm_models.dgmvae.models.sent_models import *
from ProgressivePrompts.Mistral_codebase.mistral_continual import *
from transformers import BitsAndBytesConfig
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

def get_model(corpus_client, task_id):
    try:
        ebm_model = eval(args.ebm_model)(corpus_client, args)
        if args.forward_only:
            ebm_model.load_state_dict(torch.load(args.ebm_model_file + "model_ckpt_" + str(task_id) + ".pt"))
            # ebm_model.load_state_dict(torch.load(args.ebm_model_file))
        else:
            for param in ebm_model.parameters():
                param.data.uniform_(-0.1, 0.1)
    except Exception as e:
        raise NotImplementedError("Fail to build model %s" % (args.ebm_model))
    if args.use_gpu:
        ebm_model.cuda()
    return ebm_model

def set_seed(seed, deterministic=True):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        if not deterministic:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        else:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

def getQA_dataset(train_dataset, mode, tasks, train_extra_data):
    qadata = QADataset(train_dataset, mode, SPECIAL_TOKEN_IDS[tasks[0]], train_extra_data)
    return qadata

def prepare_dirs_loggers(config, script=""):
    logFormatter = logging.Formatter("%(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setLevel(logging.DEBUG)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    # if config.forward_only:
    #     return

    if not os.path.exists(config.ebm_log_dir):
        os.makedirs(config.ebm_log_dir)

    config.time_stamp = get_time()
    config.script = script
    dir_name = "{}-{}".format(config.time_stamp, script) if script else config.time_stamp
    config.session_dir = os.path.join(config.ebm_log_dir, dir_name)
    os.mkdir(config.session_dir)

    fileHandler = logging.FileHandler(os.path.join(config.session_dir,
                                                   'session.log'))
    fileHandler.setLevel(logging.DEBUG)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    # save config
    param_path = os.path.join(config.session_dir, "params.json")
    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)

def get_corpus_client(data_paths_train=None, data_paths_valid=None, data_paths_test=None, TOKENIZER=None):
    if "ptb" in args.ebm_data.lower():
        corpus_client = corpora.PTBCorpus(args)
    elif "daily_dialog" in args.ebm_data.lower():
        corpus_client = corpora.DailyDialogCorpus(args)
    elif "stanford" in args.ebm_data.lower():
        corpus_client = corpora.StanfordCorpus(args)
    elif "lamol" in args.ebm_data.lower():
        gen_token = "GEN"
        corpus_client = corpora.QACorpus(args, gen_token, SPECIAL_TOKEN_IDS, data_paths_train, data_paths_valid, data_paths_test,TOKENIZER)
    else:
        raise ValueError("Only support three corpus: ptb, daily_dialog and stanford.")
    return corpus_client

def get_ebm_dataloader(config, corpus):
    if config.ebm_data.lower() == "ptb":
        dataloader = data_loaders.PTBDataLoader
    elif config.ebm_data.lower() == "daily_dialog":
        dataloader = data_loaders.DailyDialogSkipLoader
    elif config.ebm_data.lower() == "stanford":
        dataloader = data_loaders.SMDDataLoader
    elif config.ebm_data.lower() == 'lamol':
        dataloader = data_loaders.QADataLoader
    else:
        raise ValueError("Only support three corpus: ptb, daily_dialog and stanford.")

    train_dial, valid_dial, test_dial = corpus['train'], \
                                        corpus['valid'], \
                                        corpus['test']

    train_feed = dataloader("Train", train_dial, config)
    valid_feed = dataloader("Valid", valid_dial, config)
    test_feed = dataloader("Test", test_dial, config)

    return train_feed, valid_feed, test_feed

def train(task_ids, model, continual_learner):
    tasks = [args.tasks[task_id] for task_id in task_ids]

    logger.info("start to train { task: %s, seq train type: %s }" % (tasks, args.seq_train_type))
    model_dir = get_model_dir(tasks)
    make_dir(model_dir)

    train_dataset = [TASK_DICT[t]["train"] for t in tasks]
    train_extra_data = []

    gen_token = get_gen_token(tasks[0])
    TOKENIZER.add_tokens([gen_token])
    TOKENIZER.save_pretrained(model_dir)
    SPECIAL_TOKENS[tasks[0]] = gen_token
    SPECIAL_TOKEN_IDS[tasks[0]] = TOKENIZER.convert_tokens_to_ids(gen_token)
    logger.info('gen token = {} , gen token id = {}'.format(gen_token, SPECIAL_TOKEN_IDS[tasks[0]]))

    if "lll" in args.seq_train_type and task_ids[0] > 0 and not args.skip_tasks:
        prev_task = args.tasks[task_ids[0]-1]
        with torch.no_grad():
            create_extra_data(tasks[0], prev_task, model, train_extra_data)
    elif "gem" in args.seq_train_type and task_ids[0] > 0: 
        get_real_data(tasks[0], train_extra_data, accum=False, encode=True)
        args.memory_data.append(train_extra_data)
        train_extra_data = []
    elif "ebm" in args.seq_train_type and task_ids[0] > 0:
        set_seed(args.seed)
        prepare_dirs_loggers(args, os.path.basename(__file__))
        train_corpus = getQA_dataset(train_dataset, "train", tasks, train_extra_data)
        valid_dataset = [TASK_DICT[t]["eval"] for t in tasks]
        test_dataset = [TASK_DICT[t]["test"] for t in tasks]
        corpus_client = get_corpus_client(train_dataset, valid_dataset, test_dataset, TOKENIZER)
        # ebm_model = get_model(corpus_client, task_ids[0] - 1)
        dial_corpus = corpus_client.get_corpus()
        train_feed, valid_feed, test_feed = get_ebm_dataloader(args, dial_corpus)
        prev_task = args.tasks[task_ids[0] - 1]
        # max_utt_lens = [500, 500, 500, 500, 500]
        # max_dec_lens = [500, 500, 500, 500, 500]
        for i in range(0, task_ids[0]):
            prev_task_i = args.tasks[i]
            # args.max_utt_len = max_utt_lens[i]
            # args.max_dec_len = max_dec_lens[i]
            prev_train_dataset = [TASK_DICT[prev_task_i]["train"]]
            prev_valid_dataset = [TASK_DICT[prev_task_i]["eval"]]
            prev_test_dataset = [TASK_DICT[prev_task_i]["test"]]
            prev_corpus_client = get_corpus_client(prev_train_dataset, prev_valid_dataset, prev_test_dataset, TOKENIZER)
            prev_dial_corpus = prev_corpus_client.get_corpus()
            ebm_model = get_model(prev_corpus_client, i)
            # ebm_model = get_model(prev_corpus_client)
            prev_train_feed, prev_valid_feed, prev_test_feed = get_ebm_dataloader(args, prev_dial_corpus)
            create_extra_data_ebm(tasks[0], prev_train_feed, prev_task, model, ebm_model, train_extra_data)
    logger.info('extra training data size: {}'.format(len(train_extra_data)))

    if not model:
        # which_model_to_load = model_dir if os.path.isfile(os.path.join(model_dir, FINAL_SAVE_NAME)) else args.model_name
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = MODEL_CLASS.from_pretrained(args.model_name, quantization_config=bnb_config, device_map="auto")
        model.resize_token_embeddings(len(TOKENIZER))
        fsdp_plugin = FullyShardedDataParallelPlugin(
            state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
            optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
        )

        accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
        model = accelerator.prepare_model(model)
        if not args.fp32:
            model = FP16_Module(model)

    # freeze_weights = args.freeze_weights
    # freeze_except = args.freeze_except
    # if freeze_weights:
    #     print('Freezing weights')
    #     continual_learner.do_freeze_weights(model, except_condition=freeze_except)

    # initialize prompt
    # model.prompt = continual_learner.model.prompt

    with torch.no_grad():
        model.prompt = nn.Parameter(torch.tensor(continual_learner.init_new_prompt(continual_learner.prefix_len),
                                                 requires_grad=True))
    # model.to(args.device_ids[0])

    MODEL_CONFIG.vocab_size = len(TOKENIZER)
    MODEL_CONFIG.to_json_file(os.path.join(model_dir,CONFIG_NAME))
    global TOKENS_WEIGHT
    print("len of tokenizer: {}".format(len(TOKENIZER)))
    print('len of tokens weight: {}'.format(TOKENS_WEIGHT.shape[0]))
    # if len(TOKENIZER) != TOKENS_WEIGHT.shape[0]:
        # TOKENS_WEIGHT = torch.cat((TOKENS_WEIGHT, torch.ones(len(TOKENIZER) - TOKENS_WEIGHT.shape[0]).cuda()))
    special_tokens = {"ans_token": '__ans__', "pad_token": '__pad__', "unk_token": '__unk__',
                      "eos_token": '<|endoftext|>'}
    if args.use_sep:
        special_tokens["sep_token"] = '__sep__'
    special_token_ids = {k: TOKENIZER.convert_tokens_to_ids(v) for k, v in special_tokens.items()}
    TOKENS_WEIGHT = torch.ones([MODEL_CONFIG.vocab_size], dtype=torch.float).cuda()
    TOKENS_WEIGHT[special_token_ids["ans_token"]] = args.tokens_weight
    if args.use_sep:
        TOKENS_WEIGHT[special_token_ids["sep_token"]] = args.tokens_weight

    eos_token = '<|endoftext|>'
    SPECIAL_TOKEN_ID_END = TOKENIZER.convert_tokens_to_ids(eos_token)
    logger.info('eos token = {} , eos token id = {}'.format(eos_token, SPECIAL_TOKEN_ID_END))

    if args.skip_tasks and len(tasks) == 1:
        logger.info("*********** skip task: {} ***********".format(tasks[0]))
        if tasks[0] in args.skip_tasks:
            if len(args.skip_tasks) == 1:
                model_dir = get_model_dir(tasks)
                model_path = os.path.join(model_dir, FINAL_SAVE_NAME)
                config_path = os.path.join(model_dir,CONFIG_NAME)
                model_config = CONFIG_CLASS.from_json_file(config_path)
                model = MODEL_CLASS(model_config).cuda()
                state_dict = torch.load(model_path)
                model.load_state_dict(state_dict)
                if not args.fp32:
                    model = FP16_Module(model)
                if args.seq_train_type in REG_TYPE_KEYS:
                    logger.info("calulating reg_params ...")
                    train_qadata = QADataset(train_dataset, "train", SPECIAL_TOKEN_IDS[tasks[0]], train_extra_data)
                    max_train_batch_size = max(len(train_qadata) // args.min_n_steps // 30, args.min_batch_size)
                    train_dataloader = create_dataloader(train_qadata, "train", max_train_batch_size)
                    parallel_model = DataParallelModel(WrapModel(model), args.device_ids)
                    regularizer = REG_TYPES[args.seq_train_type](model, parallel_model, [train_dataloader], tasks[0])
                    regularizer.task_start_do()
                    regularizer.task_end_do()
                    torch.save(model.state_dict(), os.path.join(model_dir, FINAL_SAVE_NAME))
                    logger.info("done reg_params!")
            args.skip_tasks.remove(tasks[0])
            return model

    model.resize_token_embeddings(len(TOKENIZER))

    if not args.fp32:  # again because resize_token_embeddings makes embedding layer fp32
        model = FP16_Module(model)

    parallel_model = DataParallelModel(WrapModel(model), args.device_ids)

    train_qadata = QADataset(train_dataset, "train", SPECIAL_TOKEN_IDS[tasks[0]], train_extra_data)
    max_train_batch_size = max(len(train_qadata) // args.min_n_steps // 30, args.min_batch_size)
    train_dataloader = create_dataloader(train_qadata, "train", max_train_batch_size)
    if not args.unbound and args.seq_train_type != "multitask":
        #n_train_epochs = TASK_DICT[tasks[0]]["n_train_epochs"]
        n_train_epochs = args.n_train_epochs[tasks[0]]
    else:
        n_train_epochs = args.n_train_epochs['_'.join(tasks)]
    n_train_optimization_steps = len(train_qadata) * n_train_epochs
    logger.info('len of train dataset: {} , max train batch size {} , num of opt steps: {}'.format(
        len(train_qadata), max_train_batch_size, n_train_optimization_steps))

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if "gem" in args.seq_train_type:
        model.task_id = task_ids[0]
        if not hasattr(model, "grad_dims"):
            model.grad_dims = []

            for param in model.parameters():
                model.grad_dims.append(param.data.numel())
        if not hasattr(model, "grads"):
            model.grads = torch.zeros(sum(model.grad_dims),len(args.tasks))
            model.grads = model.grads.cuda()

    if args.seq_train_type in REG_TYPE_KEYS:
        optimizer = Weight_Regularized_AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    else:
        optimizer = AdamW(optimizer_grouped_parameters, weight_decay=args.weight_decay, lr=args.learning_rate, eps=args.adam_epsilon)
    if not args.fp32:
        optimizer = FP16_Optimizer(optimizer, static_loss_scale=None, dynamic_loss_scale=True,
                                   dynamic_loss_args={'scale_window': 100, 'min_scale': 1, 'delayed_shift': 2})

    scheduler = AnnealingLR(optimizer, start_lr=args.learning_rate, warmup_iter=int(args.n_warmup_ratio*len(train_qadata)),
            num_iters=int(n_train_optimization_steps), decay_style=args.decay_style)
    train_loss_fct = DataParallelCriterion(CrossEntropyLoss(ignore_index=FILL_VAL, weight=TOKENS_WEIGHT), args.device_ids)

    if args.seq_train_type in REG_TYPE_KEYS:
        copy_train_dataloader = create_dataloader(train_qadata, "train", max_train_batch_size)
        prev_task = args.tasks[task_ids[0]-1]
        regularizer = REG_TYPES[args.seq_train_type](model, parallel_model, [copy_train_dataloader], tasks[0], prev_task)
        regularizer.task_start_do()

    tot_n_steps = 0
    train_once = TrainStep(model, optimizer, scheduler)
    if "gem" in args.seq_train_type and task_ids[0] != 0:
        gem_step = GEMStep(model, parallel_model, train_loss_fct, optimizer)
    # prompt_optimizer = AdamW([model.prompt], weight_decay=args.weight_decay, lr=args.learning_rate, eps=args.adam_epsilon)
    model.train()
    for ep in range(n_train_epochs):
        cum_loss, cum_qa_loss, cum_lm_loss, cur_n_inputs = 0, 0, 0, 0
        for n_steps, (cqs, len_cqs, cqa, _, Y, gen_X, gen_Y) in enumerate(train_dataloader):
            batch = (gen_X, gen_Y)
            prompt_loss = continual_learner.train_step_lester(model, batch, FILL_VAL, TOKENIZER,
                                                              TOKENS_WEIGHT, task=None, progressive=args.progressive)

            n_inputs = sum(_cqa.shape[0] for _cqa in cqa)

            for i in range(len(cqa)):
                cqa[i] = (cqa[i].to(args.device_ids[i]),)
                Y[i] = Y[i].to(args.device_ids[i])
                gen_X[i] = (gen_X[i].to(args.device_ids[i]),)
                gen_Y[i] = gen_Y[i].to(args.device_ids[i])

            losses = get_losses(parallel_model, cqa, Y, gen_X, gen_Y, train_loss_fct)
            prompt_loss = args.prompt_lambda * prompt_loss
            loss = sum(losses, prompt_loss)
            if "gem" in args.seq_train_type and task_ids[0] != 0:
                gem_step(task_ids[0])
            train_once(loss, n_inputs)

            qa_loss = losses[0].item() * n_inputs
            lm_loss = losses[1].item() * n_inputs
            cum_loss += (qa_loss + lm_loss)
            cum_qa_loss += qa_loss
            cum_lm_loss += lm_loss
            cur_n_inputs += n_inputs

            if (n_steps + 1 ) % args.logging_steps == 0:
                logger.info('progress {:.3f} , lr {:.1E} , loss {:.3f} , qa loss {:.3f} , lm loss {:.3f} , avg batch size {:.1f}'.format(
                    ep + cur_n_inputs/len(train_qadata), scheduler.get_lr(), cum_loss/cur_n_inputs, cum_qa_loss/cur_n_inputs, cum_lm_loss/cur_n_inputs,
                    cur_n_inputs/(n_steps + 1)
                ))

        torch.save(model.state_dict(), os.path.join(model_dir, SAVE_NAME+str(ep+1)))
        tot_n_steps += (n_steps + 1)
        logger.info('epoch {}/{} done , tot steps {} , lr {:.1E} , loss {:.2f} , qa loss {:.2f} , lm loss {:.2f} , avg batch size {:.1f}'.format(
            ep+1, n_train_epochs, tot_n_steps, scheduler.get_lr(), cum_loss/cur_n_inputs, cum_qa_loss/cur_n_inputs, cum_lm_loss/cur_n_inputs, cur_n_inputs/(n_steps+1)
        ))
        logger.info('epoch {}/{} done , tot steps {} , lr {:.1E} , prompt loss {:.2f}'.format(
            ep+1, n_train_epochs, tot_n_steps, scheduler.get_lr(), prompt_loss
        ))

    if args.progressive:
        continual_learner.progress_previous_prompts(model, task=tasks[0])
    else:
        if continual_learner.early_stopping:
            continual_learner.restore_best_model()

    # task end do for reg
    if args.seq_train_type in REG_TYPE_KEYS:
        regularizer.task_end_do()
    torch.save(model.state_dict(), os.path.join(model_dir, FINAL_SAVE_NAME))

    return model


if __name__ == '__main__':

    if not args.debug:
        logging.getLogger("pytorch_transformers").setLevel(logging.WARNING)
        logging.getLogger("pytorch_transformers.tokenization_utils").setLevel(logging.CRITICAL)

    make_dir(args.model_dir_root)

    init_logging(os.path.join(args.model_dir_root, 'log_train.txt'))
    logger.info('args = {}'.format(str(args)))

    print(f'start to debugging')
    model = None
    if args.seq_train_type == "multitask":
        model = train(list(range(len(args.tasks))), model)
    else:
        if args.unbound:
            TASK_DICT = lll_unbound_setting(split_size=args.unbound)
        model_name = args.model_name
        task_list = args.tasks
        continual_learner = MistralContinualLearner(model_name,
                                                 task_list,
                                                 batch_size=args.batch_size,
                                                 select_k_per_class=args.select_k_per_class,
                                                 prefix_len=args.prefix_len,
                                                 freeze_weights=args.freeze_weights == 1,
                                                 freeze_except=args.freeze_except,
                                                 lr=args.lr,
                                                 seq_len=args.seq_len,
                                                 early_stopping=False,
                                                 prefix_MLP=args.prefix_MLP,
                                                 prefix_path=args.prefix_path if args.prefix_path != '' else None,
                                                 mlp_layer_norm=args.mlp_layer_norm == 1,
                                                 bottleneck_size=args.bottleneck_size,
                                                 get_test_subset=args.get_test_subset == 1,
                                                 memory_perc=args.memory_perc,
                                                 )
        for task_id in range(len(args.tasks)):
            model = train([task_id], model, continual_learner)
