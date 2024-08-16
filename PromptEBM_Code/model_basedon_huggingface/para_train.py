import os
from datetime import datetime
import logging

#from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs
from model_basedon_huggingface.seq2seq_model import Seq2SeqModel, Seq2SeqArgs
#from seq2seq_bart_model import Seq2SeqModel, Seq2SeqArgs
#from seq2seq_crladam_model import Seq2SeqModel, Seq2SeqArgs
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.ERROR)
from para_load import load_df, load_test_df, load_df_generic
from para_argparser import get_para_argparser

def get_args():
    args = get_para_argparser()
    return args

def para_train(args):
    model_args = Seq2SeqArgs()
    model_args.best_model_dir = args.output_dir+'/best_model'
    model_args.output_dir = args.output_dir
    model_args.do_sample = True
    model_args.local_rank = args.local_rank
    model_args.eval_batch_size = 64
    model_args.evaluate_during_training = False
    model_args.evaluate_during_training_steps = 2500
    model_args.evaluate_during_training_verbose = False
    model_args.fp16 = False
    model_args.learning_rate = args.learning_rate
    model_args.max_length = args.max_seq_length
    model_args.max_seq_length = args.max_seq_length
    model_args.num_beams = None
    model_args.optimizer = args.optimizer
    model_args.num_return_sequences = args.num_return_sequences
    model_args.num_train_epochs = args.num_train_epochs
    model_args.overwrite_output_dir = True
    model_args.reprocess_input_data = True
    model_args.save_eval_checkpoints = False
    model_args.save_steps = args.save_steps
    model_args.top_k = 50
    model_args.top_p = 0.95
    model_args.train_batch_size = args.per_gpu_train_batch_size
    model_args.use_multiprocessing = False
    #model_args.wandb_project = "Paraphrasing with BART"
    model_args.cuda_device = args.cuda_device
    model_args.start_from_pretrain = args.start_from_pretrain
    model_args.encoder_norm_self = args.encoder_norm_self
    model_args.decoder_norm_self = args.decoder_norm_self
    model_args.encoder_norm_ff = args.encoder_norm_ff
    model_args.decoder_norm_ff = args.decoder_norm_ff
    model_args.warmup_updates = args.warmup_updates

    if not os.path.exists(args.output_dir):
        print(f'args.output_dir = {args.output_dir}')
        os.makedirs(args.output_dir, exist_ok=True)

    '''
    here, the basic logic is that we initialize our current model with pretrained models. No matter for lifelong learning or for finetune learning model, we should 
    only have one model for our initialization. So, we should not have a prev_model different from encoder_decoder_name
    '''
    model = Seq2SeqModel(
        encoder_decoder_type="bart",
        #encoder_decoder_name=args.model_name_or_path, #"facebook/bart-large",
        encoder_decoder_name=args.model_name_or_path,
        #prev_model="facebook/bart-large",
        #prev_model=args.model_name_or_path,
        # encoder_decoder_name="facebook/bart-large",
        # prev_model=args.model_name_or_path,
        args=model_args,
        use_cuda=True,
        cuda_device=model_args.cuda_device
    )

    if 'quora' in args.data_dir:
        train_df, eval_df = load_df()
    else:
        train_source_path = args.data_dir +'/train.source'
        train_target_path = args.data_dir + '/train.target'
        train_df = load_df_generic(train_source_path, train_target_path)
        eval_source_path = args.data_dir +'/test.source'
        eval_target_path = args.data_dir + '/test.target'
        eval_df = load_test_df(eval_source_path, eval_target_path)

    model.train_model(train_df, eval_data=eval_df, args=args)
    # model._load_model_args(model_dir)

    to_predict = [
        prefix + ": " + str(input_text)
        for prefix, input_text in zip(eval_df["prefix"].tolist(), eval_df["input_text"].tolist())
    ]
    query = eval_df["input_text"].tolist()
    truth = eval_df["target_text"].tolist()

    '''
    considering training memory is too large,  the following may be OOM. We set do_eval usually False
    This way, let's finish training without evaluation. Then, let's restart evaluation in a separate python script.
    '''
    if args.do_eval:
        preds = model.predict(to_predict)
        # Saving the predictions if needed
        os.makedirs(f"{args.output_dir}/predictions", exist_ok=True)

        with open(f"{args.output_dir}/predictions/predictions_{datetime.now()}.txt", "w") as f:
            print('save prediction path')
            print(f'{args.output_dir}/predictions/predictions_{datetime.now()}.txt')
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
        # output_dir='output/output_bart/alexa_qr_data_training_10-11-testing_12_100k/predictions'
        time_stamp = datetime.now()

        with open(f"{args.output_dir}/predictions/test_{time_stamp}.hypo", "w") as f:
            for i, text in enumerate(eval_df["input_text"].tolist()):
                for j, pred in enumerate(preds[i]):
                    if j == 0:
                        # pred = str(pred)[10:]
                        if ':' in pred:
                            colon_ind = pred.find(':')
                            pred = pred[colon_ind + 1:].strip()
                        else:
                            pred = pred.strip()
                        f.write(str(pred) + "\n")

        with open(f"{args.output_dir}/test.target_quora", "w") as f:
            for line in truth:
                f.write(line + '\n')

        with open(f"{args.output_dir}/test.source_quora", "w") as f:
            for line in query:
                f.write(line + '\n')
        test_target = args.output_dir+'/test.target_quora'
        pred_output = args.output_dir+'/predictions/test_'+str(time_stamp)+'.hypo'
        return test_target, pred_output
    else:
        return model_args.output_dir

if __name__ == "__main__":
    args = get_args()
    #args.model_name_or_path = args.output_dir
    if args.do_eval:
        test_target, pred_output = para_train(args)
    else:
        train_model_path = para_train(args)
