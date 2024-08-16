import os
from datetime import datetime
import logging

#from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs
#from model_basedon_huggingface.seq2seq_model import Seq2SeqModel, Seq2SeqArgs
from seq2seq_model import Seq2SeqModel, Seq2SeqArgs
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.ERROR)
from para_load import load_test_df
from para_argparser import get_para_argparser
#from model_basedon_huggingface.score import run_score
from score import run_score

def get_args():
    args = get_para_argparser()
    return args

def para_eval(args):
    model_args = Seq2SeqArgs()
    model_args.best_model_dir = args.output_dir+'/best_model'
    model_args.do_sample = True
    model_args.local_rank = args.local_rank
    model_args.eval_batch_size = args.per_gpu_eval_batch_size
    model_args.evaluate_during_training = True
    model_args.evaluate_during_training_steps = 2500
    model_args.evaluate_during_training_verbose = True
    model_args.fp16 = False
    model_args.learning_rate = 5e-5
    model_args.max_length = 64
    model_args.max_seq_length = args.max_seq_length
    model_args.num_beams = None
    model_args.num_return_sequences = 5
    model_args.num_train_epochs = args.num_train_epochs
    model_args.overwrite_output_dir = True
    model_args.reprocess_input_data = True
    model_args.save_eval_checkpoints = False
    model_args.save_steps = -1
    model_args.top_k = 50
    model_args.top_p = 0.95
    #model_args.train_batch_size = 8
    model_args.use_multiprocessing = False
    model_args.wandb_project = "Paraphrasing with BART"
    model_args.cuda_device = args.cuda_device
    model_args.start_from_pretrain = args.start_from_pretrain
    model = Seq2SeqModel(
        encoder_decoder_type="bart",
        encoder_decoder_name=args.model_name_or_path, ##right now, it may still be outputs for recadam since I forgot to change the name for best_model so that best_model is saved into this folder
        args=model_args,
        use_cuda=True,
        cuda_device=model_args.cuda_device
    )

    #_, eval_df = load_df()
    # source_path = args.data_dir +'/test.source.quora_lower'
    # target_path = args.data_dir + '/test.target'
    # source_path = args.data_dir + '/test.source'
    # target_path = args.data_dir + '/test.target'
    source_path = args.data_dir + '/test_1000.source'
    target_path = args.data_dir + '/test_1000.target'
    eval_df = load_test_df(source_path, target_path)

    to_predict = [
        prefix + ": " + str(input_text)
        for prefix, input_text in zip(eval_df["prefix"].tolist(), eval_df["input_text"].tolist())
    ]
    query = eval_df["input_text"].tolist()
    truth = eval_df["target_text"].tolist()

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
            wrote = False
            for j, pred in enumerate(preds[i]):
                if not wrote:
                    # pred = str(pred)[10:]
                    if ':' in pred:
                        colon_ind = pred.find(':')
                        pred = pred[colon_ind + 1:].strip()
                    else:
                        pred = pred.strip()
                    if len(pred)==0 and j<len(preds[i])-1:
                        continue
                    else:
                        wrote = True
                        f.write(str(pred) + "\n")
                else:
                    wrote = False
                    break

    with open(f"{args.output_dir}/test.target", "w") as f:
        for line in truth:
            f.write(line + '\n')

    with open(f"{args.output_dir}/test.source", "w") as f:
        for line in query:
            f.write(line + '\n')
    test_target = args.output_dir+'/test.target'
    pred_output = args.output_dir+f'/predictions/test_{time_stamp}.hypo'
    return test_target, pred_output

if __name__ == "__main__":
    args = get_args()

    #args.model_name_or_path = args.output_dir
    test_target, pred_output = para_eval(args)
    #test_target = 'data/quora/output_adamw_bbll_adamw_lr4e-5_msl64_bs96/test.target'
    #pred_output = 'data/quora/output_adamw_bbll_adamw_lr4e-5_msl64_bs96/predictions/test_2022-02-08 01:02:44.247876.hypo'
    #os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
    run_score(test_target, pred_output, args.cuda_device)
