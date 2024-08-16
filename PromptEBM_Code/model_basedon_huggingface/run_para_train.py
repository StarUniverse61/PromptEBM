from para_train import para_train, get_args
from model_basedon_huggingface.score import run_score
args = get_args()
test_target, pred_output = para_train(args)
run_score(test_target, pred_output)