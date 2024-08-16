#!/usr/bin/env bash
export PYTHONPATH=../../:../:`pwd`/:$PYTHONPATH
bash run_para_train.sh 4e-5 64 3 RecAdam facebook/bart-large 30 ./data/quora/data4fairseq/ ./data/quora/output_ppll_recadam_lr4e05_msl64 96 power power layer layer