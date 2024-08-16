#!/bin/bash -i
export PYTHONPATH=../../:`pwd`/:$PYTHONPATH
source ./env.example

python test_pn_calibration.py \
  --data_dir $DATA_DIR \
  --model_dir_root $MODEL_ROOT_DIR \
  "$@"
