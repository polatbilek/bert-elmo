#!/bin/bash

rm -rf /tmp/pretraining_output
rm /tmp/tf_examples.tfrecord
export BERT_BASE_DIR=./bert_base

python3 run_pretraining.py   --input_file=/tmp/tf_examples_0.tfrecord   --output_dir=/tmp/pretraining_output   --do_train=True   --do_eval=True   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --train_batch_size=8   --max_seq_length=96   --max_predictions_per_seq=15   --num_train_steps=30000   --num_warmup_steps=3000   --learning_rate=2e-5


