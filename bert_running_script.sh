#!/bin/bash

rm -rf /tmp/pretraining_output
rm /tmp/tf_examples.tfrecord
export BERT_BASE_DIR=./bert_base

python3 create_pretraining_data.py   --input_file=$BERT_BASE_DIR/test_turkish.txt   --output_file=/tmp/tf_examples.tfrecord   --vocab_file=$BERT_BASE_DIR/turkish_wiki_vocab.txt   --do_lower_case=True   --max_seq_length=96   --max_predictions_per_seq=15   --masked_lm_prob=0.15   --random_seed=12345   --dupe_factor=5

python3 run_pretraining.py   --input_file=/tmp/tf_examples.tfrecord   --output_dir=/tmp/pretraining_output   --do_train=True   --do_eval=True   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --train_batch_size=8   --max_seq_length=96   --max_predictions_per_seq=15  --save_checkpoints_steps=100000   --num_train_steps=500000   --num_warmup_steps=50000   --learning_rate=2e-5

