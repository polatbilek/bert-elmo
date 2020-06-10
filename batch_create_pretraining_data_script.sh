#!/bin/bash

rm -rf /tmp/pretraining_output
rm /tmp/tf_examples_0.tfrecord
rm /tmp/tf_examples_1.tfrecord
rm /tmp/tf_examples_2.tfrecord

export BERT_BASE_DIR=./bert_base

for INDEX in {0..211}
do
python3 create_pretraining_data.py   --input_file=$BERT_BASE_DIR/sliced_bert_data/wiki_data_${INDEX}.txt   --output_file=/tmp/tf_examples_${INDEX}.tfrecord   --vocab_file=$BERT_BASE_DIR/word_vocab.txt   --do_lower_case=True   --max_seq_length=96   --max_predictions_per_seq=15   --masked_lm_prob=0.15   --random_seed=12345   --dupe_factor=5
done






