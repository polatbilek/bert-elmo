import tensorflow as tf
import numpy as np
from cnn_model import BidirectionalLanguageModelGraph
import random
import sys
from run_pretraining import input_fn_builder
from flags import FLAGS

ids = tf.placeholder(tf.int64, shape=(8, 64, 256)) #[batch_size, num_of_words, num_of_chars_in_word]
ids_numpy = [[[random.randint(1, 250) for i in range(256)] for j in range(64)] for k in range(8)]

input_files = []
for input_pattern in FLAGS.input_file.split(","):
	input_files.extend(tf.gfile.Glob(input_pattern))

input_fn = input_fn_builder(input_files=input_files,
				 max_seq_length=FLAGS.max_prediction_per_seq,
				 max_predictions_per_seq=FLAGS.max_prediction_per_seq,
				 is_training=True,
				 num_cpu_threads=4)

a = input_fn(params={"batch_size":8})
o = tf.Session().run([a.output_classes["input_ids"]])
print(o)
"""
graph = BidirectionalLanguageModelGraph(ids)

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
	sess.run(tf.global_variables_initializer())
	sess.run(ids, feed_dict={ids: ids_numpy})

	embeddings, c = sess.run([graph.embedding, graph.char_embedding], feed_dict={ids:ids_numpy})

	print("*******")
	print(np.shape(embeddings))


"""