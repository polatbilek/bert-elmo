import numpy as np
import tensorflow as tf
from flags import FLAGS
import h5py
import json
import re
import sys

DTYPE = 'float32'
DTYPE_INT = 'int64'

class BidirectionalLanguageModelGraph(object):
	'''
	Creates the computational graph and holds the ops necessary for runnint
	a bidirectional language model
	'''

	def __init__(self, ids_placeholder,
				 use_character_inputs=True,
				 max_batch_size=128):

		self._max_batch_size = max_batch_size
		self.ids_placeholder = ids_placeholder
		self.use_character_inputs = use_character_inputs

		self._n_tokens_vocab = None

		with tf.variable_scope('bilm'):
			self._build_word_char_embeddings()

	def _build_word_char_embeddings(self):

		projection_dim = FLAGS.projection_dim
		filters = FLAGS.filters
		n_filters = sum(f[1] for f in filters)
		max_chars = FLAGS.max_characters_per_token
		char_embed_dim = FLAGS.embedding_dim
		n_chars = FLAGS.char_vocab_size

		if n_chars != 262:
			print("!!!!!!!!!!!!!!!!!!!!!!1")
			print("Set n_characters=262 after training see the README.md")
			sys.exit()

		if FLAGS.activations == 'tanh':
			activation = tf.nn.tanh
		elif FLAGS.activations == 'relu':
			activation = tf.nn.relu

		# the character embeddings
		with tf.device("/cpu:0"):
			self.embedding_weights = tf.get_variable(
				"char_embed", [n_chars, char_embed_dim],
				dtype=DTYPE,
				initializer=tf.random_uniform_initializer(-1.0, 1.0)
			)
			# shape (batch_size, unroll_steps, max_chars, embed_dim)
			self.char_embedding = tf.nn.embedding_lookup(self.embedding_weights, self.ids_placeholder)

		# the convolutions
		def make_convolutions(inp, reuse):
			with tf.variable_scope('CNN', reuse=reuse) as scope:
				convolutions = []
				for i, (width, num) in enumerate(filters):
					if FLAGS.activations == 'relu':
						# He initialization for ReLU activation
						# with char embeddings init between -1 and 1
						# w_init = tf.random_normal_initializer(
						#    mean=0.0,
						#    stddev=np.sqrt(2.0 / (width * char_embed_dim))
						# )

						# Kim et al 2015, +/- 0.05
						w_init = tf.random_uniform_initializer(
							minval=-0.05, maxval=0.05)
					elif FLAGS.activations == 'tanh':
						# glorot init
						w_init = tf.random_normal_initializer(
							mean=0.0,
							stddev=np.sqrt(1.0 / (width * char_embed_dim))
						)
					w = tf.get_variable(
						"W_cnn_%s" % i,
						[1, width, char_embed_dim, num],
						initializer=w_init,
						dtype=DTYPE)
					b = tf.get_variable(
						"b_cnn_%s" % i, [num], dtype=DTYPE,
						initializer=tf.constant_initializer(0.0))
					conv = tf.nn.conv2d(
						inp, w,
						strides=[1, 1, 1, 1],
						padding="VALID") + b

					# now max pool
					conv = tf.nn.max_pool(
						conv, [1, 1, max_chars - width + 1, 1],
						[1, 1, 1, 1], 'VALID')

					# activation
					conv = activation(conv)
					conv = tf.squeeze(conv)

					convolutions.append(conv)

			return tf.concat(convolutions, axis=-1)

		reuse = tf.get_variable_scope().reuse
		embedding = make_convolutions(self.char_embedding, reuse)

		self.token_embedding_layers = [embedding]

		# for highway and projection layers:
		#   reshape from (batch_size, n_tokens, dim) to
		n_highway = FLAGS.n_highway
		use_highway = n_highway is not None and n_highway > 0
		use_proj = n_filters != projection_dim

		if use_highway or use_proj:
			embedding = tf.reshape(embedding, [-1, n_filters])


		# set up weights for projection
		if use_proj:
			assert n_filters > projection_dim
			with tf.variable_scope('CNN_proj') as scope:
				W_proj_cnn = tf.get_variable(
					"W_proj", [n_filters, projection_dim],
					initializer=tf.random_normal_initializer(
						mean=0.0, stddev=np.sqrt(1.0 / n_filters)),
					dtype=DTYPE)
				b_proj_cnn = tf.get_variable(
					"b_proj", [projection_dim],
					initializer=tf.constant_initializer(0.0),
					dtype=DTYPE)

		# apply highways layers
		def high(x, ww_carry, bb_carry, ww_tr, bb_tr):
			carry_gate = tf.nn.sigmoid(tf.matmul(x, ww_carry) + bb_carry)
			transform_gate = tf.nn.relu(tf.matmul(x, ww_tr) + bb_tr)
			return carry_gate * transform_gate + (1.0 - carry_gate) * x

		if use_highway:
			highway_dim = n_filters

			for i in range(n_highway):
				with tf.variable_scope('CNN_high_%s' % i) as scope:
					W_carry = tf.get_variable(
						'W_carry', [highway_dim, highway_dim],
						# glorit init
						initializer=tf.random_normal_initializer(
							mean=0.0, stddev=np.sqrt(1.0 / highway_dim)),
						dtype=DTYPE)
					b_carry = tf.get_variable(
						'b_carry', [highway_dim],
						initializer=tf.constant_initializer(-2.0),
						dtype=DTYPE)
					W_transform = tf.get_variable(
						'W_transform', [highway_dim, highway_dim],
						initializer=tf.random_normal_initializer(
							mean=0.0, stddev=np.sqrt(1.0 / highway_dim)),
						dtype=DTYPE)
					b_transform = tf.get_variable(
						'b_transform', [highway_dim],
						initializer=tf.constant_initializer(0.0),
						dtype=DTYPE)

				embedding = high(embedding, W_carry, b_carry,
								 W_transform, b_transform)

				self.token_embedding_layers.append(
					tf.reshape(embedding,[FLAGS.batch_size, FLAGS.sequence_length, highway_dim])
				)

		# finally project down to projection dim if needed
		if use_proj:
			embedding = tf.matmul(embedding, W_proj_cnn) + b_proj_cnn

			self.token_embedding_layers.append(
				tf.reshape(embedding,
						   [FLAGS.batch_size, FLAGS.sequence_length, projection_dim])
			)

		# reshape back to (batch_size, tokens, dim)
		if use_highway or use_proj:
			shp = [FLAGS.batch_size, FLAGS.sequence_length, projection_dim]
			embedding = tf.reshape(embedding, shp)


		self.embedding = tf.squeeze(embedding)