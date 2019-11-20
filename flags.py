class flags():
	def __init__(self):
		self.filters = [[1, 32], [2, 32], [3, 64], [4, 128], [5, 256], [6, 512], [7, 1024]]
		self.activations = "relu"
		self.n_highway = 2
		self.embedding_dim = 128
		self.char_vocab_size = 1997
		self.max_characters_per_token = 50
		self.projection_dim = 1024
		self.bert_embedding_size = 1024 #the dim for BERT's input embeddings
		self.sequence_length = 96
		self.batch_size = 8
		self.max_prediction_per_seq = 15
		self.input_file = "/tmp/tf_examples.tfrecord"
		self.vocab_file = "/media/darg2/hdd/yl_tez/bert_project/vocab.txt"
		self.word_vocab_size = 32267


FLAGS = flags()