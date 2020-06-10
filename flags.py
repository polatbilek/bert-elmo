class flags():
	def __init__(self):
		self.filters = [[1, 32], [2, 32], [3, 64], [4, 128], [5, 256], [6, 512], [7, 32]]
		self.activations = "relu"
		self.n_highway = 2
		self.embedding_dim = 128
		self.char_vocab_size = 8618
		self.max_characters_per_token = 40
		self.projection_dim = 1024
		self.bert_embedding_size = 1024 #the dim for BERT's input embeddings
		self.sequence_length = 96
		self.batch_size = 8
		self.max_prediction_per_seq = 15
		self.input_file = "/tmp/tf_examples.tfrecord"
		self.vocab_file = "./bert_base/turkish_wiki_vocab.txt"
		self.word_vocab_size = 159012


FLAGS = flags()
