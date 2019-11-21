import tokenization

fWrite = open("./bert_base/ooc.txt", "a+" , encoding="utf8")

tokenizer = tokenization.FullTokenizer(
      vocab_file="./bert_base/char_vocab.txt", do_lower_case=True)

with open("./bert_base/test.txt", "r",encoding="utf8") as fRead:
	for line in fRead:
		for token in tokenizer.tokenize(line):
			for char in token:
				if char not in list(tokenizer.vocab.keys()):
					fWrite.write(char+"\n")
					tokenizer.vocab[char] = len(list(tokenizer.vocab.keys()))

fWrite.close()

