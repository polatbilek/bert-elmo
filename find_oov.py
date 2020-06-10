import tokenization

fWrite = open("./bert_base/oov.txt", "a+" , encoding="utf8")

tokenizer = tokenization.FullTokenizer(
      vocab_file="a.txt", do_lower_case=True)

with open("./bert_base/test.txt", "r",encoding="utf8") as fRead:
	for line in fRead:
		for token in tokenizer.tokenize(line):
			if token not in list(tokenizer.vocab.keys()):
				fWrite.write(token+"\n")
				tokenizer.vocab[token] = len(list(tokenizer.vocab.keys()))

fWrite.close()

