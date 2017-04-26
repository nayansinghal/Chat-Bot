
from gensim.models.word2vec import LineSentence
from gensim.models.word2vec import Word2Vec
import logging
import multiprocessing
import os
import sys

if __name__ == "__main__":

	program = os.path.basename(sys.argv[0])
	logger = logging.getLogger(program)

	logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
	logging.root.setLevel(level=logging.INFO)
	logger.info("Running %s" % ' '.join(sys.argv))

	params = {
	'size': 50,
	'window': 10,
	'min_count': 10,
	'workers': max(1, multiprocessing.cpu_count() - 1),
	'sample': 1E-5,
	}

	max_length = 0
	inp = sys.argv[1]
	outp = sys.argv[2]
	with open(inp, 'r') as f:
		for line in f.readlines():
			max_length = max(max_length, len(line))

	logger.info("Max article length: {} words.".format(max_length))

	word2vec = Word2Vec(LineSentence(inp, max_sentence_length=max_length),
						**params)
	word2vec.save(outp)