import io
import numpy as np

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    i = 0
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
        i += 1
        if i > 50: break
    return data

data = load_vectors('wiki-news-300d-1M.vec')

def get_vectors(word):
	return(np.asarray(list(data[word]), dtype=np.float32))