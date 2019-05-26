import io
import json

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data

data = load_vectors('wiki-news-300d-1M.vec')

def get_vectors(word):
	return(list(data[word]))

print(get_vectors(','))