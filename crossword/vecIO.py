import io
import numpy as np

datatable = []
for i in range(10):
	datatable.append({})
	
#return the appropriate data table for the word (based on word length)
def get_dataTable(word):
	if (len(word)) > 10:
		return datatable[9]
	return datatable[len(word) - 1]

#load vectors into datatable
def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    for line in fin:
        tokens = line.rstrip().split(' ')
        get_dataTable(tokens[0])[tokens[0]] = map(float, tokens[1:])

#get vector for a word
def get_vector(word):
	return(np.asarray(list(get_dataTable(word)[word]), dtype=np.float32))

#get table for all words of length n
def get_n_length_table(n):
	return datatable[n-1]