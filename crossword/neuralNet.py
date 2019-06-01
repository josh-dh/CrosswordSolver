import numpy as np
import vecIO
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import csv
import re

def process_clue(clue):
	word_array = clue.split(' ')
	outvec = np.zeros(300, dtype=np.float32)
	for word in word_array:
		outvec = outvec + vecIO.get_vector(word)
	outvec = outvec / len(word_array)
	return outvec

class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		self.fc1 = nn.Linear(300, 300)

	def forward(self, x):
		return F.softmax(self.fc1(x))

def custom_loss_fn (process_phrase, answer):
	hadamard = process_phrase * answer
	return 1- math.sqrt(dot(hadamard, hadamard))

loss_fn = nn.MSELoss()

clueProcessor = Model()
optimizer = optim.SGD(clueProcessor.parameters(), lr=0.01, momentum=0.9)

#load vectors
vecIO.load_vectors('../wiki-news-300d-1M.vec')
#load training data
with open('../clues2.csv', 'r') as training_file:
	reader = csv.reader(training_file)
	i = 0
	for row in reader:
		#clean strings
		clue = re.sub(r'[^\w\s]','',row[0].lower()) #get clue STRING, not vector
		ans = vecIO.get_vector(re.sub(r'[^\w\s]','',row[1].lower())) #get answer VECTOR, not string
		pred = clueProcessor(clue) #get VECTOR prediction for clue STRING

		#get loss
		loss = loss_fn(pred, ans)

		#optimize
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if i % 10 == 0:
			print(loss.item())

		if i == 500: break

		i += 1