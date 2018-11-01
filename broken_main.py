import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
from pathlib import Path 
from broken_model import model
import time
from tqdm import tqdm

random.seed(42) # DO NOT CHANGE
np.random.seed(0) # DO NOT CHANGE
torch.manual_seed(2) # DO NOT CHANGE

def read_input():
	path = Path("data")
	data_in = path / ("data.in")
	data_out = path / ("data.out")

	with open(data_in, encoding='utf-8', errors='ignore') as data_in_file:
		data_inputs = [line.split() for line in data_in_file if len(line.strip()) > 0] #1
	data_in_file.close()

	with open(data_out, encoding='utf-8', errors='ignore') as data_out_file:
		data_outputs = [line.split() for line in data_out_file if len(line.strip()) > 0] #1
	data_out_file.close()

	all_data = list(zip(data_inputs, data_outputs))

	# The below splits data into a train and set. This is not an error. 
	# However, feel free to decrease the ratio if you are willing to compromise performance
	# To make training faster/so you can iterate through debugging solutions faster

	random.shuffle(all_data) #2

	train_ratio = 0.9  # splits data set into 90% train, 10% test
	train_bound = int(train_ratio * len(all_data))

	train_data = all_data[:train_bound] 
	test_data = all_data[train_bound:] 
	#random.shuffle(train_data) # Shuffling data is a good practice
	#random.shuffle(test_data)
	train_inputs, train_outputs = zip(*train_data) # unzips the list, i.e. [(a,b), (c,d)] -> [a,c], [b,d]
	test_inputs, test_outputs = zip(*test_data)
	return train_inputs, train_outputs, test_inputs, test_outputs


def build_indices(train_set):
	tokens = [token for line in train_set for token in line]

	# From token to its index
	forward_dict = {'UNK': 0}

	# From index to token
	backward_dict = {0: 'UNK'}
	i = 1
	for token in tokens:
		if token not in forward_dict:
			forward_dict[token] = i 
			backward_dict[i] = token 
			i += 1
	return forward_dict, backward_dict


def encode(data, forward_dict):
	return [list(map(lambda t: forward_dict.get(t,0), line)) for line in data]


def make_output(output):
	if output == ['0']:
		return torch.tensor([0])
	else:
		return torch.tensor([1])

if __name__ == '__main__':
	train_inputs, train_outputs, test_inputs, test_outputs = read_input()
	forward_dict, backward_dict = build_indices(train_inputs)
	train_inputs = encode(train_inputs, forward_dict)
	test_inputs = encode(test_inputs, forward_dict)
	m = model(vocab_size = len(forward_dict), hidden_dim = 64, out_dim = 2) #3
	optimizer = optim.SGD(m.parameters(), lr=1.0)
	minibatch_size = 100
	num_minibatches = len(train_inputs) // minibatch_size

	for epoch in (range(5)):
		# Training
		print("Training")
		# Put the model in evaluation mode
		m.train()
		start_train = time.time()

		for group in tqdm(range(num_minibatches)):
			predictions = None
			gold_outputs = None
			loss = 0
			optimizer.zero_grad()
			for i in range(group * minibatch_size, (group + 1) * minibatch_size):
				input_seq = train_inputs[i]
				gold_output = make_output(train_outputs[i])
				prediction_vec, prediction = m(input_seq)
				if predictions is None:
					predictions = [prediction_vec]
					gold_outputs = [gold_output]
				else:
					predictions.append(prediction_vec)
					gold_outputs.append(gold_output)
			loss = m.compute_Loss(torch.stack(predictions), torch.stack(gold_outputs).squeeze())
			loss.backward()
			optimizer.step()
		print("Training time: {} for epoch {}".format(time.time() - start_train, epoch))

		# Evaluation
		print("Evaluation")
		# Put the model in evaluation mode
		m.eval()
		start_eval = time.time()

		predictions = 0 # number of predictions
		correct = 0 # number of outputs predicted correctly
		for input_seq, gold_output in zip(train_inputs, train_outputs):
			_, predicted_output = m(input_seq)
			gold_output = (0 if gold_output == ['0'] else 1)
			correct += int((gold_output == predicted_output))
			predictions += 1
		accuracy = correct / predictions
		assert 0 <= accuracy <= 1
		print("Evaluation time: {} for epoch {}, Accuracy: {}".format(time.time() - start_eval, epoch, accuracy))
