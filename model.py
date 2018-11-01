import os
import random
import numpy as np
import math
import torch
from torch import nn
from torch.nn import init
from torch import optim

# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html

class model(nn.Module):

    def __init__(self, vocab_size, hidden_dim=64):
        super(model, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeds = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.LSTM(hidden_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim)
        self.loss = nn.CrossEntropyLoss()
        self.out = nn.Linear(hidden_dim, vocab_size)

    def compute_Loss(self, pred_vec, gold_seq):
        return self.loss(pred_vec, gold_seq)
        
    def forward(self, input_seq, gold_seq=None):
        input_vectors = self.embeds(torch.tensor(input_seq))
        input_vectors = input_vectors.unsqueeze(1)
        outputs, hidden = self.encoder(input_vectors)

        # Technique used to train RNNs: 
        # https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/
        teacher_force = True

        # This condition tells us whether we are in training or inference phase 
        if gold_seq is not None and teacher_force:
            gold_vectors = torch.cat([torch.zeros(1, self.hidden_dim), self.embeds(torch.tensor(gold_seq[:-1]))], 0)
            gold_vectors = gold_vectors.unsqueeze(1)
            gold_vectors = torch.nn.functional.relu(gold_vectors)
            outputs, hidden = self.decoder(gold_vectors, hidden)
            predictions = self.out(outputs)
            predictions = predictions.squeeze()
            vals, idxs = torch.max(predictions, 1)
            return predictions, list(np.array(idxs))
        else:
            prev = torch.zeros(1, 1, self.hidden_dim)
            predictions = []
            predicted_seq = []
            for i in range(len(input_seq)):
                prev = torch.nn.functional.relu(prev)
                outputs, hidden = self.decoder(prev, hidden)
                pred_i = self.out(outputs)
                pred_i = pred_i.squeeze()
                _, idx = torch.max(pred_i, 0)
                idx = idx.item()
                predictions.append(pred_i)
                predicted_seq.append(idx)
                prev = self.embeds(torch.tensor([idx]))
                prev = prev.unsqueeze(1)
            return torch.stack(predictions), predicted_seq