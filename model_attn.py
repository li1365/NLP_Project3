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

    def __init__(self, mode, vocab_size, hidden_dim=64):
        super(model, self).__init__()
        self.mode = mode
        self.hidden_dim = hidden_dim
        self.embeds = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.LSTM(hidden_dim, hidden_dim)
        if self.mode == 'default':
            self.decoder = nn.LSTM(hidden_dim, hidden_dim)
        else:
            self.decoder = nn.LSTM(hidden_dim * 2, hidden_dim)
        self.loss = nn.CrossEntropyLoss()
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.attn = Attn(self.mode, hidden_dim)

    def compute_Loss(self, pred_vec, gold_seq):
        return self.loss(pred_vec, gold_seq)
        
    def forward(self, input_seq, gold_seq=None):
        input_vectors = self.embeds(torch.tensor(input_seq))
        input_vectors = input_vectors.unsqueeze(1)
        encoder_outputs, hidden = self.encoder(input_vectors)

        # Technique used to train RNNs: 
        # https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/
        teacher_force = True

        # This condition tells us whether we are in training or inference phase 
        if gold_seq is not None and teacher_force:
            prev = torch.zeros(1, 1, self.hidden_dim)
            predictions = []
            predicted_seq = []
            for i in range(len(input_seq)):
                prev = torch.nn.functional.relu(prev)
                if self.mode != "default":
                    context = self.attn(encoder_outputs, hidden[0])
                    input = torch.cat((prev, context), dim=2)
                    outputs, hidden = self.decoder(input, hidden)
                else:
                    outputs, hidden = self.decoder(prev, hidden)
                pred_i = self.out(outputs)
                pred_i = pred_i.squeeze()
                _, idx = torch.max(pred_i, 0)
                idx = idx.item()
                predictions.append(pred_i)
                predicted_seq.append(idx)
                prev = self.embeds(torch.tensor(gold_seq[i]))
                prev = prev.unsqueeze(0).unsqueeze(1)
            return torch.stack(predictions), predicted_seq
        else:
            prev = torch.zeros(1, 1, self.hidden_dim)
            predictions = []
            predicted_seq = []
            for i in range(len(input_seq)):
                prev = torch.nn.functional.relu(prev)
                if self.mode != "default":
                    context = self.attn(encoder_outputs, hidden[0])
                    input = torch.cat((prev, context), dim=2)
                    outputs, hidden = self.decoder(input, hidden)
                else:
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

class Attn(nn.Module):
    def __init__(self, mode, hidden_size, attn_size=128):
        super(Attn, self).__init__()
        self.mode = mode
        self.U = nn.Linear(hidden_size, attn_size)
        self.W = nn.Linear(hidden_size, attn_size)
        self.V = nn.Linear(attn_size, 1)
        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()
        #self.combine = nn.Linear(hidden_size+hidden_size, hidden_size)

    def forward(self, encoder_outputs, hidden):
        if self.mode == 'add':
            W_hidden = self.W(hidden.squeeze())
            U_encode = self.U(encoder_outputs.squeeze())
            #poly = U_encode + W_hidden.expand_as(U_encode)
            poly = torch.add(U_encode, W_hidden)
            poly_tanh = self.tanh(poly)
            poly_v = self.V(poly_tanh)
            temp = poly_v
        else:
            W_hidden = self.W(hidden.squeeze())
            temp = encoder_outputs.squeeze().unsqueeze(0)
            poly = torch.bmm(encoder_outputs.squeeze().unsqueeze(0), W_hidden.unsqueeze(0).unsqueeze(2))
            temp = poly.squeeze().unsqueeze(1)
        attn_weights = self.softmax(temp)
        context = torch.bmm(attn_weights.t().unsqueeze(0),encoder_outputs.squeeze().unsqueeze(0))
        #context = attn_weights.t().matmul(encoder_outputs.squeeze())
        return context