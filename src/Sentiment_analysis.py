"""
author: Prabhu

"""
import torch
import torch.nn as nn
from  torch.autograd import Variable
import numpy as np

class LSTMChar(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers, vocab ,embedding_dim):
        super(LSTMChar, self).__init__()
        self.input_size= input_size       # input size
        self.hidden_size = hidden_size    # hidden dimension
        self.num_classes = num_classes
        self.num_layers = num_layers


        self.encoder = nn.Embedding( vocab, embedding_dim)

        self.lstm = nn.LSTM(input_size,hidden_size, batch_first= True)

        #self.h2h = nn.Linear(n_hidden, n_hidden)
        self.h2o = nn.Linear(hidden_size, num_classes)
        #self.output = nn.LogSoftmax()
        #self.dropout = nn.Dropout(p = 0.2)


    def forward(self, x):
        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        out,_ = self.lstm(x, (h0,c0))  # none represents zero intial-state
        # choose r_out at last time step
        out = self.h2o(out[:, -1, :])
        return out