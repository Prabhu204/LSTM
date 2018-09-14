"""
author: Prabhu

"""
import torch
import torch.nn as nn
from  torch.autograd import Variable

# Devise configeration

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTMchar(nn.Module):
    def __int__(self, embedding_dim, hidden_dim, vocab_size,output, n_layers):
        super(LSTMchar, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.output = output

        self.encoder = nn.Embedding(vocab_size,embedding_dim) # embeddings
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers)
        self.h_layer2out = nn.Linear(hidden_dim, output)

    def forward(self, input, hidden):
        batch_size = input.size(0)
        # def forward(self, x,h) where x = input, h = hidden
        self.encoded = self.encoder(input.view(1,-1))   # embed word ids to vectors
                                                        # x = self.encoder(x)
        output, hidden = self.lstm(self.encoded.view(1,batch_size,-1), hidden) # forward propagate LSTM
                                                                        # out, (h,c) = self.lstm(x, h)
        # reshape output to (batch_size*sequence_length, hidden_dim)
        output = self.h_layer2out(output.view(batch_size, -1))

        return output, hidden

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_dim)),
                Variable(torch.zeros(self.n_layers,batch_size,self.hidden_dim)))
