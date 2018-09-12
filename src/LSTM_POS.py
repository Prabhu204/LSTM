"""
author: Prabhu

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def prepare_data(seq, idx):
    index_id = [idx[w] for w in seq]
    return torch.tensor(index_id, dtype = torch.long)

sample_data = [
    ("John likes the blue house at the end of the street".split(), ["NNP", "V","DET","JJ","NN","IN","DET","NN","IN","DET","NN"]),
    ("Viet was crazy dude".split(), ["NNP", "V","JJ","NN"])
]

word_to_ix = {}
for sent, pos in sample_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

print(word_to_ix)

tag_to_ix = {}
for sent, tag in sample_data:
    for tag_ in tag:
        if tag_ not in tag_to_ix:
            tag_to_ix[tag_] = len(tag_to_ix)
print(tag_to_ix)

emb_dim = 10
hidden_dim = 10


class LSTMpos(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, voca_size, tagset_size):
        super(LSTMpos, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(voca_size, embedding_dim)

        #  The LSTM takes word embeddings as inputs, and outputs hidden states with dimensionality
        #  with dimensionality hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        # the layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()


    def init_hidden(self):

        #  before doing any thing, we do not have any hidden state
        #  The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1,1,self.hidden_dim),
                torch.zeros(1,1,self.hidden_dim))

    def forward(self, sentence):
        embedings_ = self.word_embeddings(sentence)
        lstm_output, self.hidden = self.lstm(embedings_.view(len(sentence), 1,-1), self.hidden)
        tag_space = self.hidden2tag(lstm_output.view(len(sentence), -1))
        tag_score = F.log_softmax(tag_space, dim = 1)
        return tag_score


model = LSTMpos(emb_dim, hidden_dim, len(word_to_ix), len(tag_to_ix))

loss = nn.NLLLoss()
opt = torch.optim.SGD(model.parameters(), lr= 0.1)

with torch.no_grad():
    inputs = prepare_data(sample_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)

for epoch in range(300):
    for sentence, tags in sample_data:
        #   Pytorch accumulate gradients, we need to clear them before each instance
        model.zero_grad()

        #  also we need to clear out the hidden state of the LSTM, detaching it from its history on the
        #  last instance
        model.hidden = model.init_hidden()

        #  getting inputs ready for the network, that is turn them into Tensor of word indices
        sentence_in = prepare_data(sentence, word_to_ix)
        targets = prepare_data(tags ,tag_to_ix)

        # run forward pass
        tag_scores = model(sentence_in)

        # compute loss, gradients and update the parameters
        loss_ = loss(tag_scores, targets)
        loss_.backward()
        opt.step()

# see what the scores are after training

with torch.no_grad():
    inputs = prepare_data(sample_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)
