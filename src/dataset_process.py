"""
author: Prabhu

"""
import os
import torch
from torch.utils.data.dataset import Dataset
from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe, Vectors
from torchtext.data import TabularDataset
import numpy as np
import pandas as pd

# class Dictionary(object):
#     def __init__(self):
#         self.word2index = {}
#         self.idx2word = []
#
#     def add_word(self,word):
#         if word not in self.word2index:
#             self.idx2word.append(word)
#             self.word2index[word] = len(self.idx2word)-1
#         return self.word2index[word]
#
#     def __len__(self):
#         return len(self.idx2word)


def load_dataset(sen = None):
    tokenize = lambda x : x.split()
    Text = data.Field(sequential= True, tokenize = tokenize, lower= True, include_lengths= True, pad_first= True, batch_first= True, fix_length= 200)
    Label = data.LabelField(tensor_type= torch.FloatTensor)
    train_data, test_data = datasets.IMDB.splits(Text, Label)
    # print(pd.DataFrame(eval(train_data)))
    m = train_data()
    print(m)
    Text.build_vocab(train_data, vectors = GloVe(name= '6B', dim = 300))
    # Text.build_vocab(train_data, vectors = 'glove.6B.300d')
    Label.build_vocab(train_data)

    word_embeddings = Text.vocab.vectors

    # print("Length of the Tex Vocabulary:"+ str(len(Text.vocab)))
    # print("Vector size of the Text Vocabulary:", Text.vocab.vectors.size())
    # print("Label length:" + str(len(Label.vocab)))

    # train_data, valid_data = train_data.split()
    train_iter,  test_iter = data.BucketIterator.splits((train_data, test_data),
                                                                   batch_size= 32,
                                                                   sort_key = lambda x : len(x.text),
                                                                   repeat = False,
                                                                   shuffle = True)
    #  or
    #  train_iter, test_iter = datasets.IMDB.iter(batch_size = 32)

    vocab_size = len(Text.vocab)
    # print(vocab_size)
    return Text, vocab_size, word_embeddings, train_iter, test_iter


if __name__ == '__main__':
    Text, vocab_size, word_embeddings, train_iter, test_iter = load_dataset()
    print(word_embeddings.size())
    print(type(train_iter))

    # for idx, batch in enumerate(train_iter):
    #     # print(idx)
    #     # print(batch)
    #     text = batch.text[1]
    #     target = batch.label
    #     print(text)
    #     print(target)

