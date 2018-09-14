"""
author: Prabhu

"""

import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import argparse
from src.Sentiment_analysis import LSTMchar
from torchext import data



def get_args():
    parser = argparse.ArgumentParser("""Implementation of Character level LSTM for sentiment analysis""")
    parser.add_argument("-a", "--alphabet", type=str,
                        default="""abcdefghijklmnopqrstuvwxyz0123456789,;. !?:’/"\|_@#$%ˆ&*̃+'-=<>()[]{}""")

    parser.add_argument("-m", "--max_length", type=int, default=1014)
    parser.add_argument("-o", "--optimizer", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("-b", "--batch_size", type=int, default=5)
    parser.add_argument("-e", "--num_epochs", type=int, default=10)
    parser.add_argument("-l", "--lr", type=float, choices=[0.01, 0.001], default=0.01,
                        help=" recommended for sgd 0.01 and for adam 0.001")
    parser.add_argument("-d", "--dataset", type=str, default="Data")
    parser.add_argument("-g", "--gpu", action="store_true", default=True)
    parser.add_argument("-s", "--save_path", type=str, default="Data")
    parser.add_argument("-t", "--model_name", type=str, default="trained_model")
    parser.add_argument("-r", "--save_result", type=str, default="Result")
    parser.add_argument("-rn", "--result_name", type=str)
    parser.add_argument("-sl","--sentence_length", type=int, default= 32)
    parser.add_argument("-hd", "--hidden_dim", type=int, default=50)
    parser.add_argument("-ed","--embedding_dim",type=int, default=100)
    parser.add_argument("-nl", "--n_layers", type=int, default= 2, help= 'Number of stacked RNN layer')
    args = parser.parse_args()
    return args
opt = get_args()
# def adjust_lr(optimizer, epoch):
#     lr = opt.lr*(0.1**(epoch//10))

def train(opt):



    model = LSTMchar()