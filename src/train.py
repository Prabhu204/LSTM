"""
author: Prabhu

"""
import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from src.evaluation_metrics import *
from src.Sentiment_analysis import LSTMchar
from src.prep_dataset import Dataset_

# def get_args():
#     parser = argparse.ArgumentParser("""Implementation of Character level CNN for text classification""")
#     parser.add_argument("-a", "--alphabet", type =str,
#                         default= """abcdefghijklmnopqrstuvwxyz0123456789,;. !?:’/"\|_@#$%ˆ&*̃+'-=<>()[]{}""")
#
#     parser.add_argument("-m", "--max_length", type=int, default= 1014)
#     parser.add_argument("-p", "--optimizer", choices=["sgd", "adam"], default="sgd")
#     parser.add_argument("-b", "--batch_size", type=int, default= 120)
#     parser.add_argument("-n", "--num_epochs", type= int, default= 10)
#     parser.add_argument("-l", "--lr", type= float, choices=[0.01,0.001], default= 0.01, help=" recommended for sgd 0.01 and for adam 0.001")
#     parser.add_argument("-d", "--dataset", type=str, default="Data")
#     parser.add_argument("-g", "--gpu", action="store_true", default=True)
#     parser.add_argument("-s", "--save_path", type= str, default="Data")
#     parser.add_argument("-t", "--model_name", type=str, default= "trained_model")
#     parser.add_argument("-r", "--save_result", type=str, default="Result")
#     parser.add_argument("-rn", "--result_name", type = str)
#     args = parser.parse_args()
#     return args
#

# def train(opt):
#
#     res_file = open(opt.save_result+ os.sep + opt.result_name, "w")
#     res_file.write("Select model parameters: {}".format(vars(opt)))
#
#     training_set = Dataset_(opt.dataset+ os.sep+"train.csv", opt.dataset+os.sep+"classes.txt",
#                             opt.max_length)
#
#     train_data_generator = DataLoader(training_set, shuffle= True, num_workers= 0, batch_size= opt.batch_size)

    # model = LSTMchar()




    # if opt.optimizer == "sgd":
    #     optimizer = torch.optim.SGD(model.parameters(),lr= opt.lr, momentum=0.9)
    # else :
    #     optimizer= torch.optim.Adam(model.parameters(),lr= opt.lr)
    #
    # if opt.gpu:
    #     model.cuda()
    #
    # criterion = nn.CrossEntropyLoss()
    # model.train()
    # num_epoch_iter = len(train_data_generator)
    # best_accuracy = 0
    # for epoch in range(opt.num_epochs):
    #     for iter, batch in enumerate(train_data_generator):
    #         _, n_true_label = batch
    #         if opt.gpu:
    #             batch = [Variable(record).cuda() for record in batch]
    #         else:
    #             batch = [Variable(record) for record in batch]
    #
    #         b_data, b_true_label = batch
    #
    #         optimizer.zero_grad()
    #         pred_label= model(b_data)
    #         prob_label = pred_label.cpu().data.numpy()
    #         loss = criterion(pred_label, b_true_label)
    #         loss.backward()
    #         optimizer.step()
    #         train_metrics = get_metrics(b_true_label, prob_label, list_metrics=['Accuracy', 'Loss','Confusion_matrics'])
    #         res = "Training: Iteraion: {}/{} Epoch: {}/{} Accuracy:{} Loss:{}".format(iter+1,num_epoch_iter,epoch+1,opt.num_epochs,
    #             train_metrics['Accuracy'], loss)
    #
    #         print(res)
    #     torch.save(model, opt.save_path + os.sep + opt.model_name)

# if __name__ == '__main__':
#     opt = get_args()
#     train(opt)


training_set = Dataset_('/home/prabhu/LSTM/Data/train.csv')

train_data_generator = DataLoader(training_set, shuffle= True, num_workers= 0, batch_size= opt.batch_size)

