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
from src.Sentiment_analysis import LSTMChar
from src.prep_dataset import Dataset_

def get_args():
    parser = argparse.ArgumentParser("""Implementation of Character level LSTM for text classification""")
    parser.add_argument("-a", "--alphabet", type =str,
                        default= """abcdefghijklmnopqrstuvwxyz0123456789,;. !?:’/"\|_@#$%ˆ&*̃+'-=<>()[]{}""")
    parser.add_argument("-d", "--dataset", type=str, default="Data")
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-m", "--max_length", type=int, default=3000)

    parser.add_argument("-i", "--input_size", type=int, default=69)
    parser.add_argument("-hs", "--hidden_size", type=int, default=128)
    parser.add_argument("-o", "--num_classes", type=int, default=2)
    parser.add_argument("-l", "--num_layers", type=int, default=1)
    parser.add_argument("-v", "--vocab", type=int, default=69)
    parser.add_argument("-em", "--embedding_dim", type=int, default=69)

    parser.add_argument("-p", "--optimizer", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("-n", "--num_epochs", type= int, default= 10)
    parser.add_argument("-lr", "--lr", type= float, choices=[0.01,0.001], default= 0.01, help=" recommended for sgd 0.01 and for adam 0.001")
    parser.add_argument("-g", "--gpu", action="store_true", default=False)

    parser.add_argument("-t", "--model_name", type=str, default="trained_model")
    parser.add_argument("-s", "--save_model", type= str, default="")
    parser.add_argument("-r", "--save_result", type=str, default="Result")

    parser.add_argument("-rn", "--file_name", type = str, default="used_param.txt")
    args = parser.parse_args()
    return args

def train(opt):

    with open(opt.save_result + os.sep + opt.file_name, "w") as file:
        file.write("Selected model parameters: {}".format(vars(opt)))

    train_dataset = Dataset_(opt.dataset + os.sep + "train.csv", opt.dataset + os.sep + "classes.txt",
                            opt.max_length)

    train_loader = DataLoader(train_dataset, opt.batch_size, shuffle=True, num_workers= 0)

    dev_dataset = Dataset_(opt.dataset + os.sep + "dev.csv", opt.dataset + os.sep + "classes.txt",
                             opt.max_length)
    dev_loader = DataLoader(dev_dataset, opt.batch_size, shuffle=False, num_workers= 0)

    model = LSTMChar(input_size=opt.input_size, hidden_size=opt.hidden_size, num_classes=opt.num_classes,num_layers= opt.num_layers,
                     vocab= opt.vocab, embedding_dim= opt.embedding_dim)

    if opt.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(),lr= opt.lr, momentum=0.9)
    else :
        optimizer= torch.optim.Adam(model.parameters(),lr= opt.lr)

    if opt.gpu:
        model.cuda()
    else:
        model.cpu()

    criterion = nn.CrossEntropyLoss()
    total_step = len(train_loader)
    for epoch in range(opt.num_epochs):
        model.train()
        print('epoch {}'.format(epoch + 1))
        print('*' * 10)
        for step, batch in enumerate(train_loader):  # gives batch_size
            # Forward propagation
            if opt.gpu:
                batch = [Variable(record).cuda() for record in batch]
            else:
                batch = [Variable(record) for record in batch]
            b_x, b_y = batch
            # b_x is reshped to (batch_size, step_size, input_size)
            out = model(b_x)
            loss = criterion(out, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (step + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, opt.num_epochs, step + 1, total_step, loss.item()))

        # validate model
        model.eval()
        with torch.no_grad():
            test_loss = 0
            predictions= []
            target_list= []
            for text, labels in dev_loader:
                labels = labels
                outputs = model(text)
                test_loss += criterion(outputs, labels)  # to sum up batch loss
                pred = outputs.argmax(1)  # get the index of the max log-probability
                predictions.extend(pred)
                target_list.extend(labels)
            print('Test Accuracy of the model on test data: {}'.format(metrics.accuracy_score(target_list,predictions)))
            print('Con_mat: {} \n'.format(metrics.confusion_matrix(target_list, predictions)))

            with open(opt.save_result + os.sep + opt.file_name, 'a') as f:
                f.write('\n Accuracy of the model on dev_data: {}   Epoch: {}'.format(metrics.accuracy_score(target_list, predictions),
                                                                                 epoch +1 ))
                f.write('\n Con_mat: {} \n'.format(metrics.confusion_matrix(target_list, predictions)))

        torch.save(model.state_dict(), opt.save_model+os.sep+opt.model_name)

if __name__ == '__main__':
    opt = get_args()
    train(opt)



