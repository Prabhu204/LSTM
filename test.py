"""
author: Prabhu

"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn

from src.evaluation_metrics import *
from src.prep_dataset import Dataset_

def get_args():
    parser = argparse.ArgumentParser("""Implementation of Character level LSTM for text classification""")
    parser.add_argument("-d", "--dataset", type=str, default="Data")
    parser.add_argument("-b", "--batch_size", type=int, default= 64)
    parser.add_argument("-m", "--max_length", type=int, default= 3000)

    parser.add_argument("-g", "--gpu", action="store_true", default=False)
    parser.add_argument("-i", "--import_model", default='trained_model')

    parser.add_argument("-r", "--save_result", type=str, default="Result")
    parser.add_argument("-rn", "--file_name", type=str, default="used_param_testdata.txt")

    args = parser.parse_args()
    return args

def test(opt):

    with open(opt.save_result + os.sep + opt.file_name, "w") as file:
        file.write("Selected model parameters: {}".format(vars(opt)))

    test_dataset = Dataset_(opt.dataset + os.sep + "test1.csv", opt.dataset + os.sep + "classes.txt",
                           opt.max_length)
    test_loader = DataLoader(test_dataset, opt.batch_size, shuffle=False, num_workers=0)

    if torch.cuda.is_available():
        model = torch.load(opt.import_model)
    else:
        model = torch.load(opt.import_model, map_location=lambda storage, loc: storage)

    criterion = nn.CrossEntropyLoss()
    # testing model
    model.eval()
    with torch.no_grad():
        test_loss = 0
        predictions = []
        target_list = []
        for batch in test_loader:
            if opt.gpu:
                batch = [Variable(record).cuda() for record in batch]
            else:
                batch = [Variable(record) for record in batch]
            text, labels = batch
            outputs = model(text)
            test_loss += criterion(outputs, labels)  # to sum up batch loss
            pred = outputs.argmax(1)  # get the index of the max log-probability
            predictions.extend(pred)
            target_list.extend(labels)
        print('Test Accuracy of the model on test data: {}'.format(metrics.accuracy_score(target_list, predictions)))
        print('Con_mat: {} \n'.format(metrics.confusion_matrix(target_list, predictions)))

        with open(opt.save_result + os.sep + opt.file_name, 'a') as f:
            f.write('\n Accuracy of the model on test_data: {} '.format(metrics.accuracy_score(target_list, predictions)))
            f.write('\n Con_mat: {} \n'.format(metrics.confusion_matrix(target_list, predictions)))

if __name__ == '__main__':
    opt = get_args()
    test(opt)