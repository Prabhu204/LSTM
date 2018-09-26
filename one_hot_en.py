"""
author: Prabhu

"""
from src.prep_dataset import Dataset_
import torch.nn as nn
import torch
from torch import optim
from sklearn import metrics
from torch.utils.data import DataLoader
from torch.autograd import Variable



input = 69 # because of the vocabulary size
embedding_dim = 69 # either 32 or 64 as per research paper
batch_size = 4    # as per the research paper
num_classes = 2 # because sentiment analysis either positive or negative
hidden_size = 128 # it consist of 128 neurons
num_epochs = 5
num_layers = 1

class LSTMChar(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers):
        super(LSTMChar, self).__init__()
        self.input_size= input_size       # input size
        self.hidden_size = hidden_size    # hidden dimension
        self.num_classes = num_classes
        self.num_layers = num_layers


        self.encoder = nn.Embedding(input, embedding_dim)

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


train_dataset = Dataset_("/home/prabhu/LSTM/Data/train.csv", "/home/prabhu/LSTM/Data/classes.txt", 2500)
train_loader= DataLoader(train_dataset, shuffle= True, num_workers= 0, batch_size= batch_size)

test_dataset = Dataset_("/home/prabhu/LSTM/Data/test.csv", "/home/prabhu/LSTM/Data/classes.txt", 2500)
test_loader= DataLoader(train_dataset, shuffle= True, num_workers= 0, batch_size= batch_size)


model = LSTMChar(input_size=69,hidden_size=128, num_classes= 2, num_layers= 1)

use_gpu = torch.cuda.is_available()  # Determine if there is GPU acceleration
if use_gpu:
    model = model.cuda()
else:
    model.cpu()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_iter_epoch = len(train_loader)
best_acc = 0

for epoch in range(num_epochs):
    model.train()
    print('epoch {}'.format(epoch + 1))
    print('*' * 10)
    running_loss = 0.0
    running_acc = 0.0
    for step, (b_x, b_y) in enumerate(train_loader): # gives batch_size
        #print(b_x)
        #b_x = nn.view(batch_size,-1,)# reshpe x to (batch_size, step_size, input_size)

        # Forward propagation
        pred_label = model(b_x)
        prob_label = pred_label.cpu().data.numpy()
        loss = criterion(pred_label, b_y)
        loss.backward()
        optimizer.step()
        if (step + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, step + 1, num_iter_epoch, loss.item()))

    # test model
    model.eval()
    with torch.no_grad():
        correct = 0
        test_loss = 0
        predictions = []
        target_list = []
        for text, labels in test_loader:
            # images = images.view(-1, 28, 28)
            labels = labels
            outputs = model(text)
            test_loss += criterion(outputs, labels)  # to sum up batch loss
            pred = outputs.argmax(1)  # get the index of the max log-probability
            predictions.extend(pred)
            target_list.extend(labels)
        print('Test Accuracy of the model on the 25000 test data: {}'.format(
            metrics.accuracy_score(target_list, predictions)))
        print('Con_mat: {} \n'.format(metrics.confusion_matrix(target_list, predictions)))
    torch.save(model.state_dict(), 'imdb_model.ckpt')
