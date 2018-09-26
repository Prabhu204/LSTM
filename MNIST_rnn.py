"""
author: Prabhu

"""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
from sklearn import metrics



batch_size = 64
learning_rate = 0.01
num_epochs = 5
time_step = 28

train_dataset = datasets.MNIST(
    root='./Data', train=True, transform=transforms.ToTensor(), download=True)


# plot train data example
print(train_dataset.train_data.size()) # (60000, 28, 28 )
print(train_dataset.train_labels.size()) # (60000)
plt.imshow(train_dataset.train_data[59999].numpy(), cmap='gray')
plt.title('%i' %train_dataset.train_labels[0])
plt.show()

test_dataset = datasets.MNIST(
    root='./Data', train=False, transform=transforms.ToTensor())

test_x = test_dataset.test_data.type(torch.FloatTensor)
test_y = test_dataset.test_labels.numpy().squeeze()


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Recurrent Network
class Rnn(nn.Module):
    def __init__(self):
        super(Rnn, self).__init__()
        self.lstm = nn.LSTM(input_size = 28, hidden_size = 64,num_layers= 1, batch_first=True)
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        # h0 = Variable(torch.zeros(self.n_layer, x.size(1),
        #   self.hidden_dim)).cuda()
        # c0 = Variable(torch.zeros(self.n_layer, x.size(1),
        #   self.hidden_dim)).cuda()
        out, (h_n, h_c)= self.lstm(x, None) # none represents zero intial-state
        # choose r_out at last time step
        out = self.classifier(out[:, -1, :])
        return out


model = Rnn()  # Image size is 28x28
use_gpu = torch.cuda.is_available()  # Determine if there is GPU acceleration
if use_gpu:
    model = model.cuda()
# define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
total_step = len(train_loader)
for epoch in range(num_epochs):
    model.train()
    print('epoch {}'.format(epoch + 1))
    print('*' * 10)
    running_loss = 0.0
    running_acc = 0.0
    for step, (b_x, b_y) in enumerate(train_loader): # gives batch_size
        b_x = b_x.view(-1, 28,28) # reshpe x to (batch_size, step_size, input_size)
        # Forward propagation
        out = model(b_x)
        loss = criterion(out, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, step + 1, total_step, loss.item()))

    # test model
    model.eval()
    with torch.no_grad():
        correct = 0
        test_loss = 0
        predictions= []
        target_list= []
        for images, labels in test_loader:
            images = images.view(-1, 28,28)
            labels = labels
            outputs = model(images)
            test_loss += criterion(outputs, labels)  # to sum up batch loss
            pred = outputs.argmax(1)  # get the index of the max log-probability
            predictions.extend(pred)
            target_list.extend(labels)
        print('Test Accuracy of the model on the 10000 test images: {}'.format(metrics.accuracy_score(target_list,predictions)))
        print('Con_mat: {}'.format(metrics.confusion_matrix(target_list, predictions)))
    torch.save(model.state_dict(), 'model.ckpt')



