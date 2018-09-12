"""
author: Prabhu

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(2)

lstm = nn.LSTM(3,3) # input dim is 3, output dim is 3

inputs = [torch.randn(1,3) for _ in range(5)] # make a sequence of length 5
print(inputs)

# initialize the hidden state
hidden = (torch.randn(1,1, 3), torch.randn(1,1,3))
for i in inputs:
    print (i.size())
    # Step through the sequence one element at a time
    # after each step, hidden contain th hidden state.
    out, hidden = lstm(i.view(1,1, -1), hidden)

inputs = torch.cat(inputs).view(len(inputs),1,-1)
hidden = (torch.randn(1,1,3), torch.randn(1,1,3))
out, hidden = lstm(inputs,hidden)





