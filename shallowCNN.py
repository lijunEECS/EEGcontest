import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 40, (22, 15))
        self.ELU = nn.ELU()
        self.BN = nn.BatchNorm2d(num_features=1)
        self.pool1 = nn.AvgPool2d((1, 75), stride=(1, 15))
        self.fc1 = nn.Linear(40*61, 50)
        self.fc2 = nn.Linear(50, 4)
        #self.fc3 = nn.Linear(184, 4)

    def forward(self, x):
        #print("C0: ", x.shape)
        x = self.conv1(x)
        x = x.permute(0, 2, 1, 3)
        x = self.BN(x)
        x = self.ELU(x)
        x = self.pool1(x)
        #print("C1: ", x.shape)
        x = x.view(x.data.shape[0], -1)
        x = F.relu(self.fc1(x))
        #print("C2: ", x.shape)
        x = self.fc2(x)
        #print("C3: ", x.shape)
        return x
