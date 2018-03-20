import toolUtils as Utl
from torch.autograd import Variable
import numpy as np
import cnn
import torch.optim as optim
import torch.nn as nn
import sklearn.model_selection as skms
import torch

print("===============Loading data...================")
x_train, x_test, y_train, y_test = Utl.load_data22_from_all()
print("x_train shape: ", x_train.shape)
print("x_test  shape: ", x_test.shape)
print("y_train shape: ", y_train.shape)
print("y_test  shape: ", y_test.shape)
print("===============Training model...==============")

net = cnn.Net()
net.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.01)

for epoch in range(1):  # loop over the dataset multiple times

    running_loss = 0.0
    for i in range(2000):
        inputs, labels = Utl.mini_batch(x_train, y_train, batch_size=200)
        labels = Utl.index_code(labels)
        inputs, labels = torch.from_numpy(inputs), torch.from_numpy(labels)
        inputs = torch.unsqueeze(inputs, 1)
        inputs = inputs.float()
        labels = labels.long()
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        optimizer.zero_grad()

        outputs = net(inputs)
        #print(outputs.shape)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]
        if i % 100 == 99:  # print every 2000 mini-batches
            inputs, labels = Utl.mini_batch(x_train, y_train, batch_size=200)
            labels = Utl.index_code(labels)
            inputs, labels = torch.from_numpy(inputs), torch.from_numpy(labels)
            inputs = torch.unsqueeze(inputs, 1)
            inputs = inputs.float()
            labels = labels.long()
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            outputs = net(inputs)
            _, y_pred = torch.max(outputs.data, 1)
            y_pred = y_pred.cpu().numpy()
            labels = labels.cpu().data.numpy()
            acc = (y_pred == labels).sum() / y_pred.shape[0]
            print('[%d, %5d] loss: %.3f, acc: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100, acc))
            running_loss = 0.0
print('Finished Training')

x_test, y_test = torch.from_numpy(x_test), torch.from_numpy(y_test)
x_test = torch.unsqueeze(x_test, 1)
y_test = Utl.index_code(y_test)
x_test = x_test.float()
x_test = Variable(x_test.cuda())

outputs_test = net(x_test)
_, y_pred = torch.max(outputs_test.data, 1)
y_pred = y_pred.cpu().numpy()
y_test = y_test.cpu().numpy()
acc = (y_pred == y_test).sum() / y_pred.shape[0]
print(acc)

