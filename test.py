import toolUtils as Utl
import numpy as np
import cnn
import torch.optim as optim
import torch.nn as nn
import sklearn.model_selection as skms

print("Loading data...")
x_train, x_test, y_train, y_test = Utl.load_data25_from_all()

print("Training model...")
kf = skms.KFold(n_splits=5, random_state=123, shuffle=False)

for train_index, val_index in kf.split(x_train):
    print("TRAIN: ", train_index.shape, " VAL: ", val_index.shape)
    x_train_tmp, x_val_tmp = x_train[train_index], x_train[val_index]
    y_train_tmp, y_val_tmp = y_train[train_index], y_train[val_index]

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


