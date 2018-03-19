import numpy as np
import h5py
import os
import sklearn.model_selection as skms


def load_data22_from(num):
    data_dir = '../project_datasets'
    data_name = 'A0' + str(num) + 'T_slice.mat'
    data_path = os.path.join(data_dir, data_name)
    data = h5py.File(data_path, 'r')
    x = np.copy(data['image'])
    x = x[:, 0:22, :]
    y = np.copy(data['type'])
    y = y[0, 0:x.shape[0]:1]
    y = np.asarray(y, dtype=np.int32)
    x, y = remove_zero(x, y)
    x_train, x_test, y_train, y_test = skms.train_test_split(x, y, test_size=50)

    return x_train, x_test, y_train, y_test


def load_data25_from(num):
    data_dir = '../project_datasets'
    data_name = 'A0' + str(num) + 'T_slice.mat'
    data_path = os.path.join(data_dir, data_name)
    data = h5py.File(data_path, 'r')
    x = np.copy(data['image'])
    y = np.copy(data['type'])
    y = y[0, 0:x.shape[0]:1]
    y = np.asarray(y, dtype=np.int32)
    x, y = remove_zero(x, y)
    x_train, x_test, y_train, y_test = skms.train_test_split(x, y, test_size=50)

    return x_train, x_test, y_train, y_test


def load_data22_from_all():
    x_train, x_test, y_train, y_test = load_data22_from(1)
    for i in range(2, 10):
        x_train_tmp, x_test_tmp, y_train_tmp, y_test_tmp = load_data22_from(i)
        x_train = np.concatenate((x_train, x_train_tmp), axis=0)
        x_test = np.concatenate((x_test, x_test_tmp), axis=0)
        y_train = np.concatenate((y_train, y_train_tmp), axis=0)
        y_test = np.concatenate((y_test, y_test_tmp), axis=0)
    return x_train, x_test, y_train, y_test


def load_data25_from_all():
    x_train, x_test, y_train, y_test = load_data25_from(1)
    for i in range(2, 10):
        x_train_tmp, x_test_tmp, y_train_tmp, y_test_tmp = load_data25_from(i)
        x_train = np.concatenate((x_train, x_train_tmp), axis=0)
        x_test = np.concatenate((x_test, x_test_tmp), axis=0)
        y_train = np.concatenate((y_train, y_train_tmp), axis=0)
        y_test = np.concatenate((y_test, y_test_tmp), axis=0)
    return x_train, x_test, y_train, y_test


def remove_zero(x, y):
    rows = np.unique(np.where(np.isnan(x))[0])
    x = np.delete(x, rows, 0)
    y = np.delete(y, rows, 0)
    return x, y


