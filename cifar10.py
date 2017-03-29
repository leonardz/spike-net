import pickle
import numpy as np


with open('cifar-10-batches-py/batches.meta', 'rb') as f:
    label_names = pickle.load(f, encoding='latin1')['label_names']

def get_train_data():
    """
    Returns all cifar train images and labels one hot encoded
    """

    with open('cifar-10-batches-py/data_batch_1', 'rb') as f:
        dict1 = pickle.load(f, encoding='latin1')

    with open('cifar-10-batches-py/data_batch_2', 'rb') as f:
        dict2 = pickle.load(f, encoding='latin1')

    with open('cifar-10-batches-py/data_batch_3', 'rb') as f:
        dict3 = pickle.load(f, encoding='latin1')

    with open('cifar-10-batches-py/data_batch_4', 'rb') as f:
        dict4 = pickle.load(f, encoding='latin1')

    with open('cifar-10-batches-py/data_batch_5', 'rb') as f:
        dict5 = pickle.load(f, encoding='latin1')

    num_classes = 10

    x = np.concatenate((dict1['data'], dict2['data'], dict3['data'], dict4['data'], dict5['data']))

    x_images = x.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
    x_flipped = np.flip(x_images, axis=2)

    x = np.concatenate((x_images, x_flipped))

    y = np.concatenate((dict1['labels'], dict2['labels'], dict3['labels'], dict4['labels'], dict5['labels']))

    y_one_hot = np.zeros((len(y), num_classes))
    y_one_hot[np.arange(len(y)), y] = 1

    y = np.concatenate((y_one_hot, y_one_hot))

    return x, y


def get_test_data():
    with open('cifar-10-batches-py/test_batch', 'rb') as f:
        dict1 = pickle.load(f, encoding='latin1')

    num_classes = 10

    x = dict1['data'].reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
    y = dict1['labels']
    y_one_hot = np.zeros((len(y), num_classes))
    y_one_hot[np.arange(len(y)), y] = 1

    return x, y_one_hot
