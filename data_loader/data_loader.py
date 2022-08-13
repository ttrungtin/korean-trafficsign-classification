import pickle
import numpy as np


def load_data(train_dir, valid_dir, test_dir):
    print("LOADING DATA...")
    with open(train_dir, mode='rb') as f:
        train = pickle.load(f)
    with open(valid_dir, 'rb') as f:
        valid = pickle.load(f)
    with open(test_dir, 'rb') as f:
        test = pickle.load(f)

    x_train, y_train = train['features'], train['labels']
    x_valid, y_valid = valid['features'], valid['labels']
    x_test, y_test = test['features'], test['labels']

    n_train = x_train.shape[0]
    n_valid = x_valid.shape[0]
    n_test = x_test.shape[0]
    image_shape = x_train[0].shape
    n_out = len(np.unique(y_train))

    print("Number of training examples: ", n_train)
    print("Number of valid examples: ", n_valid)
    print("Number of test examples: ", n_test)
    print("Input image shape: ", image_shape)
    print("Number of classes: ", n_out)

    return x_train, y_train, x_valid, y_valid, x_test, y_test


def load_data_each(data_dir):
    print("LOADING DATA...")
    with open(data_dir, 'rb') as f:
        data = pickle.load(f)

    x, y = data['features'], data['labels']

    x_shape = x.shape[0]
    n_out = len(np.unique(y))

    print("Number of examples: ", x_shape)
    print("Number of classes: ", n_out)

    return x, y
