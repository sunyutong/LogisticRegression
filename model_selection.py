'''
This is a function which aims to split train and test datasets randomly.
According to parameter test_ratio, divide orginal data X and y into 
X_train, X_test, y_train, y_test.

'''

import numpy as np

def train_test_split(X, y, test_ratio=0.2, seed=None):
    
    assert X.shape[0] == y.shape[0], \
        "the size of X must be equal to the size of y"
    assert 0.0 <= test_ratio <= 1.0, \
        "test_ration must be valid"

    # Set a random seed for debug
    if seed:
        np.random.seed(seed)

    # Set random indexes
    shuffled_indexes = np.random.permutation(len(X))

    # According to the tset_ratio, split random index into training and testing indexes.
    test_size = int(len(X) * test_ratio)
    test_indexes = shuffled_indexes[:test_size]
    train_indexes = shuffled_indexes[test_size:]

    # According to the trainn_indexes and test_indexes, get X_train, y_train, X_test, y_test.
    X_train = X[train_indexes]
    y_train = y[train_indexes]

    X_test = X[test_indexes]
    y_test = y[test_indexes]

    return X_train, X_test, y_train, y_test