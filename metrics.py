import numpy as np

def accuracy(y_hat, y):
    return np.mean(y_hat == y)