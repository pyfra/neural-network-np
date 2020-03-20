import numpy as np


class CostFunction:

    def __call__(self, y_hat, y):
        return 0

    def grad(self, y_hat, y):
        return 0


class SoftmaxCrossEntropy(CostFunction):

    def __call__(self, y_hat, y):
        logits_for_answers = y_hat[np.arange(len(y_hat)), y]
        xentropy = - logits_for_answers + np.log(np.sum(np.exp(y_hat), axis=-1))
        return xentropy

    def grad(self, y_hat, y):
        ones_for_answers = np.zeros_like(y_hat)
        ones_for_answers[np.arange(len(y_hat)), y] = 1
        softmax = np.exp(y_hat) / np.exp(y_hat).sum(axis=-1, keepdims=True)
        return (- ones_for_answers + softmax) / y_hat.shape[0]


class MSE(CostFunction):

    def __call__(self, y_hat, y):
        return (y - y_hat) ** 2

    def grad(self, y_hat, y):
        return -2 * (y - y_hat) / y_hat.shape[0]
