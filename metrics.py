import numpy as np


class Metrics:

    def __call__(self, *args, **kwargs):
        return 0


class Accuracy(Metrics):

    def __call__(self, y_hat, y):
        assert y_hat.shape[0] == y.shape[0]
        if len(y_hat.shape) > 1 and len(y.shape) == 1:  # we can only go from n to 1
            return np.mean(y_hat.argmax(axis=1) == y)
        else:
            raise NotImplementedError('The reduction can only be done to dimension 1 and not to %d' % y.shape[1])
        return np.mean(y_hat == y)
