import numpy as np


class Regularizer:

    def __call__(self, *args, **kwargs):
        """
        It computes the delta for the cost function coming from regularization
        """
        return 0


class L1L2(Regularizer):

    def __init__(self, l1=0, l2=0):
        self.l1 = l1
        self.l2 = l2

    def __call__(self, w, *args, **kwargs):
        regularization_cost = 0
        """
        Tecnichally we could not check if self.l1 is different from 0, and just write something like
        return  self.l1 * np.sum(np.abs(w)) + self.l2 * np.sum(np.square(w)), 
        but  np.sum(np.abs(w)) and np.sum(np.square(w)) are quite expensive operations we do not want to run unless
        we have to

        """
        if self.l1:
            regularization_cost += self.l1 * np.sum(np.abs(w))

        if self.l2:
            regularization_cost += self.l2 * np.sum(np.square(w))

        return regularization_cost


class L1(L1L2):

    def __init__(self, l1):
        super(L1, self).__init__(l1=l1)


class L2(L1L2):

    def __init__(self, l2):
        super(L2, self).__init__(l2=l2)
