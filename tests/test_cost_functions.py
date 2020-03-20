import unittest
import cost_functions as cf
import numpy as np
from tests.testing_utilities import eval_numerical_gradient


class TestGradientCostFunctions(unittest.TestCase):

    def test_gradient_all(self):

        y_hat = np.linspace(-1, 1, 500).reshape([50, 10])
        y = np.arange(50) % 10

        all_cf = dict([(name, cls) for name, cls in cf.__dict__.items() if isinstance(cls, type)])
        for name, cost_function in all_cf.items():
            if name != 'CostFunction':
                cost_f = cost_function()
                numerical_grad = eval_numerical_gradient(lambda x: cost_f(x, y).mean(), x=y_hat)
                grads = cost_f.grad(y_hat, y)
                self.assertTrue(np.allclose(grads, numerical_grad, rtol=1e-3, atol=0),
                                'numerical grads problem for layer %s' % name)
