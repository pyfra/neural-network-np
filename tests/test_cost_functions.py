import unittest
import cost_functions as cf
import numpy as np
from tests.testing_utilities import eval_numerical_gradient

CATEGORICAL = [cf.SoftmaxCrossEntropy]
REGRESSION = [cf.MSE]


class TestCostFunctions(unittest.TestCase):

    def setUp(self) -> None:
        """
        It defines the input for the tests. Assuming a Batch size of 50 and output of 10.
        :return:
        """
        self.y_hat = dict()
        self.y_hat['categorical'] = np.linspace(-1, 1, 500).reshape([50, 10])
        self.y_hat['regression'] = np.linspace(-1, 1, 10).reshape(-1, 1)
        self.y = dict()
        self.y['categorical'] = np.arange(50) % 10
        self.y['regression'] = np.arange(10).reshape(-1, 1)

    def test_mse(self):
        mse = cf.MSE()
        y = np.array([1, 9, 2, -5, -2, 6]).reshape(2, 3)
        y_hat = np.array([4, 8, 12, 8, 1, 3]).reshape(2, 3)
        computed_cost = mse(y_hat, y).mean()
        assert np.isclose(computed_cost, 49.5, atol=1e-3)

    def test_gradient_all(self):
        all_cf = dict([(name, cls) for name, cls in cf.__dict__.items() if isinstance(cls, type)])
        for name, cost_function in all_cf.items():
            if name != 'CostFunction':
                cost_f = cost_function()
                if cost_function in CATEGORICAL:
                    actual_y = self.y['categorical']
                    y_hat = self.y_hat['categorical']
                else:
                    actual_y = self.y['regression']
                    y_hat = self.y_hat['regression']

                numerical_grad = eval_numerical_gradient(lambda x: cost_f(x, actual_y).mean(), x=y_hat)
                grads = cost_f.grad(y_hat, actual_y)
                self.assertTrue(np.allclose(grads, numerical_grad, rtol=1e-3, atol=0),
                                'numerical grads problem for layer %s' % name)
