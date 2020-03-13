import unittest
from layers import *
import numpy as np
from tests.testing_utilities import eval_numerical_gradient
import layers


class TestActivations(unittest.TestCase):

    def setUp(self) -> None:
        self.input_checker = InputChecker()

    def test_relu_forward(self):
        relu = ReLU()
        self.assertEqual(0, relu.alpha)

        # test all positive
        input_ = self.input_checker.create_argument(np.ones(10))
        self.assertTrue(np.allclose(input_, relu.forward(input_)))
        self.input_checker.check_argument(input_)

        # test all negative
        input_ = self.input_checker.create_argument(np.ones(10) * -1)
        self.assertTrue(np.allclose(np.zeros(10), relu.forward(input_)))
        self.input_checker.check_argument(input_)

        # test positive and negative
        input_ = self.input_checker.create_argument(np.arange(-5, 5))
        self.assertTrue(np.allclose(np.concatenate((np.zeros(6), np.arange(1, 5))), relu.forward(input_)))
        self.input_checker.check_argument(input_)

    def test_relu_backward(self):
        grad_output = np.ones(10)
        relu = ReLU()

        # test all positive
        input_ = np.ones(10)
        self.assertTrue(np.allclose(input_, relu.backward(input_, grad_output)))

        # test all negative
        input_ = np.ones(10) * -1
        self.assertTrue(np.allclose(np.zeros(10), relu.backward(input_, grad_output)))

        # test positive and negative
        input_ = self.input_checker.create_argument(np.arange(-5, 5))
        self.assertTrue(np.allclose(np.concatenate((np.zeros(6), np.ones(4))), relu.backward(input_, grad_output)))
        self.input_checker.check_argument(input_)

    def test_leakyrelu_forward(self):
        lrelu = LeakyReLU(.2)
        self.assertEqual(0.2, lrelu.alpha)

        # test all positive
        input_ = np.ones(10)
        self.assertTrue(np.allclose(input_, lrelu.forward(input_)))

        # test all negative
        input_ = np.ones(10) * -1
        self.assertTrue(np.allclose(input_ * lrelu.alpha, lrelu.forward(input_)))

        # test positive and negative
        input_ = self.input_checker.create_argument(np.arange(-5, 5))
        self.assertTrue(
            np.allclose(np.concatenate((np.arange(-5, 1) * lrelu.alpha, np.arange(1, 5))), lrelu.forward(input_)))
        self.input_checker.check_argument(input_)

    def test_leakyrelu_backward(self):
        lrelu = LeakyReLU(.2)
        grad_output = np.ones(10)

        # test all positive
        input_ = np.ones(10)
        self.assertTrue(np.allclose(input_, lrelu.backward(input_, grad_output)))

        # test all negative
        input_ = np.ones(10) * -1
        self.assertTrue(np.allclose(np.ones(10) * lrelu.alpha, lrelu.backward(input_, grad_output)))

        # check input_ has not been changed
        self.assertTrue(np.allclose(input_, np.ones(10) * -1))

        # test positive and negative
        input_ = self.input_checker.create_argument(np.arange(-5, 5, dtype=np.float64))
        self.assertTrue(
            np.allclose(np.concatenate((np.ones(6) * lrelu.alpha, np.ones(4))),
                        lrelu.backward(input_, grad_output)))
        self.input_checker.check_argument(input_)


class TestGradient(unittest.TestCase):

    def test_gradient_all(self):
        input_ = np.linspace(-1, 1, 10 * 32).reshape([10, 32])
        grad_out = np.ones([10, 32]) / (32 * 10)
        all_layers = dict([(name, cls) for name, cls in layers.__dict__.items() if isinstance(cls, type)])
        for name, layer in all_layers.items():
            if layer == Dense:
                init_layer = layer(input_.shape[1], input_.shape[1])
                init_layer._trainable = False
            else:
                init_layer = layer()
            if layer == Dropout:
                numerical_grad = eval_numerical_gradient(lambda x: init_layer.forward(x, seed=1).mean(), x=input_)
            else:
                numerical_grad = eval_numerical_gradient(lambda x: init_layer.forward(x).mean(), x=input_)
            grads = init_layer.backward(input_, grad_out)
            self.assertTrue(np.allclose(grads, numerical_grad, rtol=1e-3, atol=0),
                            'numerical grads problem for layer %s' % name)


class InputChecker:

    def __init__(self):
        self.argument = None

    def create_argument(self, argument):
        self.argument = argument
        return argument

    def check_argument(self, argument_after_function):
        assert np.allclose(self.argument, argument_after_function), "the argument has been changed after the function"


if __name__ == '__main__':
    unittest.main()
