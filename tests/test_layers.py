import unittest
from layers import *
import numpy as np


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
        self.assertTrue(np.allclose(np.concatenate((np.zeros(6), np.arange(1, 5))), relu.backward(input_, grad_output)))
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
        self.assertTrue(np.allclose(input_ * lrelu.alpha, lrelu.backward(input_, grad_output)))

        # check input_ has not been changed
        self.assertTrue(np.allclose(input_, np.ones(10) * -1))

        # test positive and negative
        input_ = self.input_checker.create_argument(np.arange(-5, 5, dtype=np.float64))
        self.assertTrue(
            np.allclose(np.concatenate((np.arange(-5, 1) * lrelu.alpha, np.arange(1, 5))),
                        lrelu.backward(input_, grad_output)))
        self.input_checker.check_argument(input_)


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
