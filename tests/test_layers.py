import unittest
from layers import *
import numpy as np
from tests.testing_utilities import eval_numerical_gradient, eval_numerical_gradient_array
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
            elif layer in [BatchReguralization, RNN]:
                continue
            else:
                init_layer = layer()
            if layer == Dropout:
                numerical_grad = eval_numerical_gradient(lambda x: init_layer.forward(x, seed=1).mean(), x=input_)
            else:
                numerical_grad = eval_numerical_gradient(lambda x: init_layer.forward(x).mean(), x=input_)
            grads = init_layer.backward(input_, grad_out)
            self.assertTrue(np.allclose(grads, numerical_grad, rtol=1e-3, atol=0),
                            'numerical grads problem for layer %s' % name)


class TestLayersForward(unittest.TestCase):

    def test_dense(self):
        layer = Dense(3, 4)
        x = np.linspace(-1, 1, 2 * 3).reshape([2, 3])
        layer.weights = np.linspace(-1, 1, 3 * 4).reshape([3, 4])
        layer.biases = np.linspace(-1, 1, 4)
        self.assertTrue(np.allclose(layer.forward(x), np.array([[0.07272727, 0.41212121, 0.75151515, 1.09090909],
                                                                [-0.90909091, 0.08484848, 1.07878788, 2.07272727]])))


class TestRNN(unittest.TestCase):

    def test_forward_dimesion(self):
        X = np.ones((10, 5)) * 2
        layer = self._build_rnn(5, 3, 1)
        output = layer.forward(X)

        # check dimension
        self.assertTrue(output.shape[1] == 1)  # output is only one
        self.assertTrue(output.shape[0] == 10)  # one for each time step

        self.assertTrue(layer.s.shape[0] == 10 + 1)  # inizialed with state 0
        self.assertTrue(layer.s.shape[1] == 3)

    def test_forward_computation(self):
        X = np.ones((1, 5)) * 2
        layer = self._build_rnn(5, 3, 1)
        output = layer.forward(X)
        # check intermediate steps
        target_s = np.vstack([np.zeros(3), np.ones(3) * 10])
        self.assertTrue(np.allclose(target_s, layer.s))
        self.assertTrue(np.allclose(np.array([30.0]), output))

    def test_forward_computation2(self):
        X = np.ones((2, 5)) * 2
        layer = self._build_rnn(5, 3, 1)
        output = layer.forward(X)
        # check intermediate steps
        target_s = np.vstack([np.zeros(3), np.ones(3) * 10, np.ones(3) * (10 + 30)])
        self.assertTrue(np.allclose(target_s, layer.s))
        self.assertTrue(np.allclose(np.array([[30.0], [120.0]]), output))

    def _build_rnn(self, input, hidden, output):
        layer = RNN(input, hidden, output)

        # set all the weights to 1
        layer.w_x = np.ones((input, hidden))  # input to hidden
        layer.w_h = np.ones((hidden, hidden))  # hidden to hidden
        layer.w_y = np.ones((hidden, output))

        return layer


class TestLayersBatchReguralization(unittest.TestCase):

    def test_forward(self):
        layer = BatchReguralization(20)
        input_ = np.random.randn(10, 20) * 2 + 10
        layer.gamma = np.ones(20)
        layer.beta = np.zeros(20)
        new_input = layer.forward(input_)
        means = np.mean(new_input, axis=0)
        stds = np.std(new_input, axis=0)

        self.assertTrue(len(means) == 20)
        self.assertTrue(len(stds) == 20)
        self.assertTrue(np.allclose(means, np.zeros_like(means)))
        self.assertTrue(np.allclose(stds, np.ones_like(stds)))

    def test_forward2(self):
        N, D = 4, 5
        layer = BatchReguralization(5)
        np.random.seed(10)
        input_ = 5 * np.random.randn(N, D) + 12
        gamma = np.random.randn(D)
        beta = np.random.randn(D)
        layer.gamma = gamma
        layer.beta = beta
        result = layer.forward(input_)

        target_array = np.array([[-0.93152358, -0.80663908, 1.03557909, -2.60471575, 1.89923477],
                                 [4.63876261, 0.09174836, 1.64804102, -2.55817853, 0.48238331],
                                 [1.50806616, -1.78091162, 1.25047864, 1.20136783, 1.20017641],
                                 [1.47518366, 2.89239921, 1.65788676, 2.87653451, -1.12897775]])

        self.assertTrue(np.allclose(target_array, result))

    def test_backward(self):
        np.random.seed(10)
        N, D = 4, 5
        layer = BatchReguralization(5)
        input_ = 5 * np.random.randn(N, D) + 12
        gamma = np.random.randn(D)
        beta = np.random.randn(D)
        dout = np.random.randn(N, D)

        layer.gamma = gamma
        layer.beta = beta

        numerical_grad = eval_numerical_gradient_array(lambda x: layer.forward(x), x=input_, df=dout)
        gradient = layer.backward(input_, dout)

        self.assertTrue(np.allclose(numerical_grad, gradient))


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
