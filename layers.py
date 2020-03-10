import numpy as np
import initializers


class Layer:

    def __init__(self):
        pass

    def forward(self, input):
        return input

    def backward(self, input, grad_output):
        num_units = input.shape[1]
        d_layer_d_input = np.eye(num_units)
        return np.dot(grad_output, d_layer_d_input)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer


class LeakyReLU(Layer):

    def __init__(self, alpha=1e-3):
        self.alpha = alpha

    def forward(self, input_):
        return np.maximum(input_, self.alpha * input_)

    def backward(self, input_, grad_output):
        grad = np.where(input_ > 0, 1, self.alpha)
        return grad_output * grad


class ReLU(LeakyReLU):

    def __init__(self):
        self.alpha = 0


class TanH(Layer):

    def forward(self, input_):
        return np.tanh(input_)

    def backward(self, input_, grad_output):
        grad = 1 - np.tanh(input_) ** 2
        return grad_output * grad


class ArcTan(Layer):

    def forward(self, input_):
        return np.arctan(input_)

    def backward(self, input_, grad_output):
        grad = 1 / (input_ ** 2 + 1)
        return grad_output * grad


class SoftPlus(Layer):

    def forward(self, input_):
        return np.log(1 + np.exp(input_))

    def backward(self, input_, grad_output):
        grad = 1 / (1 + np.exp(-input_))
        return grad_output * grad


class ELU(ReLU):

    def forward(self, input_):
        return np.maximum(input_, self.alpha * (np.exp(input_) - 1))

    def backward(self, input_, grad_output):
        grad = np.where(input_ > 0, 1, self.alpha * np.exp(input_))
        return grad_output * grad


class Sigmoid(Layer):

    def forward(self, input_):
        return 1. / (1 + np.exp(-input_))

    def backward(self, input_, grad_output):
        sig = lambda x: 1. / (1 + np.exp(-x))
        grad = sig(input_) * (1 - sig(input_))
        return grad_output * grad


class Dense(Layer):

    def __init__(self, input_units, output_units, learning_rate=0.1, w_initializers=initializers.Xavier(),
                 biases_initializer=initializers.Zeros()):
        self.learning_rate = learning_rate
        self.weights = w_initializers((input_units, output_units))
        self.biases = biases_initializer((output_units))
        self.optimizer = None

    def forward(self, input):
        return input @ self.weights + self.biases

    def backward(self, input, grad_output):
        grad_input = np.dot(grad_output, self.weights.T)
        # compute gradient w.r.t. weights and biases
        grad_weights = np.dot(input.T, grad_output)
        grad_biases = np.sum(grad_output, axis=0)

        assert grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape
        self.weights = self.weights + self.optimizer.delta_params(grad_weights, "weights")
        self.biases = self.biases + self.optimizer.delta_params(grad_biases, "biases")

        return grad_input


class Dropout(Layer):

    def __init__(self, p=.5):
        assert (p >= 0) & (p <= 1), "invalid number for p (%4.f), please enter a probability between 0 and 1" % p
        self.p = p

    def forward(self, input_):
        self.mask = np.random.binomial(1, self.p, size=input_.shape) / self.p
        return input_ * self.mask

    def backward(self, input_, grad_output):
        return grad_output * self.mask * self.p
