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

    def set_regularizer(self, regularizer):
        self.regularizer = regularizer

    def get_regularization_cost(self):
        return 0


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

    def __init__(self, input_units, output_units, w_initializers=initializers.Xavier(),
                 biases_initializer=initializers.Zeros()):
        self.weights = w_initializers((input_units, output_units))
        self.biases = biases_initializer((output_units))
        self.optimizer = None
        self._trainable = True

    def forward(self, input):
        return input @ self.weights + self.biases

    def backward(self, input, grad_output):
        grad_input = np.dot(grad_output, self.weights.T)
        # compute gradient w.r.t. weights and biases
        grad_weights = np.dot(input.T, grad_output)
        grad_biases = np.sum(grad_output, axis=0)

        if self._trainable:
            assert grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape
            l2_adj = (self.regularizer.l2 * self.optimizer.learning_rate) / input.shape[0]
            if self.regularizer.l1:
                l1_adj = (self.regularizer.l1 * self.optimizer.learning_rate) * np.sign(self.weights) / input.shape[0]
            else:
                l1_adj = 0
            self.weights = (1 - l2_adj) * self.weights - l1_adj + self.optimizer.delta_params(grad_weights, "weights")
            self.biases = self.biases + self.optimizer.delta_params(grad_biases, "biases")

        return grad_input

    def get_regularization_cost(self):
        return self.regularizer(self.weights)


class Dropout(Layer):

    def __init__(self, p=.5):
        assert (p >= 0) & (p <= 1), "invalid number for p (%4.f), please enter a probability between 0 and 1" % p
        self.p = p

    def forward(self, input_, seed=None):
        if not (seed is None):
            np.random.seed(seed)
        self.mask = np.random.binomial(1, self.p, size=input_.shape) / self.p
        return input_ * self.mask

    def backward(self, input_, grad_output):
        return grad_output * self.mask


class BatchReguralization(Layer):
    """
    Reference paper: https://arxiv.org/abs/1502.03167
    """

    def __init__(self, input_units, gamma_initializer=initializers.Ones(), beta_initializer=initializers.Zeros()):
        self.gamma = gamma_initializer(shape=(1, input_units))
        self.beta = beta_initializer(shape=(1, input_units))

    def forward(self, input):
        if len(input.shape) == 2:
            mean = np.mean(input, axis=0)  # batch mean
            self.var = np.mean((input - mean) ** 2, axis=0)  # batch variance
            self.x_hat = (input - mean) / np.sqrt(self.var + 1e-5)  # normalization
            output = self.gamma * self.x_hat + self.beta  # scale and shift
        else:
            raise NotImplementedError("Batch reguralization does not currently support tensors > 2 dimensions")
        return output

    def backward(self, input_, grad_output):

        # TODO check error test gradient
        m = len(input_)
        dxhat = grad_output * self.gamma
        dx = (1. / m) * 1 / self.var * (m * dxhat - np.sum(dxhat, axis=0)
                                        - self.x_hat * np.sum(dxhat * self.x_hat, axis=0))
        self.beta -= np.sum(grad_output, axis=0)
        self.gamma -= np.sum(self.x_hat * grad_output, axis=0)

        return dx * grad_output
