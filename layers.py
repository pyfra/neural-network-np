import numpy as np
import initializers


class Layer:

    def __init__(self):
        pass

    def forward(self, X):
        return X

    def backward(self, X, grad_output):
        num_units = X.shape[1]
        d_layer_d_X = np.eye(num_units)
        return np.dot(grad_output, d_layer_d_X)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_regularizer(self, regularizer):
        self.regularizer = regularizer

    def get_regularization_cost(self):
        return 0


class LinearLayer(Layer):

    def forward(self, X):
        return X

    def backward(self, X, grad_output):
        return grad_output


class LeakyReLU(Layer):

    def __init__(self, alpha=1e-3):
        self.alpha = alpha

    def forward(self, X):
        return np.maximum(X, self.alpha * X)

    def backward(self, X, grad_output):
        grad = np.where(X > 0, 1, self.alpha)
        return grad_output * grad


class ReLU(LeakyReLU):

    def __init__(self):
        self.alpha = 0


class TanH(Layer):

    def forward(self, X):
        return np.tanh(X)

    def backward(self, X, grad_output):
        grad = 1 - np.tanh(X) ** 2
        return grad_output * grad


class ArcTan(Layer):

    def forward(self, X):
        return np.arctan(X)

    def backward(self, X, grad_output):
        grad = 1 / (X ** 2 + 1)
        return grad_output * grad


class SoftPlus(Layer):

    def forward(self, X):
        return np.log(1 + np.exp(X))

    def backward(self, X, grad_output):
        grad = 1 / (1 + np.exp(-X))
        return grad_output * grad


class ELU(ReLU):

    def forward(self, X):
        return np.maximum(X, self.alpha * (np.exp(X) - 1))

    def backward(self, X, grad_output):
        grad = np.where(X > 0, 1, self.alpha * np.exp(X))
        return grad_output * grad


class Sigmoid(Layer):

    def forward(self, X):
        return 1. / (1 + np.exp(-X))

    def backward(self, X, grad_output):
        sig = lambda x: 1. / (1 + np.exp(-x))
        grad = sig(X) * (1 - sig(X))
        return grad_output * grad


class Dense(Layer):

    def __init__(self, input_units, output_units, w_initializers=initializers.Xavier(),
                 biases_initializer=initializers.Zeros()):
        self.weights = w_initializers((input_units, output_units))
        self.biases = biases_initializer((output_units))
        self.optimizer = None
        self._trainable = True

    def forward(self, X):
        return X @ self.weights + self.biases

    def backward(self, X, grad_output):
        grad_X = np.dot(grad_output, self.weights.T)
        # compute gradient w.r.t. weights and biases
        grad_weights = np.dot(X.T, grad_output)
        grad_biases = np.sum(grad_output, axis=0)

        if self._trainable:
            assert grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape
            l2_adj = (self.regularizer.l2 * self.optimizer.learning_rate) / X.shape[0]
            if self.regularizer.l1:
                l1_adj = (self.regularizer.l1 * self.optimizer.learning_rate) * np.sign(self.weights) / X.shape[0]
            else:
                l1_adj = 0
            self.weights = (1 - l2_adj) * self.weights - l1_adj + self.optimizer.delta_params(grad_weights, "weights")
            self.biases = self.biases + self.optimizer.delta_params(grad_biases, "biases")

        return grad_X

    def get_regularization_cost(self):
        return self.regularizer(self.weights)


class Dropout(Layer):

    def __init__(self, p=.5):
        assert (p >= 0) & (p <= 1), "invalid number for p (%4.f), please enter a probability between 0 and 1" % p
        self.p = p

    def forward(self, X, seed=None):
        if not (seed is None):
            np.random.seed(seed)
        self.mask = np.random.binomial(1, self.p, size=X.shape) / self.p
        return X * self.mask

    def backward(self, X, grad_output):
        return grad_output * self.mask


class BatchReguralization(Layer):
    """
    Reference paper: https://arxiv.org/abs/1502.03167
    """

    def __init__(self, input_units, gamma_initializer=initializers.RandomNormal(),
                 beta_initializer=initializers.RandomNormal(),
                 eps=1e-5):
        self.gamma = gamma_initializer(shape=(1, input_units))
        self.beta = beta_initializer(shape=(1, input_units))
        self.eps = eps

    def forward(self, X):
        if len(X.shape) == 2:
            self.mean = np.mean(X, axis=0)  # batch mean
            var = np.mean((X - self.mean) ** 2, axis=0)  # batch variance
            self.var = np.sqrt(var + self.eps)
            self.x_hat = (X - self.mean) / self.var  # normalization
            output = self.gamma * self.x_hat + self.beta  # scale and shift
        else:
            raise NotImplementedError("Batch reguralization does not currently support tensors > 2 dimensions")
        return output

    def backward(self, X, grad_output):
        m = len(X)
        dxhat = grad_output * self.gamma
        dx = (1. / m) * 1 / self.var * (m * dxhat - np.sum(dxhat, axis=0)
                                        - self.x_hat * np.sum(dxhat * self.x_hat, axis=0))
        self.beta -= np.sum(grad_output, axis=0)
        self.gamma -= np.sum(self.x_hat * grad_output, axis=0)
        return dx


class RNN(Layer):

    def __init__(self, input_units, hidden_units, output_units, w_initializers=initializers.Xavier(),
                 recursive_w_initializers=initializers.Xavier(), ):
        # how many sets of biases??
        self.w_x = w_initializers((input_units, hidden_units))  # input to hidden
        self.w_h = recursive_w_initializers((hidden_units, hidden_units))  # hidden to hidden
        self.w_y = w_initializers((hidden_units, output_units))  # hidden to output
        self.optimizer = None
        self._trainable = True

    def forward(self, X):
        t = len(X)
        self.s = np.zeros((t + 1, len(self.w_h)))
        output = np.zeros((t, self.w_y.shape[1]))
        self.s[0] = np.zeros(len(self.w_h))  # first hidden set to zero
        for k in np.arange(t):
            self.s[k + 1] = X[k] @ self.w_x + self.s[k] @ self.w_h
            output[k] = self.s[k + 1] @ self.w_y
        return output

    def backward(self, X, grad_output):
        pass
