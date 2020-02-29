import numpy as np
from initializers import *


class Layer:

    def __init__(self):
        pass

    def forward(self, input):
        return input

    def backward(self, input, grad_output):
        num_units = input.shape[1]
        d_layer_d_input = np.eye(num_units)
        return np.dot(grad_output, d_layer_d_input)  # chain rule


class ReLU(Layer):

    def __init__(self):
        pass

    def forward(self, input):
        return np.maximum(input, 0)

    def backward(self, input, grad_output):
        relu_grad = input > 0
        return grad_output * relu_grad


class Dense(Layer):

    def __init__(self, input_units, output_units, learning_rate=0.1, w_initializers=Xavier(),
                 biases_initializer=Zeros()):
        self.learning_rate = learning_rate
        self.weights = w_initializers(input_units, output_units)
        self.biases = biases_initializer(input_units, output_units)
        self.optimizer = None

    def forward(self, input):
        return input @ self.weights + self.biases

    def backward(self, input, grad_output):
        grad_input = np.dot(grad_output, self.weights.T)
        # compute gradient w.r.t. weights and biases
        grad_weights = np.dot(input.T, grad_output)
        grad_biases = np.sum(grad_output, axis=0)

        assert grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape
        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases

        return grad_input

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer


class Dropout(Layer):
    pass
