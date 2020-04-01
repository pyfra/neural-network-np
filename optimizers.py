import numpy as np


class Optimizer:

    def delta_params(self):
        raise NotImplementedError()


class Momentum(Optimizer):
    # TODO fix optimizers to save params with different names and to use the name to access gradient information
    def __init__(self, learning_rate=0.1, momentum=.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.gradient_w = list()
        self.gradient_b = list()
        self.accumulated_grad_weights = None
        self.accumulated_grad_biases = None

    def delta_params(self, grad, params_name='weights'):
        if params_name == 'weights':
            self.gradient_w.append(grad)
            self.accumulated_grad_weights = self._apply_momentum(self.accumulated_grad_weights, grad)
            return self.accumulated_grad_weights
        else:
            self.accumulated_grad_biases = self._apply_momentum(self.accumulated_grad_biases, grad)
            return self.accumulated_grad_biases

    def _apply_momentum(self, accumulated_grad, grad):
        if accumulated_grad is None:
            accumulated_grad = np.zeros_like(grad)
        accumulated_grad = accumulated_grad * self.momentum - (
                1 - self.momentum) * self.learning_rate * grad
        return accumulated_grad


class NesterovMomentum(Momentum):

    def _apply_momentum(self, accumulated_grad, grad):
        if accumulated_grad is None:
            accumulated_grad = np.zeros_like(grad)
        accumulated_grad = accumulated_grad * self.momentum - (
                1 - self.momentum) * self.learning_rate * grad
        return accumulated_grad