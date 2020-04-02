import numpy as np


class Optimizer:

    def delta_params(self):
        raise NotImplementedError()


class Momentum(Optimizer):
    
    def __init__(self, learning_rate=0.1, momentum=.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.gradient_w = list()
        self.gradient_b = list()
        self.accumulated_grads = dict()
        self.params_grads = dict()

    def delta_params(self, grad, params_name='weights'):
        self._save_grad(grad, params_name)
        self._apply_momentum(params_name, grad)
        return self.accumulated_grads[params_name]

    def _apply_momentum(self, params_name, grad):
        if not params_name in self.accumulated_grads:
            accumulated_grad = np.zeros_like(grad)
        else:
            accumulated_grad = self.accumulated_grads[params_name]
        accumulated_grad = accumulated_grad * self.momentum - (
                1 - self.momentum) * self.learning_rate * grad
        self.accumulated_grads[params_name] = accumulated_grad

    def _save_grad(self, grad, params_name):
        if not params_name in self.params_grads:
            self.params_grads[params_name] = [grad]
        else:
            self.params_grads[params_name].append(grad)


class NesterovMomentum(Momentum):

    def _apply_momentum(self, accumulated_grad, grad):
        if accumulated_grad is None:
            accumulated_grad = np.zeros_like(grad)
        accumulated_grad = accumulated_grad * self.momentum - (
                1 - self.momentum) * self.learning_rate * grad
        return accumulated_grad
