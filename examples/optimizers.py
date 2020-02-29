class Optimizer:

    def delta_params(self):
        raise NotImplementedError()


class SGD(Optimizer):

    def __init__(self, learning_rate, momentum, nesterov):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.gradient_w = list()
        self.gradient_b = list()

    def delta_params(self, grad, params_name='weights'):
        if params_name == 'weights':
            self.gradient_w.append(grad)
        else:
            self.gradient_b.append(grad)

        return -self.learning_rate * grad
