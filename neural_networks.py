import numpy as np


class ANN:

    def __init__(self):
        self._layers = []
        self.cost_function = None
        self.optimizer = None
        self.metrics = None

    def add(self, layer):
        self.layer.append(layer)

    def forward(self, X):
        activations = []
        input_ = X
        for layer in self._layers:
            input_ = layer.forward(input_)
            activations.append(input_)

        assert len(activations) == len(self)
        return activations

    def train(self, X, y):
        # Get the layer activations
        layer_activations = self.forward(X)
        layer_inputs = [X] + layer_activations  # layer_input[i] is an input for network[i]
        logits = layer_activations[-1]

        # Compute the loss and the initial gradient
        loss = self.cost_function.compute(logits, y)
        loss_grad = self.cost_function.grad(logits, y)

        grad_output = loss_grad
        layer_inputs = layer_inputs[:-1]
        for input_, layer in zip(layer_inputs[::-1], self._layers[::-1]):
            grad_output = layer.backward(input_, grad_output)

        return np.mean(loss)

    def compile(self, cost_function, optimizer, metrics):
        self.cost_function = cost_function
        self.optimizer = optimizer
        self.metrics = metrics

    def __len__(self):
        return len(self._layers)
