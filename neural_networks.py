import numpy as np


class ANN:

    def __init__(self):
        self._layers = []
        self.cost_function = None

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

    def train(self, X, y, cost_function):
        # Get the layer activations
        layer_activations = self.forward(X)
        layer_inputs = [X] + layer_activations  # layer_input[i] is an input for network[i]
        logits = layer_activations[-1]

        # Compute the loss and the initial gradient
        loss = cost_function.compute(logits, y)
        loss_grad = cost_function.grad(logits, y)

        # <your code: propagate gradients through the network>
        grad_output = loss_grad
        layer_inputs = layer_inputs[:-1]
        for input, layer in zip(layer_inputs[::-1], self._layers[::-1]):
            # print(input.shape)
            # print(layer.weights.shape,grad_output.shape)#200,10 32,10, 32 10
            grad_output = layer.backward(input, grad_output)

        return np.mean(loss)

    def __len__(self):
        return len(self._layers)
