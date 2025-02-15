import numpy as np
from data_handlers import BatchGenerator
from tqdm.notebook import tqdm
import copy
from regularizers import L1L2


class ANN:

    def __init__(self):
        self._layers = []
        self.cost_function = None
        self.optimizer = None
        self.metrics = None
        self.train_log = []
        self.validation_log = []
        self.validation_metric_log = []

    def add(self, layer):
        self._layers.append(layer)

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
        layer_inputs = [X] + layer_activations
        y_hat = layer_activations[-1]

        # Compute the loss and the initial gradient
        loss = self.cost_function(y_hat, y)
        loss_grad = self.cost_function.grad(y_hat, y)

        grad_output = loss_grad
        layer_inputs = layer_inputs[:-1]
        for input_, layer in zip(layer_inputs[::-1], self._layers[::-1]):
            grad_output = layer.backward(input_, grad_output)

        return np.mean(loss)

    def compile(self, cost_function, optimizer, metric=None, regularizer=L1L2()):
        self.cost_function = cost_function
        self.optimizer = optimizer
        if metric is None:
            self.metric = cost_function
        else:
            self.metric = metric

        for layer in self._layers:
            layer.set_optimizer(copy.deepcopy(optimizer))
            layer.set_regularizer(regularizer)

    def __len__(self):
        return len(self._layers)

    def predict(self, X):
        return self.forward(X)[-1]

    def fit(self, X_train, y_train, X_val, y_val, batch_size, epochs):

        for epoch in range(epochs):
            print("Epoch", epoch)
            data_generator = BatchGenerator(batch_size)(X_train, y_train)
            for (X_batch, y_batch) in tqdm(data_generator):
                self.train(X_batch, y_batch)
                reg_cost = self.regularization_cost()
                self.train_log.append(self.cost_function(self.predict(X_batch), y_batch) + reg_cost)
                self.validation_log.append(self.cost_function(self.predict(X_val), y_val) + reg_cost)
                self.validation_metric_log.append(self.metric(self.predict(X_val), y_val))
            print("Val metric: ", self.validation_metric_log[-1])

    def summary(self):
        raise NotImplementedError()

    def regularization_cost(self):
        return np.sum([layer.get_regularization_cost() for layer in self._layers])
