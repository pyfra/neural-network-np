class ANN:

    def __init__(self):
        self._layers = []

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

    def train(self):
        pass

    def __len__(self):
        return len(self._layers)
