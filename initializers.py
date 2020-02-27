import numpy as np


class Initializer:

    def __call__(self, shape):
        raise NotImplementedError


class Zeros(Initializer):

    def __call__(self, shape):
        return np.zeros(shape=shape)


class Ones(Initializer):

    def __call__(self, shape):
        return np.ones(shape=shape)


class Constant(Initializer):

    def __init__(self, value):
        self.value = value

    def __call__(self, shape):
        return np.ones(shape=shape) * self.value


class RandomInitializer(Initializer):

    def _set_and_update_seed(self):
        if self.seed is not None:
            np.seed(self.seed)
            self.seed += 1


class RandomNormal(RandomInitializer):

    def __init__(self, mean=0, std=1, seed=None):
        self.mean = mean
        self.std = std
        self.seed = seed

    def __call__(self, shape):
        self._set_and_update_seed()
        return np.random.normal(loc=self.mean, scale=self.std, size=shape)


class RandomUniform(RandomInitializer):

    def __init__(self, low=-0.05, high=0.05, seed=None):
        self.low = low
        self.high = high
        self.seed = seed

    def __call__(self, shape):
        return np.random.uniform(low=self.low, high=self.high, size=shape)


class Xavier(RandomInitializer):

    def __init__(self, mean=0, seed=None):
        self.mean = mean
        self.seed = seed

    def __call__(self, shape):
        self._set_and_update_seed()
        return np.random.normal(loc=self.mean, scale=2 / (shape[0] + shape[1]), size=shape)
