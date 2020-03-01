import numpy as np
from tqdm import tqdm


class DataGenerator:

    def __call__(self):
        pass


class BatchGenerator(DataGenerator):

    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __call__(self, X, y, shuffle=True):
        assert len(X) == len(y)
        if shuffle:
            indices = np.random.permutation(len(X))

        for start_idx in tqdm(range(0, len(X) - self.batch_size + 1, self.batch_size)):
            if shuffle:
                excerpt = indices[start_idx:start_idx + self.batch_size]
            else:
                excerpt = slice(start_idx, start_idx + self.batch_size)
            yield X[excerpt], y[excerpt]




