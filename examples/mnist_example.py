from keras.datasets import mnist

from neural_networks import ANN
from layers import *
from cost_functions import SoftmaxCrossEntropy
from optimizers import SGD
from metrics import accuracy


# load and prepare dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype(float) / 255.
X_test = X_test.astype(float) / 255.
X_train = X_train.reshape([X_train.shape[0], -1])
X_test = X_test.reshape([X_test.shape[0], -1])

# define network
ann = ANN()
ann.add(Dense(X_train.shape[1], 100))
ann.add(ReLU())
ann.add(Dense(100, 200))
ann.add(ReLU())
ann.add(Dense(200, 10))

# train network
ann.compile(SoftmaxCrossEntropy(), SGD(), accuracy)
ann.fit(X_train, y_train, X_test, y_test, batch_size=32, epochs=25)