from keras.datasets import mnist

from neural_networks import ANN
from layers import *
from cost_functions import SoftmaxCrossEntropy
from optimizers import SGD

(X_train, y_train), (X_test, y_test) = mnist.load_data()

image_vector_size = 28*28
X_train = X_train.reshape(X_train.shape[0], image_vector_size)
X_test = X_test.reshape(X_test.shape[0], image_vector_size)


ann = ANN()
ann.add(Dense(X_train.shape[1], 100))
ann.add(ReLU())
ann.add(Dense(100, 200))
ann.add(ReLU())
ann.add(Dense(200, 10))


ann.compile(SoftmaxCrossEntropy(), SGD())
ann.fit(X_train, y_train, X_test, y_test, batch_size=50, epochs=25)