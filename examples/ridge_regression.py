import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import normalize
import numpy as np

LAMBDA = .1

# target dataset
X, y = datasets.load_diabetes(return_X_y=True)
y = y.reshape(-1, 1)
X_norm = normalize(X)
n_obs, n_feature = X_norm.shape

# create ridge ridge_regression benchmark
ridge_reg = linear_model.Ridge(alpha=LAMBDA)
ridge_reg.fit(X_norm, y)
mse_ridge_reg = mean_squared_error(ridge_reg.predict(X_norm), y) + LAMBDA * np.sum(np.square(ridge_reg.coef_))

# create linear NN model
from neural_networks import ANN
from layers import *
from cost_functions import MSE
from optimizers import Momentum
from regularizers import L2

# define network
ann = ANN()
ann.add(Dense(X_norm.shape[1], 1))

# train network
"""
Need to be careful in translating the alpha from the Ridge Regression to the L2 regularization. 
In our implementation the penalty term is lambda / 2 * n, where n is the batch size.
"""

LEARNING_RATE = .1
BATCH_SIZE = 1
ann.compile(MSE(), Momentum(learning_rate=LEARNING_RATE), MSE(), L2(LAMBDA * BATCH_SIZE * 2))
# ignoring for this example train/validation split
ann.fit(X_norm, y, X_norm, y, batch_size=BATCH_SIZE, epochs=30)  # small batch size

y_hat_nn = ann.predict(X_norm)
w_ann = ann._layers[-1].weights.reshape(1, -1)
mse_nn = mean_squared_error(y_hat_nn, y) + LAMBDA * np.sum(np.square(w_ann))

#### printing stats
print('#' * 10)
print("MSE for ridge_regression: %.4f" % mse_ridge_reg)
print("MSE for nn: %.4f" % mse_nn)
print('#' * 10)

print("coef from ridge ridge_regression: %s" % ', '.join(map(str, ridge_reg.coef_)))
print("coef from nn: %s" % ', '.join(map(str, w_ann)))

#### plot MSE nn
val_log = ann.validation_log
pd.Series(val_log)[:100].plot()  # plot first 100 iterations
