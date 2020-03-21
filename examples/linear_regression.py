import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import normalize

# target dataset
X, y = datasets.load_diabetes(return_X_y=True)
y = y.reshape(-1, 1)
X_norm = normalize(X)
n_obs, n_feature = X_norm.shape

# create linear regression benchmark
reg = linear_model.LinearRegression()
reg.fit(X_norm, y)
mse_reg = mean_squared_error(reg.predict(X_norm), y)
print("coef from linear regression: %s" % ', '.join(map(str, reg.coef_)))
print("MSE for regression: %.4f" % mse_reg)

# create linear NN model
from neural_networks import ANN
from layers import *
from cost_functions import MSE
from optimizers import Momentum

# define network
ann = ANN()
ann.add(Dense(X_norm.shape[1], 1))

# train network
ann.compile(MSE(), Momentum(), MSE())
# ignoring for this example train/validation split
ann.fit(X_norm, y, X_norm, y, batch_size=10, epochs=100) # small batch size

y_hat_nn = ann.predict(X_norm)

mse_nn = mean_squared_error(y_hat_nn, y)
print("MSE for nn: %.4f" % mse_nn)
