import unittest
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import normalize

# create linear NN model
from neural_networks import ANN
from layers import *
from cost_functions import MSE
from optimizers import Momentum

ERROR_RELATIVE = .05  # percent


class TestANN(unittest.TestCase):

    def test_linear_regression(self):
        # target dataset
        X, y = datasets.load_diabetes(return_X_y=True)
        y = y.reshape(-1, 1)
        X_norm = normalize(X)
        n_obs, n_feature = X_norm.shape

        # create linear regression benchmark
        reg = linear_model.LinearRegression()
        reg.fit(X_norm, y)
        mse_reg = mean_squared_error(reg.predict(X_norm), y)
        atol = mse_reg * ERROR_RELATIVE

        # define network
        ann = ANN()
        ann.add(Dense(X_norm.shape[1], 1))

        # train network
        ann.compile(MSE(), Momentum(), MSE())
        # ignoring for this example train/validation split
        ann.fit(X_norm, y, X_norm, y, batch_size=10, epochs=50)  # small batch size

        y_hat_nn = ann.predict(X_norm)

        mse_nn = mean_squared_error(y_hat_nn, y)

        self.assertTrue(np.allclose(mse_reg, mse_nn, atol=atol))


if __name__ == '__main__':
    unittest.main()
