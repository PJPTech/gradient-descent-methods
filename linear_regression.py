import numpy as np
from sklearn.linear_model import LinearRegression

from utils import create_data_for_linear_model

if __name__ == '__main__':

    num_samples = 1000
    num_features = 2
    X, y = create_data_for_linear_model(num_samples, num_features)

    # solve the Normal Equation long-handed way with linear algebra
    #  adjust feature matrix to include column of ones for the intercept term
    X_b = np.c_[np.ones((num_samples,1)), X]
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    print('Long-hand linear algrebra calculation')
    print('Intercept {0}, coefficient {1}'.format(theta_best[0], theta_best[1:]))

    # use SKLearn functions to solve same problem
    lin_reg = LinearRegression()
    lin_reg.fit(X,y)
    print('SKLearn solver')
    print('Intercept {0}, coefficient {1}'.format(lin_reg.intercept_, lin_reg.coef_))