import numpy as np

from sklearn.linear_model import LinearRegression

def create_data(N):

    # create 1D array of random numbers with N elements, over the range [0, 1]
    X = np.random.rand(N,1)

    # create 1D array of y values, add some noise to y
    y = 0.25 + 2 * X + np.random.randn(N,1)
    return X, y


if __name__ == '__main__':

    N=1000
    X, y = create_data(N)

    # solve the Normal Equation long-handed way with linear algebra
    X_b = np.c_[np.ones((N,1)), X]
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    print('Long-hand linear algrebra calculation')
    print('Intercept {0}, coefficient {1}'.format(theta_best[0], theta_best[1]))

    # use SKLearn functions to solve same problem
    lin_reg = LinearRegression()
    lin_reg.fit(X,y)
    print('SKLearn solve')
    print('Intercept {0}, coefficient {1}'.format(lin_reg.intercept_, lin_reg.coef_))