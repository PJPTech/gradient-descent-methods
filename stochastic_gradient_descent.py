import numpy as np
from utils import create_data_for_linear_model, plot_scatter
from sklearn.linear_model import SGDRegressor

def run_SGD(num_epochs, num_samples, theta, X, y):

    #  adjust feature matrix to include column of ones for the intercept term
    X_b = np.c_[np.ones((num_samples,1)), X]

    # define learning schedule
    T0, T1 = 5, 50 
    def learning_schedule(t):
        return T0 / (t + T1)

    theta_path = []
    theta_path.append(theta)
    for epoch in range(0,num_epochs):
        for i in range(0,num_samples):
            random_index = np.random.randint(num_samples)
            xi = X_b[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            gradients = 2 * xi.T.dot(xi.dot(theta)-yi)
            eta = learning_schedule(epoch * num_samples + i)
            theta = theta - eta * gradients        
            theta_path.append(theta)

    # return final values of theta, and array of intermediary values in parameter space
    return theta, theta_path


if __name__ == '__main__':

    ''' 
    Example of stochastic gradient descent. Assumes mean square error (MSE) cost function
    '''

    num_samples=1000
    num_features = 2
    X, y = create_data_for_linear_model(num_samples,num_features)

    num_epochs = 50

    # initial guess at unknown parameters
    # +1 for the constant/intercept term
    theta = np.random.randn(num_features + 1,1)

    # run solver
    theta, theta_path = run_SGD(num_epochs, num_samples, theta, X, y)

    print('Long-hand linear algrebra calculation')
    print('Intercept {0}, coefficient {1}'.format(theta[0], theta[1:]))

    # plot the evolution of theta in the 2D parameter space
    theta_0 = [ theta[0][0] for theta in theta_path]
    theta_1 = [ theta[1][0] for theta in theta_path]
    plot_scatter(theta_0,theta_1)

    # use SKLearn functions to solve same problem
    sgd_reg = SGDRegressor(max_iter=num_epochs, penalty=None, eta0=0.1)
    sgd_reg.fit(X,y.ravel())
    print('SKLearn solver')
    print('Intercept {0}, coefficient {1}'.format(sgd_reg.intercept_, sgd_reg.coef_))