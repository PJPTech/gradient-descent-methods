import numpy as np
from utils import create_data, plot_scatter

if __name__ == '__main__':

    ''' 
    Example of batch gradient descent. Assumes mean square error (MSE) cost function
    '''

    N=1000
    X, y = create_data(N)

    eta = 0.1
    num_iterations = 1000

    # initial guess at unknown parameters
    theta = np.random.randn(2,1)

    #  adjust feature matrix to include column of ones for the intercept term
    X_b = np.c_[np.ones((N,1)), X]

    theta_path = []
    theta_path.append(theta)
    for iteration in range(0,num_iterations):
        gradients = 2/N * X_b.T.dot(X_b.dot(theta)-y)
        theta = theta - eta * gradients        
        theta_path.append(theta)

    print('Long-hand linear algrebra calculation')
    print('Intercept {0}, coefficient {1}'.format(theta[0], theta[1]))

    # there is no SKLearn module for Batch Gradient Descent, other methods
    # are more commonly used in practice instead.

    # plot the evolution of theta in the 2D parameter space
    x = [ theta[0][0] for theta in theta_path]
    y = [ theta[1][0] for theta in theta_path]
    plot_scatter(x,y)
