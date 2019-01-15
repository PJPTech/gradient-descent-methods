import numpy as np
from utils import create_data_for_linear_model, plot_scatter

def run_BGD(num_epochs, num_samples, theta, X, y):
    
    eta = 0.1
    
    #  adjust feature matrix to include column of ones for the intercept term
    X_b = np.c_[np.ones((num_samples,1)), X]

    theta_path = []
    theta_path.append(theta)
    for iteration in range(0,num_epochs):
        gradients = 2/num_samples * X_b.T.dot(X_b.dot(theta)-y)
        theta = theta - eta * gradients        
        theta_path.append(theta)

    return theta, theta_path

if __name__ == '__main__':

    ''' 
    Example of batch gradient descent. Assumes mean square error (MSE) cost function
    '''

    num_samples=1000
    num_features = 2
    X, y = create_data_for_linear_model(num_samples,num_features)

    num_epochs = 1000

    # initial guess at unknown parameters
    # +1 for the constant/intercept term
    theta = np.random.randn(num_features + 1,1)

    # run solver
    theta, theta_path = run_BGD(num_epochs, num_samples, theta, X, y)

    print('Long-hand linear algrebra calculation')
    print('Intercept {0}, coefficient {1}'.format(theta[0], theta[1:]))

    # there is no SKLearn module for Batch Gradient Descent, other methods
    # are more commonly used in practice instead.

    # plot the evolution of theta in the 2D parameter space
    x = [ theta[0][0] for theta in theta_path]
    y = [ theta[1][0] for theta in theta_path]
    plot_scatter(x,y)
