import numpy as np
from utils import create_data, plot_scatter
from sklearn.linear_model import SGDRegressor

T0, T1 = 5, 50 

def learning_schedule(t):
    return T0 / (t + T1)


if __name__ == '__main__':

    ''' 
    Example of stochastic gradient descent. Assumes mean square error (MSE) cost function
    '''

    N=1000
    X, y = create_data(N)

    num_epochs = 50

    # initial guess at unknown parameters
    theta = np.random.randn(2,1)

    #  adjust feature matrix to include column of ones for the intercept term
    X_b = np.c_[np.ones((N,1)), X]

    theta_path = []
    theta_path.append(theta)
    for epoch in range(0,num_epochs):
        for i in range(0,N):
            random_index = np.random.randint(N)
            xi = X_b[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            gradients = 2 * xi.T.dot(xi.dot(theta)-yi)
            eta = learning_schedule(epoch * N + i)
            theta = theta - eta * gradients        
            theta_path.append(theta)

    #theta = scaler.inverse_transform(theta)
    print('Long-hand linear algrebra calculation')
    print('Intercept {0}, coefficient {1}'.format(theta[0], theta[1]))

    # plot the evolution of theta in the 2D parameter space
    theta_0 = [ theta[0][0] for theta in theta_path]
    theta_1 = [ theta[1][0] for theta in theta_path]
    plot_scatter(theta_0,theta_1)

        # use SKLearn functions to solve same problem
    sgd_reg = SGDRegressor(max_iter=num_epochs, penalty=None, eta0=0.1)
    sgd_reg.fit(X,y.ravel())
    print('SKLearn solver')
    print('Intercept {0}, coefficient {1}'.format(sgd_reg.intercept_, sgd_reg.coef_))