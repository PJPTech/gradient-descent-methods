import numpy as np
from utils import create_data_for_linear_model, plot_scatter

def run_MBGD(num_epochs, batch_size, num_samples, theta, X, y):

    num_batches = int(num_samples / batch_size) #assume this divides to give and interger, not added any checks

    #  adjust feature matrix to include column of ones for the intercept term
    X_b = np.c_[np.ones((num_samples,1)), X]    

    theta_path = []
    theta_path.append(theta)

    # define learning schedule
    T0, T1 = 5, 50 
    def learning_schedule(t):
        return T0 / (t + T1)

    for epoch in range(0,num_epochs):
        for i in range(0,num_batches):
            i_start = i * batch_size
            i_end = i_start + batch_size
            X_b_mini = X_b[i_start:i_end]
            Y_mini = y[i_start:i_end]
            gradients = 2 / batch_size * X_b_mini.T.dot(X_b_mini.dot(theta)-Y_mini)
            eta = learning_schedule(epoch * num_batches + i)
            theta = theta - eta * gradients        
            theta_path.append(theta)

    # return final values of theta, and array of intermediary values in parameter space
    return theta, theta_path


if __name__ == '__main__':

    ''' 
    Example of mini-batch gradient descent. Assumes mean square error (MSE) cost function
    '''

    num_samples=1000
    num_features = 2
    X, y = create_data_for_linear_model(num_samples,num_features)

    num_epochs = 50
    batch_size = 100

    # initial guess at unknown parameters
    # +1 for the constant/intercept term
    theta = np.random.randn(num_features + 1,1)

    # run solver
    theta, theta_path = run_MBGD(num_epochs, batch_size, num_samples, theta, X, y)

    print('Long-hand linear algrebra calculation')
    print('Intercept {0}, coefficient {1}'.format(theta[0], theta[1:]))

    # plot the evolution of theta in the 2D parameter space
    theta_0 = [ theta[0][0] for theta in theta_path]
    theta_1 = [ theta[1][0] for theta in theta_path]
    plot_scatter(theta_0,theta_1)