import numpy as np
from plotly import tools
import plotly
import plotly.graph_objs as go

def plot_scatter(x,y):

    trace = go.Scatter(
        x=x,
        y=y,
        mode='lines+markers'
    )
    
    data = [trace]
    plotly.offline.plot(data, filename='scatter.html', auto_open=False)
    

def create_data_for_linear_model(num_samples, num_features):

    # creates data for a linear model, with coefficients starting at 1 and simply
    # increasing by one for each new variable/feature

    # seed the random number generator
    np.random.seed(123)

    # create num_features-D array of random numbers with num_samples elements, over the range [0, 1]
    X = np.random.rand(num_samples,num_features)
    
    # +2, +1 for the intercept, +1 more for the stop value
    coeffs = np.arange(2,num_features+2,1).reshape((2,1))

    # create 1D array of y values, add some noise to y
    y = 1 + np.dot(X, coeffs) + np.random.randn(num_samples,1)
    return X, y