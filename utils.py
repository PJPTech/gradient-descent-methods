import numpy as np
from plotly import tools
import plotly
import plotly.graph_objs as go

def plot_scatter(x,y):

    trace = go.Scatter(
        x=x,
        y=y,
        mode='markers'
    )
    
    data = [trace]
    plotly.offline.plot(data, filename='scatter.html', auto_open=False)
    

def create_data(N):

    # seed the random number generator
    np.random.seed(123)

    # create 1D array of random numbers with N elements, over the range [0, 1]
    X = np.random.rand(N,1)

    # create 1D array of y values, add some noise to y
    y = 0.25 + 2 * X + np.random.randn(N,1)
    return X, y