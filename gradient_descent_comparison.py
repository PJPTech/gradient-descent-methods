import numpy as np
from utils import create_data_for_linear_model
from mini_batch_gradient_descent import run_MBGD
from batch_gradient_descent import run_BGD
from stochastic_gradient_descent import run_SGD

from plotly import tools
import plotly
import plotly.graph_objs as go
import plotly.io as pio

if __name__ == '__main__':

    ''' 
        Comparison of Batch, Stochastic and Mini-Batch Gradient Descent
    '''

    num_samples=1000
    num_features = 2
    X, y = create_data_for_linear_model(num_samples,num_features)

    num_epochs_bgd = 1000
    num_epochs_sgd = 50
    num_epochs_mbgd = 2000
    batch_size = 10

    # calculate BGD
    theta_bgd = np.random.randn(num_features + 1,1)
    theta_bgd, theta_bgd_path = run_BGD(num_epochs_bgd, num_samples, theta_bgd, X, y)

    # calculate SGD
    theta_sgd = np.random.randn(num_features + 1,1)
    theta_sgd, theta_sgd_path = run_SGD(num_epochs_sgd, num_samples, theta_sgd, X, y)

    # calculate MBGD
    theta_mbgd = np.random.randn(num_features + 1,1)
    theta_mbgd, theta_mbgd_path = run_MBGD(num_epochs_mbgd, batch_size, num_samples, theta_mbgd, X, y)

    # plot the evolution of theta in the 2D parameter space
    theta_bgd_0 = [ theta[0][0] for theta in theta_bgd_path]
    theta_bgd_1 = [ theta[1][0] for theta in theta_bgd_path]
    theta_mbgd_0 = [ theta[0][0] for theta in theta_mbgd_path]
    theta_mbgd_1 = [ theta[1][0] for theta in theta_mbgd_path]
    theta_sgd_0 = [ theta[0][0] for theta in theta_sgd_path]
    theta_sgd_1 = [ theta[1][0] for theta in theta_sgd_path]

    trace_bgd = go.Scatter(
        x=theta_bgd_0,
        y=theta_bgd_1,
        mode='lines+markers',
        name='Batch'
    )
        
    trace_sgd = go.Scatter(
        x=theta_sgd_0,
        y=theta_sgd_1,
        mode='lines+markers',
        name='Stochastic'
    ) 
    
    trace_mbgd = go.Scatter(
        x=theta_mbgd_0,
        y=theta_mbgd_1,
        mode='lines+markers',
        name='Mini-Batch'
    )

    layout = go.Layout(
        title='Comparison of Gradient Descent Methods',
        font=dict(family='Arial', size=14, color='#7f7f7f')
    )

    data = [trace_bgd, trace_sgd, trace_mbgd]
    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename='scatter.html', auto_open=False)
    pio.write_image(fig, './gradient_descent_comparison.png')