"""
Created on Thu Aug 27 03:55:41 2020

@author: kevin

TODO: add DataBlock object to model and use this info to make the plots more
informative as to what is being displayed
"""

import time

import pandas as pd
import plotly.graph_objects as go
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from kipet.library.common.read_write_tools import df_from_pyomo_data

from kipet.library.ResultsObject import ResultsObject
from pyomo.core.base.PyomoModel import ConcreteModel

# Default Matlab colors
colors_rgb = [(0,    0.4470,    0.7410),
          (0.8500,    0.3250,    0.0980),
          (0.9290,    0.6940,    0.1250),
          (0.4940,    0.1840,    0.5560),
          (0.4660,    0.6740,    0.1880),
          (0.3010,    0.7450,    0.9330),
          (0.6350,    0.0780,    0.1840)
          ]

# Convert to rgb format used in plotly
colors = ['rgb(' + ','.join([str(int(255*c)) for c in color]) + ')' for color in colors_rgb]

exp_to_pred = {'C': 'Z',
               'U': 'X',
               }


def _make_plots(model, var, filename=None, show_plot=True):
    """Makes the actual plots and filters out the data that is missing from
    the model.
    
    """
    if hasattr(model, var) and getattr(model, var) is not None:    
       
        if isinstance(model, ResultsObject):
            exp = getattr(model, var)
        elif isinstance(model, ConcreteModel):
            exp = df_from_pyomo_data(getattr(model, var))
        pred = None
        
        if hasattr(model, exp_to_pred[var]) and getattr(model, exp_to_pred[var]) is not None:
            if isinstance(model, ResultsObject):
                exp = getattr(model, exp_to_pred[var])
            elif isinstance(model, ConcreteModel):
                pred = df_from_pyomo_data(getattr(model, exp_to_pred[var]))  
    
        fig = go.Figure()    
        
        if pred is not None:
            for i, col in enumerate(pred.columns):
                fig.add_trace(
                    go.Scatter(x=pred.index,
                           y=pred[col],
                           name=col + ' (pred)',
                           line=dict(color=colors[i])
                           ))
        for i, col in enumerate(exp.columns):
            fig.add_trace(
                go.Scatter(x=exp.index,
                       y=exp[col],
                       name=col + ' (exp)',
                       mode='markers',
                       marker=dict(size=10, color=colors[i])),
                   )
    
        x_data = [t for t in model.alltime]
        x_axis_mod = 0.025*(x_data[-1] - x_data[0])
        fig.update_xaxes(range=[x_data[0]-x_axis_mod, x_data[-1]+x_axis_mod])
        fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor='black')
        fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='black')
    
        if show_plot:
            if filename is None:
                filename = f'chart_{str(time.time())[-4:]}.html'
            plot(fig, filename=filename)

    return None

def plot_results(model, var=None, filename=None, show_plot=True):
    """Function to plot experimental data and model predictions using plotly.
    Automatically finds the concentration and complementary state data in 
    the model (if the the model has the attribute, it will be checked)
    
    Args:
        model (pyomo Concrete): the model object after optimization
            This can be a single model or a dict
        
        var (str): the variable C or U to be displayed
        
        filename (str): optional filename
        
        show_plot (bool): defaults to True, shows the plots in the browswer
        
    Returns:
        None
    
    """
    if not isinstance(model, dict):
        model_dict = {'Results': model}
    else:
        model_dict = model
    
    for title, model in model_dict.items():
        
        if var is None:
            for _var in exp_to_pred.keys():
                _make_plots(model, _var, filename, show_plot)
            
        else:
            _make_plots(model, var, filename, show_plot)
    
    return None
 #%%       
if __name__ == '__main__':
        
    plot_results(nsd.models_dict[1])
        