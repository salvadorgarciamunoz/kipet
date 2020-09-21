# Standard library imports
import time

# Thirdparty library imports
import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from pyomo.core import *
from pyomo.core.base.PyomoModel import ConcreteModel
from pyomo.environ import *

# Kipet library imports
from kipet.library.common.read_write_tools import df_from_pyomo_data

"""
Constants used for plotting
"""
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

exp_data_maps = {'Z': ['C', 'Cm'],
                 'X': ['U'],
                 'S': None,
                 }

plot_vars = ['Z', 'X', 'S']

class ResultsObject(object):
    """Container for all of the results. Includes plotting functions"""
    
    def __init__(self):
        """
        A class to store simulation and optimization results.
        """
        # Data series
        self.generated_datetime = datetime.datetime
        self.results_name = None
        self.solver_statistics = {}

    def __str__(self):
        string = "\nRESULTS\n"
        
        result_vars = ['Z', 'C', 'Cm', 'K', 'S', 'X', 'dZdt', 'dXdt', 'P', 'sigma_sq']
        
        for var in result_vars:
            if hasattr(self, var) and getattr(self, var) is not None:
                var_str = var
                if var == 'sigma_sq':
                    var_str = 'Sigmas2'
                string += f'{var_str}:\n {getattr(self, var)}\n\n'
        
        return string

    def compute_var_norm(self, variable_name, norm_type=np.inf):
        var = getattr(self,variable_name)
        var_array = np.array(var)
        return np.linalg.norm(var_array,norm_type)
    
    def load_from_pyomo_model(self, instance, to_load=None):

        if to_load is None:
            to_load = []

        model_variables = set()
        for block in instance.block_data_objects():
            block_map = block.component_map(Var)
            for name in block_map.keys():
                model_variables.add(name)
                
        user_variables = set(to_load)

        if user_variables:
            variables_to_load = user_variables.intersection(model_variables)
        else:
            variables_to_load = model_variables

        # diff = user_variables.difference(model_variables)
        # if diff:
        #     print("WARNING: The following variables are not part of the model:")
        #     print(diff) 
        
        for block in instance.block_data_objects():
            block_map = block.component_map(Var)
            for name in variables_to_load:
                v = block_map[name]
                if v.dim()==0:
                    setattr(self, name, v.value)
                elif v.dim()==1:
                    setattr(self, name, pd.Series(v.get_values()))
                elif v.dim()==2:
                    d = v.get_values()
                    keys = d.keys()
                    if keys:
                        data_frame = df_from_pyomo_data(v)
                    else:
                        data_frame = pd.DataFrame(data=[],
                                                  columns = [],
                                                  index=[])
                    setattr(self, name, data_frame)        
                else:
                    raise RuntimeError('load_from_pyomo_model function not supported for models with variables with dimension > 2')
    
    def _make_plots(self, var, predict, filename=None, show_plot=True):
        """Makes the actual plots and filters out the data that is missing from
        the model.
        
        """
        if hasattr(self, var) and getattr(self, var) is not None and len(getattr(self, var)) > 0:    
            pred = getattr(self, var)
            exp = None
                
            fig = go.Figure()    
            
            if exp_data_maps[var] is not None:
      
                for exp_var in exp_data_maps[var]:
        
                    marker_options = {'size': 10,
                                      'opacity': 0.5,
                                     }
                    label = 'spectral'
                    
                    if exp_var == 'Cm':
                        marker_options = {'size': 15,
                                          'opacity': 0.75,
                                         }
                        label = 'measured'
                    
                    if predict and hasattr(self, exp_var) and getattr(self, exp_var) is not None and len(getattr(self, exp_var)) > 0:
                        exp = getattr(self, exp_var)
           
                        for i, col in enumerate(exp.columns):
                            fig.add_trace(
                                go.Scatter(x=exp.index,
                                        y=exp[col],
                                        name=col + f' ({label})',
                                        mode='markers',
                                        marker={**marker_options, 'color' :colors[i]}),
                                    )
       
            for i, col in enumerate(pred.columns):
                fig.add_trace(
                    go.Scatter(x=pred.index,
                           y=pred[col],
                           name=col,
                           line=dict(color=colors[i]),
                       )
                    )
                
            if var == 'S':
                fig.update_layout(
                    title="Absorbance Profile",
                    xaxis_title="Wavelength (cm)",
                    yaxis_title="Absorbance (L/(mol cm))",
                    )
            else:
                fig.update_layout(
                    title="Concentration Profile",
                    xaxis_title="Time",
                    yaxis_title="Concentration",
                    )
            #x_data = [t for t in model.alltime]
            #x_axis_mod = 0.025*(x_data[-1] - x_data[0])
            #fig.update_xaxes(range=[x_data[0]-x_axis_mod, x_data[-1]+x_axis_mod])
            fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor='black')
            fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='black')
        
            if show_plot:
                if filename is None:
                    filename = f'chart_{str(time.time())[-4:]}.html'
                plot(fig, filename=filename)
    
        return None
    
    def plot(self, var=None, predict=True, filename=None, show_plot=True):
        """Function to plot experimental data and model predictions using plotly.
        Automatically finds the concentration and complementary state data in 
        the model (if the the model has the attribute, it will be checked)
        
        Args:
            model (pyomo Concrete): the model object after optimization
                This can be a single model or a dict
            
            var (str): the variable Z, U, S to be displayed
            
            filename (str): optional filename
            
            show_plot (bool): defaults to True, shows the plots in the browswer
            
        Returns:
            None
        
        """
        vars_to_plot = plot_vars if var is None else [var]
            
        for _var in vars_to_plot:
            _predict = predict
            if _var == 'S':
                _predict = False
            
            if hasattr(self, _var) and getattr(self, _var) is not None: 
                self._make_plots(_var, _predict, filename, show_plot)
                
    @property
    def parameters(self):
        for k, v in self.P.items():
            print(k, v)
            
    @property
    def variances(self):
        for k, v in self.sigma_sq.items():
            print(k, v)

        
