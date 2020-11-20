# Standard library imports
import time
from pathlib import Path

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

colors += ['#4285F4', '#DB4437', '#F4B400', '#0F9D58',
                     '#185ABC', '#B31412', '#EA8600', '#137333',
                     '#d2e3fc', '#ceead6']

exp_data_maps = {'Z': ['C', 'Cm'],
                 'X': ['U'],
                 'S': None,
                 'Y': ['UD'],
                 }

plot_vars = ['Z', 'X', 'S', 'Y']
result_vars = ['Z', 'C', 'Cm', 'K', 'S', 'X', 'dZdt', 'dXdt', 'P', 'Pinit', 'sigma_sq', 'estimable_parameters', 'Y', 'UD']

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
        
        for var in result_vars:
            if hasattr(self, var) and getattr(self, var) is not None:
                var_str = var
                if var == 'sigma_sq':
                    var_str = 'Sigmas2'
                string += f'{var_str}:\n {getattr(self, var)}\n\n'
        
        return string
    
    def __repr__(self):
        return self.__str__()

    def compute_var_norm(self, variable_name, norm_type=np.inf):
        var = getattr(self,variable_name)
        var_array = np.array(var)
        return np.linalg.norm(var_array,norm_type)
    
    def load_from_pyomo_model(self, instance, to_load=None):

        if to_load is None:
            to_load = result_vars

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
    
    def _make_plots(self, pred, var, show_exp, extra_data, filename=None, show_plot=True, description=None):
        """Makes the actual plots and filters out the data that is missing from
        the model.
        
        """   
        exp = None
        fig = go.Figure()    
        
        for i, col in enumerate(pred.columns):
            fig.add_trace(
                go.Scatter(x=pred.index,
                       y=pred[col],
                       name=col,
                       line=dict(color=colors[i], width=4),
                   )
                )
        
            if exp_data_maps[var] is not None:
  
                for exp_var in exp_data_maps[var]:      
                    if show_exp and hasattr(self, exp_var) and getattr(self, exp_var) is not None and len(getattr(self, exp_var)) > 0:
                        exp = getattr(self, exp_var)
                        
                        marker_options = {'size': 10,
                                          'opacity': 0.5,
                                         }
                        label = 'spectral'
                        
                        if exp_var in ['Cm', 'U']:
                            marker_options = {'size': 15,
                                              'opacity': 0.75,
                                             }
                            label = 'measured'    
                    
                        if col in exp.columns:
                            fig.add_trace(
                            go.Scatter(x=exp.index,
                                    y=exp[col],
                                    name=col + f' ({label})',
                                    mode='markers',
                                    marker={**marker_options, 'color':colors[i]}),
                                )
            
            if extra_data is not None:
                
                if isinstance(extra_data['data'], pd.DataFrame):
                    col_check = col in pd.DataFrame(extra_data['data']).columns
                    data = extra_data['data']
                elif isinstance(extra_data['data'], pd.Series):
                    col_check = col == extra_data['data'].name
                    data = pd.DataFrame(extra_data['data'])
                    data.columns = [col]
                else:
                    raise ValueError('Extra data must be a pandas DataFrame or Series object')
                
                if col_check:
                    #data = extra_data['data']
                    mode = extra_data.get('mode', 'line')
                    label = extra_data.get('label', 'extra')
                    color = extra_data.get('color', colors[i])
                    line_dict = extra_data.get('line_options', {'color': color,
                                                                'dash': 'dash',
                                                                'width': 4,
                                                                })
                    
                    marker_dict = extra_data.get('marker_options', {'size': 10,
                                                                    'opacity': 0.5,
                                                                    'color': color,
                                                                    })
                    
                    trace_dict = line_dict if mode == 'line' else marker_dict
                    
                    fig.add_trace(
                        go.Scatter({'x': data.index,
                                    'y': data[col],
                                    'name': col + f' {label}',
                                    'mode': f'{mode}s',
                                     f'{mode}': {**trace_dict}}
                                   ),
                            )
                
        if extra_data is not None:
            
            counter = i + 1
            if isinstance(extra_data['data'], pd.DataFrame):
                col_check = col in pd.DataFrame(extra_data['data']).columns
                data = extra_data['data']
            elif isinstance(extra_data['data'], pd.Series):
                col_check = col == extra_data['data'].name
                data = pd.DataFrame(extra_data['data'])
                data.columns = [col]
            else:
                raise ValueError('Extra data must be a pandas DataFrame or Series object')
            
            for col in data.columns:
                if col in pred.columns:
                    continue
                
                mode = extra_data.get('mode', 'line')
                label = extra_data.get('label', 'extra')
                color = extra_data.get('color', colors[counter])
                line_dict = extra_data.get('line_options', {'color': color,
                                                            'dash': 'dash',
                                                            'width': 4,
                                                            })
                
                marker_dict = extra_data.get('marker_options', {'size': 10,
                                                                'opacity': 0.5,
                                                                'color': color,
                                                                })
                
                trace_dict = line_dict if mode == 'line' else marker_dict
                
                fig.add_trace(
                    go.Scatter({'x': data.index,
                                'y': data[col],
                                'name': col + f' {label}',
                                'mode': f'{mode}s',
                                 f'{mode}': {**trace_dict}}
                               ),
                        )
                counter += 1
        
        if description is not None:
            title = description.get('title', '')
            xaxis_title = description.get('xaxis', '')
            yaxis_title = description.get('yaxis', '')
        
            fig.update_layout(
                title=title,
                xaxis_title=xaxis_title,
                yaxis_title=yaxis_title,
                )
        
        else:
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
                
        x_data = [t for t in pred.index]
        
        x_axis_mod = 0.025*(float(x_data[-1]) - float(x_data[0]))
        fig.update_xaxes(range=[float(x_data[0])-x_axis_mod, float(x_data[-1])+x_axis_mod])
        fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor='#4e4e4e')
        fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='#4e4e4e')
    
        if show_plot:
            if filename is None:
                filename = f'chart_{str(time.time())[-4:]}.html'
                
            default_dir = Path(self.file_dir)
            filename = default_dir.joinpath(filename)
            print(f'Plot saved as: {filename}')
                
            plot(fig, filename=filename.as_posix())
    
        return None
    
    def plot(self, var=None, subset=None, show_exp=True, extra_data=None, filename=None, show_plot=True, description=None):
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
        vars_to_plot = exp_data_maps.keys() if var is None else [var]
            
        for _var in vars_to_plot:
            _show_exp = show_exp
            if _var == 'S':
                _show_exp = False
            
            if hasattr(self, _var) and getattr(self, _var) is not None and len(getattr(self, _var)) > 0:
                if subset is not None and subset in getattr(self, _var).columns:
                    pred = pd.DataFrame(getattr(self, _var)[subset])
                else:
                    pred = getattr(self, _var)
                    
                self._make_plots(pred, _var, _show_exp, extra_data, filename, show_plot, description)
                
    @property
    def parameters(self):
        return self.P
            
    @property
    def show_parameters(self):
        print('\nThe estimated parameters are:')
        for k, v in self.P.items():
            print(k, v)
            
    @property
    def variances(self):
        print('\nThe estimated variances are:')
        for k, v in self.sigma_sq.items():
            print(k, v)

        
