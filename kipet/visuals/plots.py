"""
Plotting class for KIPET
"""
# Standard library imports
from pathlib import Path
import os
import sys

# Third party imports 
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from kipet.model_tools.pyomo_model_tools import convert

pio.templates.default = "plotly_white"

# Kipet library imports
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

# Use for making SVGs

# plot_options = {
#     'label_font': dict(
#             size=32,
#         ),
#     'title_font': dict(
#             size=32,
#         ),
#     'tick_font': dict(
#             size=24,
#         ),
#     }

plot_options = {
    'label_font': dict(
            size=14,
        ),
    'title_font': dict(
            size=18,
        ),
    'tick_font': dict(
            size=14,
        ),
    }


class PlotObject:

    """This will hold the relevant information needed to make a plot in KIPET

    This object is created in ReactionModel and accessed using the plot method therein.

    :param ReactionModel reaction_model: A ReactionModel instance
    :param bool jupyter: Indicates if the user is using a Jupyter notebook
    :param str filename: Optional file name for the plot
    """
    def __init__(self, reaction_model=None, jupyter=False, filename=None, show=False):
        """Initialization of the PlotObject instance

        :param ReactionModel reaction_model: A ReactionModel instance
        :param bool jupyter: Indicates if the user is using a Jupyter notebook
        :param str filename: Optional file name for the plot

        """
        self.reaction_model = reaction_model
        self.name = reaction_model.name
        self.color_num = 0
        self.filename = filename
        self.jupyter = jupyter
        self.show = show
        
        self.folder_name = self.reaction_model.timestamp
        if self.filename is not None:
            self.folder_name = self.filename

    @staticmethod
    def _make_line_trace(fig, x, y, name, color):
        """Convenience method for making traces in place.

        :param go.Figure fig: A figure object
        :param list x: X-axis values for plot
        :param list y: Y-axis values for plot
        :param str name: Name of the data
        :param int color: Index for the color

        :return: None
        
        """
        line = dict(color=colors[color], width=2)
        fig.add_trace(
            go.Scatter(x=x,
                       y=y,
                       name=name,
                       line=line,
               )
            )
        return None
    
    def _make_marker_trace(self, fig, x, y, name, color, marker_options):
        """Convenience method for making marker traces in place.

        :param go.Figure fig: A figure object
        :param list x: X-axis values for plot
        :param list y: Y-axis values for plot
        :param str name: Name of the data
        :param int color: Index for the color
        :param dict marker_options: Options for the marker characteristics

        :return: None
        
        """
        fig.add_trace(
            go.Scatter(x=x,
                       y=y,
                       name=name,
                       mode='markers',
                       marker={**marker_options, 'color':colors[self.color_num]}),
                    )
        return None

    def _fig_finishing(self, fig, pred, plot_name='Z', use_index=True, exp=None):
        """Finish the plots before showing.

        This method creates the plots and opens the browser to show them. It also saves the plots as SVGs in the
        same charts directory as the HTML versions.

        :param go.Figure fig: The figure object
        :param pandas.DataFrame pred: The predicted data from the model
        :param str plot_name: The name of the plot (based on data)

        :return: None

        """
        if use_index:
            x_data = [t for t in pred.index]
        else:
            if isinstance(pred, np.ndarray):
                 x_data = (np.min(pred), np.max(pred))
                 y_data = (np.min(exp), np.max(exp)) 
            else:
                x_data = (np.min(pred.values), np.max(pred.values))
                y_data = (np.min(exp.values), np.max(exp.values)) 
            y_axis_mod = 0.025*(float(y_data[-1]) - float(y_data[0]))
            fig.update_yaxes(range=[float(y_data[0])-y_axis_mod, float(y_data[-1])+y_axis_mod])

        x_axis_mod = 0.025*(float(x_data[-1]) - float(x_data[0]))
        fig.update_xaxes(range=[float(x_data[0])-x_axis_mod, float(x_data[-1])+x_axis_mod])
        
        fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor='#4e4e4e', titlefont=plot_options['label_font'], tickfont=plot_options['tick_font'])
        fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='#4e4e4e', titlefont=plot_options['label_font'], tickfont=plot_options['tick_font'])
        
        fig.update_layout(title_font=plot_options['title_font'])
        fig.update_layout(legend_font=plot_options['label_font'])
        
        filename = f'{plot_name}.html'
        # Change the folder directory
            
        stem = self.reaction_model.file.stem
        
        if self.jupyter:
            chart_dir = Path.cwd().joinpath('results', f'{stem}-{self.folder_name}' , 'charts', f'{self.name}')
            plot_method = pio.show
            fig.update_layout(         
                autosize=False,
                width=1200,
                height=800,
                margin=dict(
                    l=50,
                    r=50,
                    b=50,
                    t=50,
                    pad=4
                    ),
                )
        else:
        
            calling_file_name = os.path.dirname(os.path.realpath(sys.argv[0]))
            chart_dir = Path(calling_file_name).joinpath('results', f'{stem}-{self.folder_name}', 'charts', f'{self.name}')
            plot_method = pio.write_html
        
        chart_dir.mkdir(parents=True, exist_ok=True)
        filename = chart_dir.joinpath(filename)
        #if not self.jupyter:
        #   print(f'Plot saved as: {filename}')
           
        self.save_static_image = True
        if self.save_static_image:
            fig.write_image(f'{filename}.svg', width=1400, height=900)

        plot_method(fig,
                    file=filename.as_posix(),
                    auto_open=self.show,
                    include_plotlyjs='cdn')
    
        return filename

    def _state_plot(self, fig, var, pred, exp, use_spectral_format=False):
        """Generic method to plot state profiles

        :param go.Figure fig: The figure object
        :param str var: The variable to plot
        :param pandas.DataFrame pred: The predicted data from the model
        :param pandas.DataFrame exp: The experimental data
        :param bool use_spectral_format: For absorbance profiles True, otherwise False

        :return: None

        """
        self._make_line_trace(fig=fig,
                              x=pred.index,
                              y=pred[var],
                              name=var,
                              color=self.color_num,
                              )
        marker_options = {'size': 8,
                          'opacity': 0.5,
                          }
        label = 'exp.'   
        if use_spectral_format:
            marker_options = {'size': 8,
                              'opacity': 0.5,
                             }
            label = 'spectral'         
        if exp is not None and var in exp.columns:
            self._make_marker_trace(fig=fig,
                                    x=exp.index,
                                    y=exp[var],
                                    name=f'{var} ({label})',
                                    color=self.color_num,
                                    marker_options=marker_options,
                                    )
        return None


    def _plot_input_D_data(self):
        """Plot all input data concentration profiles
        
        """
        data = self.reaction_model.spectra.data

        fig = go.Figure()
        fig.add_trace(go.Surface(x=data.columns,
                                 y=data.index,
                                 z=data.values,
                                 ))

        fig.update_layout(scene=dict(
            xaxis_title='Wavelength',
            yaxis_title='Time',
            zaxis_title='Absorbance'),
            margin=dict(r=100, b=50, l=100, t=50),
            #title_text=f'{self.name}: Spectral Data',
            #title_font_size=plot_options['title_font']['size'],
            )

        calling_file_name = os.path.dirname(os.path.realpath(sys.argv[0]))
        chart_dir = Path(calling_file_name).joinpath('results', f'{self.reaction_model.file.stem}-{self.folder_name}', 'charts', f'{self.name}')
        plot_method = pio.write_html
        
        chart_dir.mkdir(parents=True, exist_ok=True)
        filename = chart_dir.joinpath('spectral_data.html')
        if not self.jupyter:
           print(f'Plot saved as: {filename}')
           
        self.save_static_image = True
        #if self.save_static_image:
        #    fig.write_image(f'{filename.as_posix()[:-5]}.svg', width=1400, height=900)

        plot_method(fig, file=filename.as_posix(), auto_open=False)
        
        return filename


    def _plot_all_Z(self):
        """Plot all concentration profiles
        
        """
        fig = go.Figure()
        use_spectral_format = False
        pred = getattr(self.reaction_model.results, 'Z')
        if hasattr(self.reaction_model.results, 'Cm'):
            exp = getattr(self.reaction_model.results, 'Cm')
        elif hasattr(self.reaction_model.results, 'C'):
            if self.reaction_model.models['_s_model'] and not self.reaction_model.models['v_model'] and not self.reaction_model.models['p_model']:
                exp = None
            else:
                exp = getattr(self.reaction_model.results, 'C')
                use_spectral_format = True
        else:
            exp = None
        for i, col in enumerate(pred.columns):
            self._state_plot(fig, col, pred, exp, use_spectral_format=use_spectral_format)
            self.color_num += 1
        self.color_num = 0
        var_data = self.reaction_model.components[col]
        state = f'{self.reaction_model.components[col].state}'.capitalize()
        title = f'Model: {self.reaction_model.name} | Concentration Profiles'
        time_scale = f'Time [{self.reaction_model.unit_base.time}]'

        state_units = self._get_proper_unit_str(var_data)
        fig.update_layout(
                title=title,
                xaxis_title=f'{time_scale}',
                yaxis_title=f'{state} [{state_units}]',
                )
        self._fig_finishing(fig, pred, plot_name='all-concentration-profiles')

        return None

    def _plot_Z(self, var):
        """Plot state profiles

        :param str var: concentration variable
        
        """
        fig = go.Figure()
        use_spectral_format = False
        pred = getattr(self.reaction_model.results, 'Z')
        if hasattr(self.reaction_model.results, 'Cm'):
            exp = getattr(self.reaction_model.results, 'Cm')
            self._state_plot(fig, var, pred, exp)
        elif hasattr(self.reaction_model.results, 'C'):
            exp = getattr(self.reaction_model.results, 'C')
            use_spectral_format = True
            self._state_plot(fig, var, pred, exp, use_spectral_format=use_spectral_format)
        else:
            exp = None
        var_data = self.reaction_model.components[var]
        state = f'{self.reaction_model.components[var].state}'.capitalize()
        description = f'| Description: {var_data.description}' if var_data.description is not None else ''
        title = f'Model: {self.reaction_model.name} | Variable: {var_data.name} {description}'
        time_scale = f'Time [{self.reaction_model.unit_base.time}]'
        state_units = self._get_proper_unit_str(var_data)
        fig.update_layout(
                title=title,
                xaxis_title=f'{time_scale}',
                yaxis_title=f'{state} [{state_units}]',
                )
        self._fig_finishing(fig, pred, plot_name=f'{var}-concentration-profile')
        
    def _plot_Y(self, var, extra=None):
        """Plot state profiles

        :param str var: algebraic variable

        :return: None
        """
        fig = go.Figure()
        pred = self.reaction_model.results.Y
        if hasattr(self.reaction_model.results, 'UD'):
            exp = self.reaction_model.results.UD
        else:
            exp = None
        self._state_plot(fig, var, pred, exp)
        var_data = self.reaction_model.algebraics[var]
        description = f'| Description: {var_data.description}' if var_data.description is not None else ''
        title = f'Model: {self.reaction_model.name} | Variable: {var_data.name} {description}'
        time_scale = f'Time [{self.reaction_model.unit_base.time}]'
        state_units = self._get_proper_unit_str(var_data, check_expr=True)
        fig.update_layout(
            title=title,
            xaxis_title=f'{time_scale}',
            yaxis_title=f'[{state_units}]',
            )
        self._fig_finishing(fig, pred, plot_name=f'{var}-profile')

    def _plot_X(self, var):
        """Plot state profiles
        
        :param str var: state variable

        :return: None

        """
        fig = go.Figure()
        pred = getattr(self.reaction_model.results, 'X')
        if hasattr(self.reaction_model.results, 'U'):
            exp = getattr(self.reaction_model.results, 'U')
        else:
            exp = None
        self._state_plot(fig, var, pred, exp)
        var_data = self.reaction_model.states[var]
        description = f'| Description: {var_data.description}' if var_data.description is not None else ''
        title = f'Model: {self.reaction_model.name} | Variable: {var_data.name} {description}'
        time_scale = f'Time [{self.reaction_model.unit_base.time}]'
        state = f'{var_data.description}'.capitalize() if var_data.description is not None else 'State' 
        state_units = self._get_proper_unit_str(var_data)
        fig.update_layout(
                title=title,
                xaxis_title=f'{time_scale}',
                yaxis_title=f'{state} [{state_units}]',
                )
        self._fig_finishing(fig, pred, plot_name=f'{var}-state-profile')
        
    def _plot_all_S(self):
        """Plot all S profiles

        :return: None

        """
        fig = go.Figure()
        pred = getattr(self.reaction_model.results, 'S')
        exp = None
        for i, col in enumerate(pred.columns):
            self._state_plot(fig, col, pred, exp)
            self.color_num += 1
        self.color_num = 0
        description = f'Description: Single species absorbance profiles'
        title = f'Model: {self.reaction_model.name} | {description}'
        time_scale = f'Wavelength [centimeter]'
        state = 'Absorbance'
        state_units = 'liter / mol / centimeter'
        fig.update_layout(
                title=title,
                xaxis_title=f'{time_scale}',
                yaxis_title=f'{state} [{state_units}]',
                )
        self._fig_finishing(fig, pred, plot_name='absorbance-spectra-all')

    def _plot_S(self, var, orig=False):
        """Plot individual S profile

        :param str var: component name

        :return: None

        """
        fig = go.Figure()
        if not orig:
            pred = getattr(self.reaction_model.results, 'S')
        else:
            pred = self.reaction_model.components[var].S
        exp = None
        self._state_plot(fig, var, pred, exp)
        var_data = self.reaction_model.results.S[var]
        description = f'| Description: Single species absorbance profiles'
        title = f'Model: {self.reaction_model.name} | Variable: {var} {description}'
        time_scale = f"Wavelength [centimeter]"
        state = 'Absorbance'
        state_units = 'liter / mol / centimeter'
        fig.update_layout(
                title=title,
                xaxis_title=f'{time_scale}',
                yaxis_title=f'{state} [{state_units}]',
                )
        self._fig_finishing(fig, pred, plot_name=f'{var}-absorbance-spectra')
        
    def _plot_step(self, var):
        """Plot the step function

        :param str var: The step variable to plot

        :return: None

        """
        fig = go.Figure()
        pred = self.reaction_model.results.step
        self._state_plot(fig, var, pred, None)
        title = f'Model: {self.reaction_model.name} | Variable: {var} | Step Function'
        time_scale = f'Time [{self.reaction_model.unit_base.time}]'
        fig.update_layout(
            title=title,
            xaxis_title=f'{time_scale}',
            yaxis_title=f'[ - ]',
            )
        self._fig_finishing(fig, pred, plot_name=f'{var}-step-profile')

        return None


    def _residual_plot(self, fig, var, pred, exp, use_spectral_format=False):
        """Generic method to plot state profiles

        :param go.Figure fig: The figure object
        :param str var: The variable to plot
        :param pandas.DataFrame pred: The predicted data from the model
        :param pandas.DataFrame exp: The experimental data
        :param bool use_spectral_format: For absorbance profiles True, otherwise False

        :return: None

        """
        marker_options = {'size': 10,
                          'opacity': 0.8,
                          }
        if use_spectral_format:
            marker_options = {'size': 6,
                              'opacity': 0.4,
                          }    
        
        label = 'res.'   
        residuals = exp - pred
        
        self._make_marker_trace(fig=fig,
                                x=residuals.index,
                                y=residuals,
                                name=f'{var} ({label})',
                                color=self.color_num,
                                marker_options=marker_options,
                                )
        return None

    def _plot_Z_residuals(self):
        """Plot state profiles

        :param str var: concentration variable
        
        """
        fig = go.Figure()
        use_spectral_format = False
        pred = getattr(self.reaction_model.results, 'Z')
        exp = getattr(self.reaction_model.results, 'Cm')
        
        for i, col in enumerate(exp.columns):
            self._residual_plot(fig, col, pred[col], exp[col])
            self.color_num += 1
        self.color_num = 0
            
        title = f'Model: {self.reaction_model.name} | Concentration Residuals'
        time_scale = f'Time [{self.reaction_model.unit_base.time}]'

        fig.update_layout(
                title=title,
                xaxis_title=f'{time_scale}',
                yaxis_title='Residuals',
                )
        
        filename = self._fig_finishing(fig, pred, plot_name=f'concentration-residuals')
        return filename
    
    def _plot_D_residuals(self):
        """Plot state profiles

        :param str var: concentration variable
        
        """
        fig = go.Figure()
        use_spectral_format = False
        
        exp = convert(self.reaction_model.p_model.D)
        C = getattr(self.reaction_model.results, 'C')
        S = getattr(self.reaction_model.results, 'S')
        C = C.loc[:, S.columns]
        pred = C.dot(S.T)
        
        for i, col in enumerate(exp.columns):
            self._residual_plot(fig, col, pred[col], exp[col], use_spectral_format=True)
            #self.color_num += 1
        self.color_num = 0
            
        title = f'Model: {self.reaction_model.name} | Spectral Residuals'
        time_scale = f'Time [{self.reaction_model.unit_base.time}]'

        fig.update_layout(
                title=title,
                xaxis_title=f'{time_scale}',
                yaxis_title='Residuals',
                showlegend=False,
                )
        
        filename = self._fig_finishing(fig, pred, plot_name=f'spectral-residuals')
        return filename
        
        
    def _parity_plot(self, fig, var, pred, exp, use_spectral_format=False):
        """Generic method to plot state profiles

        :param go.Figure fig: The figure object
        :param str var: The variable to plot
        :param pandas.DataFrame pred: The predicted data from the model
        :param pandas.DataFrame exp: The experimental data
        :param bool use_spectral_format: For absorbance profiles True, otherwise False

        :return: None

        """
        marker_options = {'size': 8,
                          'opacity': 0.6,
                          }
        
        self._make_marker_trace(fig=fig,
                                x=pred,
                                y=exp,
                                name=f'{var}',
                                color=self.color_num,
                                marker_options=marker_options,
                                )
        
        return None
        
    def _plot_Z_parity(self):
        """Plot state profiles

        :param str var: concentration variable
        
        """
        fig = go.Figure()
        pred_raw = getattr(self.reaction_model.results, 'Z')
        exp = getattr(self.reaction_model.results, 'Cm')
        pred = pred_raw.loc[exp.index]
            
        line = dict(color='gray', width=2, dash='dash')
        fig.add_trace(
            go.Scatter(x=[0, np.max(pred.values)],
                       y=[0, np.max(pred.values)],
                       line=line,
               )
            )
            
        for i, col in enumerate(pred.columns):
            if col not in exp.columns:
                self.color_num += 1
                continue
            self._parity_plot(fig, col, pred[col], exp[col])
            self.color_num += 1
        self.color_num = 0

        title = f'Model: {self.reaction_model.name} | Concentration Parity'

        fig.update_layout(
                title=title,
                xaxis_title='Model Prediction',
                yaxis_title='Measured',
                autosize=False,
                width=550,
                height=550,
                )
        filename = self._fig_finishing(fig, pred, plot_name='concentration-parity', use_index=False, exp=exp)
        return filename
    
    def _plot_D_parity(self):
        """Plot state profiles

        :param str var: concentration variable
        
        """
        fig = go.Figure()
        use_spectral_format = False
       
        exp = convert(self.reaction_model.p_model.D)
        
        C = getattr(self.reaction_model.results, 'C')
        S = getattr(self.reaction_model.results, 'S')
        C = C.loc[:, S.columns]
        pred = C.dot(S.T)
        
        exp = exp.values.flatten()
        pred = pred.values.flatten()
        
        line = dict(color='gray', width=2, dash='dash')
        fig.add_trace(
            go.Scatter(x=[0, np.max(pred)],
                       y=[0, np.max(pred)],
                       line=line,
               )
            )
            
        self.color_num = 0
        self._parity_plot(fig, 'D', pred, exp)            
        
        title = f'Model: {self.reaction_model.name} | Spectral Parity'
        time_scale = f'Time [{self.reaction_model.unit_base.time}]'
        
        fig.update_layout(
                title=title,
                xaxis_title=f'Model Prediction',
                yaxis_title='Measured',
                autosize=False,
                width=550,
                height=550,
                )
        filename = self._fig_finishing(fig, pred, plot_name=f'spectral-parity', use_index=False, exp=exp)
        return filename
        
    def _plot_X_residuals(self, var):
        """Plot state profiles

        :param str var: concentration variable
        
        """
        fig = go.Figure()
        
        pred = getattr(self.reaction_model.results, 'X')
        if hasattr(self.reaction_model.results, 'U'):
            exp = getattr(self.reaction_model.results, 'U')
        else:
            exp = None
        
        self._residual_plot(fig, var, pred, exp)
        title = f'Model: {self.reaction_model.name} | State Residuals'
        time_scale = f'Time [{self.reaction_model.unit_base.time}]'
        fig.update_layout(
                title=title,
                xaxis_title=f'{time_scale}',
                yaxis_title='Residuals',
                )
        
        filename = self._fig_finishing(fig, pred, plot_name=f'{var}-state-residuals')
        return filename

    def _get_proper_unit_str(self, var_data, check_expr=False):
        """Gets the proper units for the charts labels

        :param str var_data: The variable data object (ModelElement)
        :param bool check_expr: Optional checking of expression units

        :return: The units
        :rtype: str

        """
        try:
            str_units = var_data.units.u
        except:
            str_units = var_data.units
            
        if check_expr:
            if var_data.name in self.reaction_model.alg_obj.exprs:
                str_units = str(self.reaction_model.alg_obj.exprs[var_data.name].units)
            else:
                if var_data.name in self.reaction_model.algebraics.names:
                    str_units = str(self.reaction_model.algebraics[var_data.name].units)
            
        return str_units
