"""
Plotting class for KIPET
"""
# Standard library imports
from pathlib import Path
import os
import time
import sys

# Thirdparty library imports 
import plotly.graph_objects as go
import plotly.io as pio

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

class PlotObject():
    
    """This will hold the relevant information needed to make a plot in KIPET"""
    
    def __init__(self, reaction_model=None, jupyter=False):
        
        self.reaction_model = reaction_model
        self.name = reaction_model.name
        self.color_num = 0
        self.filename = None
        self.jupyter = jupyter
        
    def _make_line_trace(self, fig, x, y, name, color):
        """Make line traces
        
        """
        line = dict(color=colors[color], width=4)
        fig.add_trace(
            go.Scatter(x=x,
                       y=y,
                       name=name,
                       line=line,
               )
            )
        return None
    
    def _make_marker_trace(self, fig, x, y, name, color, marker_options):
        """Make marker traces
        
        """
        fig.add_trace(
            go.Scatter(x=x,
                       y=y,
                       name=name,
                       mode='markers',
                       marker={**marker_options, 'color':colors[self.color_num]}),
                    )
        return None

    def _fig_finishing(self, fig, pred, plot_name='Z'):
        """Finish the plots before showing
        
        """
        x_data = [t for t in pred.index]
        x_axis_mod = 0.025*(float(x_data[-1]) - float(x_data[0]))
        fig.update_xaxes(range=[float(x_data[0])-x_axis_mod, float(x_data[-1])+x_axis_mod])
        fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor='#4e4e4e')
        fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='#4e4e4e')

        if self.filename is None:
            t = time.localtime()
            date = f'{t.tm_year}-{t.tm_mon:02}-{t.tm_mday:02}-{t.tm_hour:02}-{t.tm_min:02}-{t.tm_sec:02}'
            # filename = f'{self.name}-{plot_name}-{date}.html'
            filename = f'{plot_name}.html'
            
        if self.jupyter:
            chart_dir = Path.cwd().joinpath('charts', f'{self.name}-{date}')
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
            chart_dir = Path(calling_file_name).joinpath('charts', f'{self.name}-{date}')
            plot_method = pio.write_html
        
        chart_dir.mkdir(parents=True, exist_ok=True)
        filename = chart_dir.joinpath(filename)
        if not self.jupyter:
           print(f'Plot saved as: {filename}')

        plot_method(fig, file=filename.as_posix(), auto_open=True)
    
        return None

    def _state_plot(self, fig, var, pred, exp, use_spectral_format=False):
        """Plot the state profiles
        
        """
        self._make_line_trace(fig=fig,
                              x=pred.index,
                              y=pred[var],
                              name=var,
                              color=self.color_num,
                              )
        marker_options = {'size': 15,
                           'opacity': 0.75,
                          }
        label = 'exp.'   
        if use_spectral_format:
            marker_options = {'size': 10,
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

    def _plot_all_Z(self):
        """Plot all concentration profiles
        
        """
        fig = go.Figure()
        use_spectral_format = False
        pred = getattr(self.reaction_model.results, 'Z')
        if hasattr(self.reaction_model.results, 'Cm'):
            exp = getattr(self.reaction_model.results, 'Cm')
        elif hasattr(self.reaction_model.results, 'C'):
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
        time_scale = f'Time [{self.reaction_model.ub.TIME_BASE}]'

        state_units = self._get_proper_unit_str(var_data)
        # state_units = var_data.units.u
        fig.update_layout(
                title=title,
                xaxis_title=f'{time_scale}',
                yaxis_title=f'{state} [{state_units}]',
                )
        self._fig_finishing(fig, pred, plot_name='all-concentration-profiles')

    def _plot_Z(self, var):
        """Plot state profiles
        
        Args:
            var (str): concentration variable
        
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
        time_scale = f'Time [{self.reaction_model.ub.TIME_BASE}]'
        # state_units = var_data.units.u
        state_units = self._get_proper_unit_str(var_data)
        fig.update_layout(
                title=title,
                xaxis_title=f'{time_scale}',
                yaxis_title=f'{state} [{state_units}]',
                )
        self._fig_finishing(fig, pred, plot_name=f'{var}-concentration-profile')
        
    def _plot_Y(self, var, extra=None):
        """Plot state profiles

        Args:
            var (str): algebraic variable
        
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
        time_scale = f'Time [{self.reaction_model.ub.TIME_BASE}]'
        # state_units = var_data.units
        state_units = self._get_proper_unit_str(var_data, check_expr=True)
        fig.update_layout(
            title=title,
            xaxis_title=f'{time_scale}',
            yaxis_title=f'[{state_units}]',
            )
        self._fig_finishing(fig, pred, plot_name=f'{var}-profile')

    def _plot_X(self, var):
        """Plot state profiles
        
        Args:
            var (str): state variable
        
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
        time_scale = f'Time [{self.reaction_model.ub.TIME_BASE}]'
        state = f'{var_data.description}'.capitalize() if var_data.description is not None else 'State' 
        # state_units = var_data.units
        state_units = self._get_proper_unit_str(var_data)
        fig.update_layout(
                title=title,
                xaxis_title=f'{time_scale}',
                yaxis_title=f'{state} [{state_units}]',
                )
        self._fig_finishing(fig, pred, plot_name=f'{var}-state-profile')
        
    def _plot_all_S(self):
        """Plot all S profiles
        
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
        self._fig_finishing(fig, pred, plot_name='all-absorbance-spectra')

    def _plot_S(self, var):
        """Plot individual S profile
        
        """
        fig = go.Figure()
        pred = getattr(self.reaction_model.results, 'S')
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
        
    def _plot_step(self, var, extra=None):
    
        fig = go.Figure()
        pred = self.reaction_model.results.step
        self._state_plot(fig, var, pred, None)
        title = f'Model: {self.reaction_model.name} | Variable: {var} | Step Function'
        time_scale = f'Time [{self.reaction_model.ub.TIME_BASE}]'
        fig.update_layout(
            title=title,
            xaxis_title=f'{time_scale}',
            yaxis_title=f'[dimensionless]',
            )
        self._fig_finishing(fig, pred, plot_name=f'{var}-step-profile')
        
    
    def _get_proper_unit_str(self, var_data, check_expr=False):
        
        try:
            str_units = var_data.units.u
        except:
            str_units = var_data.units
            
        if check_expr:
            str_units = str(self.reaction_model.alg_obj.exprs[var_data.name].units)
            
        return str_units
        