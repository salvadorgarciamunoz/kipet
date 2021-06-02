"""
KipetModel Mixins
"""
# Standard library imports
import copy
import os
import pathlib
import sys
import time

# Kipet library imports
from kipet.estimator_tools.parameter_estimator import wavelength_subset_selection

# Thirdparty library imports 
import plotly.graph_objects as go
import plotly.io as pio


class WavelengthSelectionMixins:
    
    """Wrapper class mixin of wavelength subset selection methods for ReactionModel"""
    
    def lack_of_fit(self):
        """Wrapper for ParameterEstimator lack_of_fit method

        :return: lof from p_estimator

        """
        lof = self.p_estimator.lack_of_fit()
        return lof
        
    def wavelength_correlation(self, corr_plot=False):
        """Wrapper for wavelength_correlation method in ParameterEstimator
        
        :param bool corr_plot: Option to plot the correlation vs wavelength plot
        
        :return pandas.DataFrame correlations: The correlation data
        
        """
        correlations = self.p_estimator.wavelength_correlation()
    
        if corr_plot:
            
            plot_options = {
                'label_font': dict(
                        size=32,
                    ),
                'title_font': dict(
                        size=32,
                    ),
                'tick_font': dict(
                        size=24,
                    ),
                }
            
            filename = "wavelength-correlations.html"
            fig = go.Figure()
            lists1 = sorted(correlations.items())
            x1, y1 = zip(*lists1)
            
            line = dict(width=4)
            fig.add_trace(
                go.Scatter(x=x1,
                           y=y1,
                           line=line,
                   )
                )
            
            fig.update_layout(
                title="Wavelength Correlations",
                xaxis_title="Wavelength (cm)",
                yaxis_title="Correlation",
                )
            plot_method = pio.write_html 
            
            t = time.localtime()
            date = f'{t.tm_year}-{t.tm_mon:02}-{t.tm_mday:02}-{t.tm_hour:02}-{t.tm_min:02}-{t.tm_sec:02}'
            
            fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor='#4e4e4e', titlefont=plot_options['label_font'], tickfont=plot_options['tick_font'])
            fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='#4e4e4e', titlefont=plot_options['label_font'], tickfont=plot_options['tick_font'])
            
            fig.update_layout(title_font=plot_options['title_font'])
            fig.update_layout(legend_font=plot_options['label_font'])
            
            # if self.jupyter:
            #     chart_dir = Path.cwd().joinpath('charts', f'{self.name}-{folder_name}')
            #     plot_method = pio.show
            #     fig.update_layout(         
            #         autosize=False,
            #         width=1200,
            #         height=800,
            #         margin=dict(
            #             l=50,
            #             r=50,
            #             b=50,
            #             t=50,
            #             pad=4
            #         ),
            #     )
            # else:
            #     calling_file_name = os.path.dirname(os.path.realpath(sys.argv[0]))
            #     chart_dir = Path(calling_file_name).joinpath('charts', f'{self.name}-{folder_name}')
            #     plot_method = pio.write_html

            calling_file_name = os.path.dirname(os.path.realpath(sys.argv[0]))
            chart_dir = pathlib.Path(calling_file_name).joinpath('charts', f'{self.name}-{date}')
            
            chart_dir.mkdir(parents=True, exist_ok=True)
            filename = chart_dir.joinpath(filename)
            print(f'Plot saved as: {filename}')
            
            self.save_static_image = True
            if self.save_static_image:
                fig.write_image(f'{filename}.svg', width=1400, height=900)

            plot_method(fig, file=filename.as_posix(), auto_open=True)

        self.wavelength_correlations = correlations
        return correlations
    
    def run_lof_analysis(self, **kwargs):
        """Wrapper for run_lof_analysis method in ParameterEstimator

        :param dict kwargs: The options to pass through to the ParameterEstimator object

        :return: None

        """
        builder_before_data = copy.copy(self._builder)
        builder_before_data.clear_data()
        
        end_time = self._builder.end_time
        
        correlations = self.wavelength_correlation(corr_plot=False)
        lof = self.lack_of_fit()
        
        nfe = self.settings.collocation.nfe
        ncp = self.settings.collocation.ncp
        sigmas = self.settings.parameter_estimator.variances
        
        self.p_estimator.run_lof_analysis(builder_before_data, end_time, correlations, lof, nfe, ncp, sigmas, **kwargs)

        return None
    
    def wavelength_subset_selection(self, n=0):
        """Wrapper for wavelength_subset_selection method in ParameterEstimator

        :param float n: The subset of wavelengths to select (based on correlation)

        :return: The results of the wavelength_subset_selection method in ParameterEstimator

        """
        if n == 0:
            raise ValueError('You need to choose a subset!')
            
        if not hasattr(self, 'wavelength_correlations'):
            correlations = self.wavelength_correlation(corr_plot=False)
        else:
            correlations = self.wavelength_correlations
        
        return wavelength_subset_selection(correlations=correlations, n=n)
    
    def run_opt_with_subset_lambdas(self, wavelength_subset, **kwargs):
        """Wrapper for run_param_est_with_subset_lambdas method in ParameterEstimator

        :param list wavelength_subset: The subset of chosen wavelengths
        :param dict kwargs: The dict of options to pass through (here the solver)

        :return: Parameter estimation results with the given subset
        :rtype: ResultsObject
        """
        solver = kwargs.pop('solver', 'k_aug')    
    
        builder_before_data = copy.copy(self._builder)
        builder_before_data.clear_data()
        end_time = self.p_estimator.model.end_time.value
        nfe = self.settings.collocation.nfe
        ncp = self.settings.collocation.ncp
        sigmas = self.settings.parameter_estimator.variances
        
        results = self.p_estimator.run_param_est_with_subset_lambdas(builder_before_data, end_time, wavelength_subset, nfe, ncp, sigmas, solver=solver)    
        return results
