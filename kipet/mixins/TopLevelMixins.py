"""
KipetModel Mixins
"""
# Standard library imports
import copy

# Third party imports

# Kipet library imports
from kipet.core_methods.ParameterEstimator import wavelength_subset_selection


class WavelengthSelectionMixins():
    
    """Wrapper class mixin of wavelength subset selection methods for KipetModel"""
    
    def lack_of_fit(self):
        """Wrapper for ParameterEstimator lack_of_fit method"""
    
        lof = self.p_estimator.lack_of_fit()
        return lof
        
    def wavelength_correlation(self, corr_plot=False):
        """Wrapper for wavelength_correlation method in ParameterEstimator"""
        
        correlations = self.p_estimator.wavelength_correlation()
    
        if corr_plot:
            import matplotlib.pyplot as plt
            
            lists1 = sorted(correlations.items())
            x1, y1 = zip(*lists1)
            plt.plot(x1,y1)   
            plt.xlabel("Wavelength (cm)")
            plt.ylabel("Correlation between species and wavelength")
            plt.title("Correlation of species and wavelength")
            plt.show()   
    
        self.wavelength_correlations = correlations
        return correlations
    
    def run_lof_analysis(self, **kwargs):
        """Wrapper for run_lof_analysis method in ParameterEstimator"""
        
        builder_before_data = copy.copy(self.builder)
        builder_before_data.clear_data()
        
        end_time = self.settings.general.simulation_times[1]
        
        correlations = self.wavelength_correlation(corr_plot=False)
        lof = self.lack_of_fit()
        
        nfe = self.settings.collocation.nfe
        ncp = self.settings.collocation.ncp
        sigmas = self.settings.parameter_estimator.variances
        
        self.p_estimator.run_lof_analysis(builder_before_data, end_time, correlations, lof, nfe, ncp, sigmas, **kwargs)
    
        # Make a dict for the results - why is this not the case?
    
    def wavelength_subset_selection(self, n=0):
        """Wrapper for wavelength_subset_selection method in ParameterEstimator"""
        
        if n == 0:
            raise ValueError('You need to choose a subset!')
            
        if not hasattr(self, 'wavelength_correlations'):
            correlations = self.wavelength_correlation(corr_plot=False)
        else:
            correlations = self.wavelength_correlations
        
        return wavelength_subset_selection(correlations=correlations, n=n)
    
    def run_opt_with_subset_lambdas(self, wavelength_subset, **kwargs):
        """Wrapper for run_param_est_with_subset_lambdas method in ParameterEstimator"""
        
        solver = kwargs.pop('solver', 'k_aug')    
    
        builder_before_data = copy.copy(self.builder)
        builder_before_data.clear_data()
        
        #end_time = self.settings.general.simulation_times[1]
        end_time = self.p_estimator.model.end_time.value
        
        nfe = self.settings.collocation.nfe
        ncp = self.settings.collocation.ncp
        sigmas = self.settings.parameter_estimator.variances
        
        results = self.p_estimator.run_param_est_with_subset_lambdas(builder_before_data, end_time, wavelength_subset, nfe, ncp, sigmas, solver=solver)    
        return results
    