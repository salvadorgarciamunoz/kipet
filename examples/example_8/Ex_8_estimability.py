"""Example 8: Using estimability analysis tools

This example currently uses the old EstimabilityAnalyzer style with the new
KipetModel framework 
""" 
# Standard libary imports

# Third party imports

# Kipet library imports
from kipet import KipetModel
from kipet.core_methods.EstimabilityAnalyzer import *

if __name__ == "__main__":
 
    kipet_model = KipetModel()   
 
    r1 = kipet_model.new_reaction('reaction-1')   
 
    # Add the model parameters
    k1 = r1.parameter('k1', bounds=(0.1,2))
    k2 = r1.parameter('k2', bounds=(0.0,2))
    k3 = r1.parameter('k3', bounds=(0.0,2))
    k4 = r1.parameter('k4', bounds=(0.0,2))
    
    # Declare the components and give the initial values
    A = r1.component('A', value=0.3)
    B = r1.component('B', value=0.0)
    C = r1.component('C', value=0.0)
    D = r1.component('D', value=0.01)
    E = r1.component('E', value=0.0)
    
    filename = 'data/new_estim_problem_conc.csv'
    r1.add_data('C_frame', category='concentration', file=filename) 
    
    r1.add_ode('A', -k1*A - k4*A )
    r1.add_ode('B',  k1*A - k2*B - k3*B )
    r1.add_ode('C',  k2*B - k4*C )
    r1.add_ode('D',  k4*A - k3*D )
    r1.add_ode('E',  k3*B )
    
    r1.set_times(0, 20)
    r1.create_pyomo_model()

    """
    USER INPUT SECTION - ESTIMABILITY ANALYSIS
    """
    param_uncertainties = {'k1':0.09,'k2':0.01,'k3':0.02,'k4':0.5}
    # sigmas, as before, represent the variances in regard to component
    sigmas = {'A':1e-10,'B':1e-10,'C':1e-11, 'D':1e-11,'E':1e-11,'device':3e-9}
    # measurement scaling
    meas_uncertainty = 0.05
    
    r1.analyze_parameters(method='yao',
                          parameter_uncertainties=param_uncertainties,
                          meas_uncertainty=meas_uncertainty,
                          sigmas=sigmas)
