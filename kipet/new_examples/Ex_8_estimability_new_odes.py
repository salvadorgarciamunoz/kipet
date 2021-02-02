"""Example 8: Using estimability analysis tools

This example currently uses the old EstimabilityAnalyzer style with the new
KipetModel framework 
""" 
# Standard libary imports

# Third party imports

# Kipet library imports
from kipet import KipetModel
from kipet.library.core_methods.EstimabilityAnalyzer import *

if __name__ == "__main__":
 
    kipet_model = KipetModel()   
 
    r1 = kipet_model.new_reaction('reaction-1')   
 
    # Add the model parameters
    r1.add_parameter('k1', bounds=(0.1,2))
    r1.add_parameter('k2', bounds=(0.0,2))
    r1.add_parameter('k3', bounds=(0.0,2))
    r1.add_parameter('k4', bounds=(0.0,2))
    
    # Declare the components and give the initial values
    r1.add_component('A', value=0.3)
    r1.add_component('B', value=0.0)
    r1.add_component('C', value=0.0)
    r1.add_component('D', value=0.01)
    r1.add_component('E', value=0.0)
    
    filename = 'example_data/new_estim_problem_conc.csv'
    r1.add_dataset('C_frame', category='concentration', file=filename) 
    
    c = r1.get_model_vars()
    
    r1.add_ode('A', -c.k1*c.A - c.k4*c.A )
    r1.add_ode('B',  c.k1*c.A - c.k2*c.B - c.k3*c.B )
    r1.add_ode('C',  c.k2*c.B - c.k4*c.C )
    r1.add_ode('A',  c.k4*c.A - c.k3*c.D )
    r1.add_ode('A',  c.k3*c.B )
    
    r1.set_times(0, 20)
    r1.create_pyomo_model()

    """
    USER INPUT SECTION - ESTIMABILITY ANALYSIS
    """
    # Here we use the estimability analysis tools
    e_analyzer = EstimabilityAnalyzer(r1.model)
    # Problem needs to be discretized first
    e_analyzer.apply_discretization('dae.collocation',nfe=60,ncp=1,scheme='LAGRANGE-RADAU')
    # define the uncertainty surrounding each of the parameters
    # This is used for scaling the variables (i.e. 0.01 means that we are sure that the initial 
    # value ofthat parameter is within 1 % of the real value)
    param_uncertainties = {'k1':0.09,'k2':0.01,'k3':0.02,'k4':0.5}
    # sigmas, as before, represent the variances in regard to component
    sigmas = {'A':1e-10,'B':1e-10,'C':1e-11, 'D':1e-11,'E':1e-11,'device':3e-9}
    # measurement scaling
    meas_uncertainty = 0.05
    # The rank_params_yao function ranks parameters from most estimable to least estimable 
    # using the method of Yao (2003). Notice the required arguments. Returns a dictionary of rankings.
    listparams = e_analyzer.rank_params_yao(meas_scaling = meas_uncertainty, param_scaling = param_uncertainties, sigmas =sigmas)
    print(listparams)
    # Now we can run the analyzer using the list of ranked parameters
    params_to_select = e_analyzer.run_analyzer(method = 'Wu', parameter_rankings = listparams,meas_scaling = meas_uncertainty, variances =sigmas)
    # We can then use this information to fix certain parameters and run the parameter estimation
    print(params_to_select)