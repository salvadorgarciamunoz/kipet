"""Example 8: Using estimability analysis tools

This example currently uses the old EstimabilityAnalyzer style with the new
KipetModel framework 
""" 
# Standard libary imports

# Third party imports

# Kipet library imports
from kipet.kipet import KipetModel
from kipet.library.EstimabilityAnalyzer import *

if __name__ == "__main__":
 
    kipet_model = KipetModel()   
    
    # Add the model parameters
    kipet_model.add_parameter('k1', bounds=(0.1,2))
    kipet_model.add_parameter('k2', bounds=(0.0,2))
    kipet_model.add_parameter('k3', bounds=(0.0,2))
    kipet_model.add_parameter('k4', bounds=(0.0,2))
    
    # Declare the components and give the initial values
    kipet_model.add_component('A', state='concentration', init=0.3)
    kipet_model.add_component('B', state='concentration', init=0.0)
    kipet_model.add_component('C', state='concentration', init=0.0)
    kipet_model.add_component('D', state='concentration', init=0.01)
    kipet_model.add_component('E', state='concentration', init=0.0)
    
    filename = kipet_model.set_directory('new_estim_problem_conc.csv')
    kipet_model.add_dataset('C_frame', category='concentration', file=filename) 
    
    # define explicit system of ODEs
    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.Z[t,'A']-m.P['k4']*m.Z[t,'A']
        exprs['B'] = m.P['k1']*m.Z[t,'A']-m.P['k2']*m.Z[t,'B']-m.P['k3']*m.Z[t,'B']
        exprs['C'] = m.P['k2']*m.Z[t,'B']-m.P['k4']*m.Z[t,'C']
        exprs['D'] = m.P['k4']*m.Z[t,'A']-m.P['k3']*m.Z[t,'D']
        exprs['E'] = m.P['k3']*m.Z[t,'B']
        
        return exprs
    
    kipet_model.add_equations(rule_odes)
    kipet_model.set_times(0, 20)
    kipet_model.create_pyomo_model()

    """
    USER INPUT SECTION - ESTIMABILITY ANALYSIS
    """
    # Here we use the estimability analysis tools
    e_analyzer = EstimabilityAnalyzer(kipet_model.model)
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
