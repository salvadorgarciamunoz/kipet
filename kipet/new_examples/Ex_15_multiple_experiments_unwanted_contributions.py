"""Example 15: Multiple Experimental Datasets and unwanted contributions with
 the new KipetModel
 
"""
# Standard library imports
import sys # Only needed for running the example from the command line

# Third party imports

# Kipet library imports
from kipet.kipet import KipetModel

if __name__ == "__main__":

    with_plots = True
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False
 
    # Create the general model shared amongst datasets   
    kipet_model = KipetModel()
    
    r1 = kipet_model.new_reaction('reaction-1')
    
    # Add the model parameters
    r1.add_parameter('k1', init=1.3, bounds=(0.0, 2.0))
    r1.add_parameter('k2', init=0.25, bounds=(0.0, 0.5))
    
    # Declare the components and give the initial values
    r1.add_component('A', state='concentration', init=1e-2)
    r1.add_component('B', state='concentration', init=0.0)
    r1.add_component('C', state='concentration', init=0.0)
    
    # define explicit system of ODEs
    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.Z[t,'A']
        exprs['B'] = m.P['k1']*m.Z[t,'A']-m.P['k2']*m.Z[t,'B']
        exprs['C'] = m.P['k2']*m.Z[t,'B']
        return exprs
    
    r1.add_equations(rule_odes)
    
    filename1 = 'Dij_multexp_tiv_G.txt'
    filename2 = 'Dij_multexp_tv_G.txt'
    filename3 = 'Dij_multexp_no_G.txt'
    
    
    # Create the other two models
    r2 = kipet_model.new_reaction(name='reaction-2', model_to_clone=r1, items_not_copied='datasets')
    r3 = kipet_model.new_reaction(name='reaction-3', model_to_clone=r1, items_not_copied='datasets')
    
    # Model 1
    r1.add_dataset(category='spectral', file=filename1)

    # Set up the parameter estimator
    Ex1_St = dict()
    Ex1_St["r1"] = [-1, 1, 0]
    Ex1_St["r2"] = [0, -1 ,0]

    # Each model has it's own unwanted G settings for the parameter estimator
    r1.settings.parameter_estimator.G_contribution = 'time_invariant_G'
    r1.settings.parameter_estimator.St = Ex1_St
    
    # Model 2 
    r2.add_dataset(category='spectral', file=filename2)
    r2.settings.parameter_estimator.G_contribution = 'time_variant_G'

    # Model 3  
    r3.add_dataset(category='spectral', file=filename3)
    
    # Settings
    kipet_model.settings.general.use_wavelength_subset = False
    kipet_model.settings.solver.linear_solver = 'ma57'
    kipet_model.settings.parameter_estimator.shared_spectra = True
    kipet_model.settings.parameter_estimator.solver = 'ipopt'
    kipet_model.settings.parameter_estimator.scaled_variance = False
    kipet_model.settings.parameter_estimator.tee = True
    kipet_model.settings.collocation.nfe = 100

    # Perform the parameter estimation
    kipet_model.run_opt()

    # Plot the results
    for model, results in kipet_model.results.items():
        results.show_parameters
        results.plot(show_plot=with_plots)