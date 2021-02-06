"""Example 15: Multiple Experimental Datasets and unwanted contributions with
 the new KipetModel
 
"""
# Standard library imports
import sys # Only needed for running the example from the command line

# Third party imports

# Kipet library imports
from kipet import KipetModel

if __name__ == "__main__":

    with_plots = True
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False
 
    # Create the general model shared amongst datasets   
    kipet_model = KipetModel()
    
    r1 = kipet_model.new_reaction(name='reaction-1')
    
    # Add the parameters
    r1.add_parameter('k1', value=1.3, bounds=(0.0, 2.0))
    r1.add_parameter('k2', value=0.25, bounds=(0.0, 0.5))
    
    # Declare the components and give the initial values
    r1.add_component('A', value=1.0e-2)
    r1.add_component('B', value=0.0)
    r1.add_component('C', value=0.0)
    
    # define explicit system of ODEs
    c = r1.get_model_vars()
    # define explicit system of ODEs
    rates = {}
    rates['A'] = -c.k1 * c.A
    rates['B'] = c.k1 * c.A - c.k2 * c.B
    rates['C'] = c.k2 * c.B
    
    r1.add_odes(rates)
    
    filename1 = 'example_data/Dij_multexp_tiv_G.txt'
    filename2 = 'example_data/Dij_multexp_tv_G.txt'
    filename3 = 'example_data/Dij_multexp_no_G.txt'
    
    # Model 1
    r1.add_data(category='spectral', file=filename1)
    
    # Set up the parameter estimator
    Ex1_St = dict()
    Ex1_St["r1"] = [-1, 1, 0]
    Ex1_St["r2"] = [0, -1 ,0]

    r1.unwanted_contribution('time_invariant_G', St=Ex1_St)
    # Each model has it's own unwanted G settings for the parameter estimato
    r1.settings.general.freq_subset_lambdas = 2
    r1.settings.solver.linear_solver = 'ma57'
    r1.settings.parameter_estimator.solver = 'ipopt'
    r1.settings.parameter_estimator.scaled_variance = False
    r1.settings.parameter_estimator.tee = True
    r1.settings.collocation.ncp = 3
    r1.settings.collocation.nfe = 60
    
    # Model 2
    r2 = kipet_model.new_reaction(name='reaction-2', model=r1)
    r2.add_data(category='spectral', file=filename2)
    r2.unwanted_contribution('time_variant_G')
    
    # Create the other two models
    r3 = kipet_model.new_reaction(name='reaction-3', model=r1)
    r3.add_data(category='spectral', file=filename3)
    
    # Solve the models
    r1.run_opt()
    r2.run_opt()
    r3.run_opt()
    
    # Settings
    kipet_model.settings.general.shared_spectra=True
    # Perform the parameter estimation
   
    kipet_model.run_opt()
 
    # Plot the results
    if with_plots:
        for name, model in kipet_model.models.items():
            kipet_model.results[name].show_parameters
            model.plot()