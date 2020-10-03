"""Example 15: Multiple Experimental Datasets and unwanted contributions with
 the new KipetModel
 
"""
# Standard library imports
import sys # Only needed for running the example from the command line

# Third party imports

# Kipet library imports
from kipet.kipet import KipetModel, KipetModelBlock

if __name__ == "__main__":

    with_plots = True
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False
 
    # Create the general model shared amongst datasets   
    kipet_model = KipetModel()
    
    # Add the model parameters
    kipet_model.add_parameter('k1', init=1.3, bounds=(0.0, 2.0))
    kipet_model.add_parameter('k2', init=0.25, bounds=(0.0, 0.5))
    
    # Declare the components and give the initial values
    kipet_model.add_component('A', state='concentration', init=1e-2)
    kipet_model.add_component('B', state='concentration', init=0.0)
    kipet_model.add_component('C', state='concentration', init=0.0)
    
    # define explicit system of ODEs
    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.Z[t,'A']
        exprs['B'] = m.P['k1']*m.Z[t,'A']-m.P['k2']*m.Z[t,'B']
        exprs['C'] = m.P['k2']*m.Z[t,'B']
        return exprs
    
    kipet_model.add_equations(rule_odes)
    
    # Needed for the unwanted contributions
    # kipet_model.builder.add_qr_bounds_init(bounds = (0,None),init = 1.0)
    # kipet_model.builder.add_g_bounds_init(bounds = (0,None))
    kipet_model.test_wrapper()
    
    filename1 = kipet_model.set_directory('Dij_multexp_tiv_G.txt')
    filename2 = kipet_model.set_directory('Dij_multexp_tv_G.txt')
    filename3 = kipet_model.set_directory('Dij_multexp_no_G.txt')
    
    # Model 1
    
    # For clarity, the first model is cloned although this is not necessary
    kipet_model1 = kipet_model.clone(name='Model-1')
    kipet_model1.add_dataset('D_frame1', category='spectral', file=filename1)

    # Set up the parameter estimator
    Ex1_St = dict()
    Ex1_St["r1"] = [-1, 1, 0]
    Ex1_St["r2"] = [0, -1 ,0]

    # Each model has it's own unwanted G settings for the parameter estimator
    kipet_model1.settings.parameter_estimator.G_contribution = 'time_invariant_G'
    kipet_model1.settings.parameter_estimator.St = Ex1_St
    
    # Model 2
    
    # Repeat for the second model - the only difference is the dataset    
    kipet_model2 = kipet_model.clone(name='Model-2')
    kipet_model2.add_dataset('D_frame2', category='spectral', file=filename2)
    kipet_model2.settings.parameter_estimator.G_contribution = 'time_variant_G'

    # Model 3
 
    # Repeat for the third model - the only difference is the dataset    
    kipet_model3 = kipet_model.clone(name='Model-3')
    kipet_model3.add_dataset('D_frame3', category='spectral', file=filename3)

    # Create the KipetModelBlock instance to hold the KipetModels
    model_block = KipetModelBlock()
    model_block.add_model(kipet_model1)
    model_block.add_model(kipet_model2)
    model_block.add_model(kipet_model3)
    
    # Settings
    model_block.settings.general.use_wavelength_subset = False
    
    model_block.settings.solver.linear_solver = 'ma27'
    
    model_block.settings.parameter_estimator.tee = False
    model_block.settings.parameter_estimator.shared_spectra = True
    model_block.settings.parameter_estimator.solver = 'ipopt'
    #model_block.settings.parameter_estimator.covariance = True
    model_block.settings.parameter_estimator.scaled_variance = True
    
    model_block.settings.collocation.nfe = 100

    # Perform the parameter estimation
    model_block.run_opt()
    
    # If you only want to solve each problem individually:
    #model_block.run_opt(multiple_experiments=False)
    
    # Plot the results
    for model, results in model_block.results.items():
        results.show_parameters
        results.plot()
