"""Example 2: Abs known non-reacting with new KipetTemplate"""

# Standard library imports
import sys # Only needed for running the example from the command line

# Third party imports
import numpy as np

# Kipet library imports
from kipet.kipet import KipetModel # The only thing you need for using Kipet
from kipet.library.common.prob_gen_tools import generate_absorbance_data
from kipet.library.common.read_write_tools import write_file

def Lorentzian_parameters():
    """Used to generate some spectral data for demonstration purposes
    
    """
    params_d = dict()
    params_d['alphas'] = [0.0,20]
    params_d['betas'] = [1600.0,2200.0]
    params_d['gammas'] = [30000.0,1000.0]
    
    return {'D': params_d}
    
if __name__ == "__main__":

    with_plots = False
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False
        
    # In this example we are taking a known D matrix and adding a solvent that we 
    # already know the absorbance profile of. 
    #=========================================================================
    # PROBLEM DEFINITION / SIMULATION
    #========================================================================= 
    # First we generate the new absorbance of species D
    wl_span = np.arange(1610, 2601, 10)
    S_parameters = Lorentzian_parameters()
    S_frame = generate_absorbance_data(wl_span, S_parameters) 
    write_file('Sij2.txt', S_frame)
    
    # add this new S data to the Dij used in previous example across all times by
    # reading in the old dataset
    
    kipet_model = KipetModel()
    
    # Add the model parameters
    kipet_model.add_parameter('k1', bounds=(0.01, 5.0))
    kipet_model.add_parameter('k2', bounds=(0.01, 5.0))
    
    # Declare the components and give the initial values
    kipet_model.add_component('A', state='concentration', init=1)
    kipet_model.add_component('B', state='concentration', init=0.0)
    kipet_model.add_component('C', state='concentration', init=0.0)
    kipet_model.add_component('D', state='concentration', init=0.001)
    
    # Use this function to replace the old filename set-up
    filename = kipet_model.set_directory('Dijmod.txt')
    kipet_model.add_dataset('D_frame', category='spectral', file=filename)

    # define explicit system of ODEs
    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.Z[t,'A']
        exprs['B'] = m.P['k1']*m.Z[t,'A']-m.P['k2']*m.Z[t,'B']
        exprs['C'] = m.P['k2']*m.Z[t,'B']
        exprs['D'] = 0
        return exprs
    
    kipet_model.add_equations(rule_odes)
    # If no times are given to the builder, it will use the times in the data
    kipet_model.create_pyomo_model()
    
    # Display the KipetTemplate object attributes
    print(kipet_model)

    known_abs = ['D']
    
    kipet_model.set_known_absorbing_species(kipet_model.model, known_abs, S_frame)    
    kipet_model.create_variance_estimator(ncp=3, nfe=60)
    
    # It is often requried for larger problems to give the solver some direct instructions
    # These must be given in the form of a dictionary
    options = {}
    # While this problem should solve without changing the deault options, example code is 
    # given commented out below. See Section 5.6 for more options and advice.
    # options['bound_push'] = 1e-8
    # options['tol'] = 1e-9
    
    # Data reduction to improve speed
    # The set A_set is then decided. This set, explained in Section 4.3.3 is used to make the
    # variance estimation run faster and has been shown to not decrease the accuracy of the variance 
    # prediction for large noisey data sets.
    #A_set = [l for i, l in enumerate(kipet_model.model.meas_lambdas) if (i % 7 == 0)]

    # Finally we run the variance estimatator using the arguments shown in Section 4.3.3
    # New method run_ve_opt to call VarianceEstimator.run_opt()
    
    kipet_model.run_ve_opt('ipopt',
                     tee=True,
                     solver_options=options,
                     tolerance=1e-5,
                     max_iter=15,
                     #method='alternate',
                     #subset_lambdas=A_set
                     )

    # Variances can then be displayed 
    print("\nThe estimated variances are:\n")
    kipet_model.results['ve'].variances

    # and the sigmas for the parameter estimation step are now known and fixed
    sigmas = kipet_model.results['ve'].sigma_sq
    
    if with_plots:
        kipet_model.results['ve'].plot('C', predict=False)
        kipet_model.results['ve'].plot('S', predict=False)
   
    #=========================================================================
    # USER INPUT SECTION - PARAMETER ESTIMATION 
    #=========================================================================
    # In order to run the parameter estimation we create a pyomo model as described in section 4.3.4

    # New method to initialize the parameter estimation problem and discretization strategy
    kipet_model.create_parameter_estimator(ncp=3, nfe=60)
    
    # Certain problems may require initializations and scaling and these can be provided from the 
    # varininace estimation step. This is optional.
    
    # The p_estimator attribute is the ParameterEstimator object from previous versions
    kipet_model.p_estimator.initialize_from_trajectory('Z', kipet_model.results['ve'].Z)
    kipet_model.p_estimator.initialize_from_trajectory('S', kipet_model.results['ve'].S)
    kipet_model.p_estimator.initialize_from_trajectory('C', kipet_model.results['ve'].C)
    
    # Again we provide options for the solver
    options = dict()

    #options['mu_init'] = 1e-6
    #options['bound_push'] =1e-6
    # finally we run the optimization
    
    # New method run_pe_opt to call ParameterEstimator.run_opt()
    kipet_model.run_pe_opt('ipopt',
                    tee=True,
                    solver_opts = options,
                    variances=sigmas)

    # And display the results
    print("The estimated parameters are:")
    kipet_model.results['pe'].parameters
    
    # New plotting methods
    if with_plots:
        kipet_model.results['pe'].plot('C')
        kipet_model.results['pe'].plot('S')
        