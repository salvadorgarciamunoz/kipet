#  _________________________________________________________________________
#
#  Kipet: Kinetic parameter estimation toolkit
#  _________________________________________________________________________

# Sample Problem 
# Estimation with unknow variances of spectral data using pyomo discretization 
#
#		\frac{dZ_a}{dt} = -k_1*Z_a	                Z_a(0) = 1
#		\frac{dZ_b}{dt} = k_1*Z_a - k_2*Z_b		Z_b(0) = 0
#               \frac{dZ_c}{dt} = k_2*Z_b	                Z_c(0) = 0
#               C_k(t_i) = Z_k(t_i) + w(t_i)    for all t_i in measurement points
#               D_{i,j} = \sum_{k=0}^{Nc}C_k(t_i)S(l_j) + \xi_{i,j} for all t_i, for all l_j 
#       Initial concentration 

from __future__ import print_function
from kipet.library.TemplateBuilder import *
from kipet.library.PyomoSimulator import *
from kipet.library.ParameterEstimator import *
from kipet.library.VarianceEstimator import *
from kipet.library.data_tools import *
import matplotlib.pyplot as plt
import os
import sys
import inspect
import six

if __name__ == "__main__":

    with_plots = True
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False
        
    #=========================================================================
    #USER INPUT SECTION - REQUIRED MODEL BUILDING ACTIONS
    #=========================================================================       
    
    # Load spectral data from the relevant file location. As described in section 4.3.1
    #################################################################################
    dataDirectory = os.path.abspath(
        os.path.join( os.path.dirname( os.path.abspath( inspect.getfile(
            inspect.currentframe() ) ) ), 'data_sets'))
    filename =  os.path.join(dataDirectory,'varest2.csv')
    D_frame = read_spectral_data_from_csv(filename, negatives_to_zero = True)
    D_frame[D_frame < 0] = 0
    # Then we build dae block for as described in the section 4.2.1. Note the addition
    # of the data using .add_spectral_data
    #################################################################################    
    builder = TemplateBuilder()    
    components = {'A':1e-2,'B':0,'C':0}
    builder.add_mixture_component(components)
    builder.add_parameter('k1', init=1.2, bounds=(0.01,5.0)) 
    #There is also the option of providing initial values: Just add init=... as additional argument as above.
    builder.add_parameter('k2',init = 0.2, bounds=(0.001,1.0))
    builder.add_spectral_data(D_frame)
    

    # define explicit system of ODEs
    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.Z[t,'A']
        exprs['B'] = m.P['k1']*m.Z[t,'A']-m.P['k2']*m.Z[t,'B']
        exprs['C'] = m.P['k2']*m.Z[t,'B']
        return exprs
    
    builder.set_odes_rule(rule_odes)
    opt_model = builder.create_pyomo_model(0.0,10.0)
    
    #=========================================================================
    #USER INPUT SECTION - VARIANCE ESTIMATION 
    #=========================================================================
    # For this problem we have an input D matrix that has some noise in it
    # We can therefore use the variance estimator described in the Overview section
    # of the documentation and Section 4.3.3 or we can use the alternative methodology,
    # whereby we directly solve for the sigmas given different values of device variance
    #opt_model = kipet_model.model
    
    v_estimator = VarianceEstimator(opt_model)
    v_estimator.apply_discretization('dae.collocation',nfe=50,ncp=3,scheme='LAGRANGE-RADAU')
    
    # It is often requried for larger problems to give the solver some direct instructions
    # These must be given in the form of a dictionary
    options = {}
    # While this problem should solve without changing the deault options, example code is 
    # given commented out below. See Section 5.6 for more options and advice.
    # options['bound_push'] = 1e-8
    # options['tol'] = 1e-9
    options['linear_solver'] = 'ma57'
    
    # The set A_set is then decided. This set, explained in Section 4.3.3 is used to make the
    # variance estimation run faster and has been shown to not decrease the accuracy of the variance 
    # prediction for large noisey data sets.
    #A_set = [l for i,l in enumerate(opt_model.meas_lambdas) if (i % 4 == 0)]
    # For this small problem we do not need this, however it should be set for larger problems
    
    # This is the standard Chen et al. method used within KIPET and added here for comparison with
    # alternative method. Once can see that the results are quite different and it is not obvious which
    # method provides the correct results. On close inspection, the alternative method is closer,
    # however this may be due to the fact that we were able to get the device variance fairly accurately
    
    # This will provide a list of sigma values based on the different delta values evaluated.
    results_variances = v_estimator.run_opt('ipopt',
                                            method = 'Chen',
                                            tee=False,
                                            solver_opts=options,
                                            #num_points = num_points,                                            
                                            tolerance = 1e-10)
                                            #device_range = search_range)
    
    # Variances can then be displayed 
    print("\nThe estimated variances are:\n")
    for k,v in six.iteritems(results_variances.P):
        print(k,v)
    
    sigmas = results_variances.sigma_sq
    for k,v in six.iteritems(sigmas):
        print(k,v)
#%%
    #=========================================================================
    # USER INPUT SECTION - PARAMETER ESTIMATION 
    #=========================================================================
    # In order to run the paramter estimation we create a pyomo model as described in section 4.3.4
    #opt_model = builder.create_pyomo_model(0.0,10.0)
    opt_model = kipet_model.model
    # and define our parameter estimation problem and discretization strategy
    p_estimator = ParameterEstimator(opt_model)
    p_estimator.apply_discretization('dae.collocation',nfe=50,ncp=3,scheme='LAGRANGE-RADAU')
    
    # Certain problems may require initializations and scaling and these can be provided from the 
    # varininace estimation step. This is optional.
    p_estimator.initialize_from_trajectory('Z',results_variances.Z)
    p_estimator.initialize_from_trajectory('S',results_variances.S)
    p_estimator.initialize_from_trajectory('C',results_variances.C)

    # Scaling for Ipopt can also be provided from the variance estimator's solution
    # these details are elaborated on in the manual
    p_estimator.scale_variables_from_trajectory('Z',results_variances.Z)
    p_estimator.scale_variables_from_trajectory('S',results_variances.S)
    p_estimator.scale_variables_from_trajectory('C',results_variances.C)
    
    # Again we provide options for the solver, this time providing the scaling that we set above
    options = dict()
    #options['nlp_scaling_method'] = 'user-scaling'
    options['linear_solver'] = 'ma57'
    # finally we run the optimization
    results_pyomo = p_estimator.run_opt('ipopt_sens',
                                      tee=False,
                                      covariance = True,
                                      solver_opts = options,
                                      variances=sigmas)

    # And display the results
    print("The estimated parameters are:")
    for k,v in six.iteritems(results_pyomo.P):
        print(k, v)
        
    # display results
    if with_plots:
        results_pyomo.C.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile")

        results_pyomo.S.plot.line(legend=True)
        plt.xlabel("Wavelength (cm)")
        plt.ylabel("Absorbance (L/(mol cm))")
        plt.title("Absorbance  Profile")
    
        plt.show()