
#  _________________________________________________________________________
#
#  Kipet: Kinetic parameter estimation toolkit
#  Copyright (c) 2016 Eli Lilly.
#  _________________________________________________________________________

# Sample Problem 
# Estimation with unknow variancesof spectral data using pyomo discretization 
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
    filename =  os.path.join(dataDirectory,'Dij.txt')
    D_frame = read_spectral_data_from_txt(filename)

    # Then we build dae block for as described in the section 4.2.1. Note the addition
    # of the data using .add_spectral_data
    #################################################################################    
    builder = TemplateBuilder()    
    components = {'A':1e-3,'B':0,'C':0}
    builder.add_mixture_component(components)
    
    builder.add_parameter('k1', init=4.0, bounds=(0.0,5.0)) 
    #There is also the option of providing initial values: Just add init=... as additional argument as above.
    builder.add_parameter('k2',bounds=(0.0,1.0))
    
    # define explicit system of ODEs
    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.Z[t,'A']
        exprs['B'] = m.P['k1']*m.Z[t,'A']-m.P['k2']*m.Z[t,'B']
        exprs['C'] = m.P['k2']*m.Z[t,'B']
        return exprs
    
    builder.set_odes_rule(rule_odes)
    # Notice that to use the wavelength subset optimization we need to use make a copy of the builder
    # before we add the spectral data matrix
    builder_before_data = builder
    
    builder.add_spectral_data(D_frame)
    end_time = 10
    opt_model = builder.create_pyomo_model(0.0,end_time)
    
    #=========================================================================
    #USER INPUT SECTION - VARIANCE ESTIMATION 
    #=========================================================================
    # For this problem we have an input D matrix that has some noise in it
    # We can therefore use the variance estimator described in the Overview section
    # of the documentation and Section 4.3.3
    nfe = 60
    ncp = 3
    
    v_estimator = VarianceEstimator(opt_model)
    v_estimator.apply_discretization('dae.collocation', nfe = nfe, ncp = ncp, scheme = 'LAGRANGE-RADAU')
    
    # It is often requried for larger problems to give the solver some direct instructions
    # These must be given in the form of a dictionary
    options = {}
    # While this problem should solve without changing the deault options, example code is 
    # given commented out below. See Section 5.6 for more options and advice.
    # options['bound_push'] = 1e-8
    # options['tol'] = 1e-9
    
    # The set A_set is then decided. This set, explained in Section 4.3.3 is used to make the
    # variance estimation run faster and has been shown to not decrease the accuracy of the variance 
    # prediction for large noisey data sets.
    A_set = [l for i,l in enumerate(opt_model.meas_lambdas) if (i % 4 == 0)]
    
    # Finally we run the variance estimatator using the arguments shown in Seciton 4.3.3
    results_variances = v_estimator.run_opt('ipopt',
                                            tee=True,
                                            solver_options=options,
                                            tolerance=1e-5,
                                            max_iter=15,
                                            subset_lambdas=A_set)

    # Variances can then be displayed 
    print("\nThe estimated variances are:\n")
    for k,v in six.iteritems(results_variances.sigma_sq):
        print(k, v)

    # and the sigmas for the parameter estimation step are now known and fixed
    sigmas = results_variances.sigma_sq
        
    #=========================================================================
    # USER INPUT SECTION - PARAMETER ESTIMATION 
    #=========================================================================
    # In order to run the parameter estimation we create a pyomo model as described in section 4.3.4
    opt_model = builder.create_pyomo_model(0.0,end_time)

    # and define our parameter estimation problem and discretization strategy
    p_estimator = ParameterEstimator(opt_model)
    p_estimator.apply_discretization('dae.collocation',nfe=nfe,ncp=ncp,scheme='LAGRANGE-RADAU')
    
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
    options['nlp_scaling_method'] = 'user-scaling'

    # finally we run the optimization
    results_pyomo = p_estimator.run_opt('ipopt',
                                      tee=True,
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

    #=========================================================================
    # USER INPUT SECTION - LACK OF FIT and WAVELENGTH SELECTION
    #=========================================================================
    # After solving the full parameter estimation problem above, it is also possible in KIPET
    # to assess the lack of fit (i.e. the amount of mismatch between the data and the model)
    # We do this by calling the following on our ParameterEstimator class
    lof = p_estimator.lack_of_fit()
    
    # This will provide us with the overall lack of fit as a percentage
    # It is also possible to determinine the amount of correlation that each specific wavelength 
    # has with the concentratin profiles. This is done through the wavelength_correlation function
    correlations = p_estimator.wavelength_correlation()
    
    # This outputs a dictionary containing the correlations with the wavelengths.
    # NOTE that in order to graph these, the dictionary (which is unordered) needs to be split
    # and ordered into lists.
    if with_plots:
        lists1 = sorted(correlations.items())
        x1, y1 = zip(*lists1)
        plt.plot(x1,y1)   
        plt.xlabel("Wavelength (cm)")
        plt.ylabel("Correlation between species and wavelength")
        plt.title("Correlation of species and wavelength")
        plt.show()      
    
    # Now that we have the correlations for the wavelengths and the full, reference model, we may want to assess
    # how removing certain wavelengths may improve the fit of our model. To do this, we can run the lof analysis
    # where the problem is solved with different subsets of data. These subsets are filtered so that only the 
    # wavelengths with high correlation remain. The default settings perform the estiamtion for 0.2, 0.4, 0.6, and 
    # 0.8. This means that only wavelengths with correlation above these numbers are considered in the parameter 
    # optimization
    p_estimator.run_lof_analysis(builder_before_data, end_time, correlations, lof, nfe, ncp, sigmas)   
    
    #From this output we can assess which wavelengths, more or less, should be removed in order to obtain the best fit
    # Due to the issues with memory of performing so many optimizations, these models are not stored.
    
    # If the user wishes to search in a particular area of the llof, it is possible to set the step size and range of search
    p_estimator.run_lof_analysis(builder_before_data, end_time, correlations, lof, nfe, ncp, sigmas, step_size = 0.01, search_range = (0, 0.12))
    
    # From this output, it is possible to determine where where, in particular, the best fit is obtained
    # We can now run the parameter estimation based on a particular wavelength correlation filter.
    # We do this by first obtaining a new data matrix with fewer wavelengths included:
    
    new_subs = wavelength_subset_selection(correlations = correlations, n = 0.095)

    # And finally we can run the new parameter estimator with the subset inputted
    results_pyomo = p_estimator.run_param_est_with_subset_lambdas(builder_before_data, end_time, new_subs, nfe, ncp, sigmas)

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