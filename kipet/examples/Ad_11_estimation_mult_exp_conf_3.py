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
from kipet.library.MultipleExperimentsEstimator import *
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
    filename1 =  os.path.join(dataDirectory,'multexp1D.csv')
    filename2 = os.path.join(dataDirectory,'multexp2D.csv')
    D_frame1 = read_spectral_data_from_csv(filename2, negatives_to_zero = True)
    D_frame2 = read_spectral_data_from_csv(filename1, negatives_to_zero = True)

    #This function can be used to remove a certain number of wavelengths from data
    # in this case only every 2nd wavelength is included
    #D_frame1 = decrease_wavelengths(D_frame1,A_set = 2)
    #D_frame1 = baseline_shift(D_frame1)
    #Here we add noise to datasets in order to make our data differenct between experiments
    #D_frame2 = add_noise_to_signal(D_frame2, 0.000000011)
    
    #D_frame2 = decrease_wavelengths(D_frame2,A_set = 2)
    #D_frame2 = baseline_shift(D_frame2)
    D_frame3 = add_noise_to_signal(D_frame2, 0.000002)
    D_frame4 = add_noise_to_signal(D_frame2, 0.0000001)
    D_frame5 = add_noise_to_signal(D_frame1, 0.0000001)
    #################################################################################   

    builder1 = TemplateBuilder()    
    components1 = dict()
    components1['A'] = 1e-3
    components1['B'] = 0.0
    components1['C'] = 0.0
    components1['D'] = 0.0
    components1['E'] = 0

    builder1.add_mixture_component(components1)

    builder1.add_parameter('k1', init=1.5, bounds=(0.1,5)) 
    #There is also the option of providing initial values: Just add init=... as additional argument as above.
    builder1.add_parameter('k2',init = 0.2, bounds=(0.01,2))
    builder1.add_parameter('k3',init =  0.4, bounds=(0.01,0.9))
    # If you have multiple experiments, you need to add your experimental datasets to a dictionary:
    datasets = {'Exp1': D_frame1, 'Exp2': D_frame2, 'Exp3': D_frame3, 'Exp4': D_frame4, 'Exp5': D_frame5}
    # Additionally, we do not add the spectral data to the TemplateBuilder, rather supplying the 
    # TemplateBuilder before data is added as an argument into the function
    
    # define explicit system of ODEs
    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.Z[t,'A']
        exprs['B'] = m.P['k1']*m.Z[t,'A']-m.P['k2']*m.Z[t,'B']
        exprs['C'] = m.P['k2']*m.Z[t,'B']
        exprs['D'] = -m.P['k3']*m.Z[t,'D']
        exprs['E'] = m.P['k3']*m.Z[t,'D']
        return exprs
    
    builder1.set_odes_rule(rule_odes)
    #builder1.bound_profile(var = 'S', bounds = (0, 3))
    #datasets = {'Exp1': D_frame1, 'Exp2': D_frame2}
    builder2 = TemplateBuilder()    
    components2 = {'A':1e-3,'B':0,'C':0, 'D': 1e-3,'E':0}
    builder2.add_mixture_component(components2)
    builder2.add_parameter('k1', init=1.5, bounds=(0.1,10)) 
    #There is also the option of providing initial values: Just add init=... as additional argument as above.
    builder2.add_parameter('k2',init = 0.2, bounds=(0.01,2))
    builder2.add_parameter('k3',init =  0.4, bounds=(0.01,0.9))
    
    # If you have multiple experiments, you need to add your experimental datasets to a dictionary:
    #datasets = {'Exp1': D_frame1, 'Exp2': D_frame2, 'Exp3': D_frame3, 'Exp4': D_frame4}
    # Additionally, we do not add the spectral data to the TemplateBuilder, rather supplying the 
    # TemplateBuilder before data is added as an argument into the function
    
    # define explicit system of ODEs
    '''
    def rule_odes1(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.Z[t,'A']
        exprs['B'] = m.P['k1']*m.Z[t,'A']-m.P['k2']*m.Z[t,'B']-m.P['k3']*m.Z[t,'B']*m.Z[t,'D']
        exprs['C'] = m.P['k2']*m.Z[t,'B']
        exprs['D'] = -m.P['k3']*m.Z[t,'D']*m.Z[t,'B']
        exprs['E'] = m.P['k3']*m.Z[t,'D']*m.Z[t,'B']
        return exprs
    '''
    builder2.set_odes_rule(rule_odes)
    #opt_model = builder.create_pyomo_model(,10.0)
    start_time = {'Exp1':0.0, 'Exp2':0.0, 'Exp3':0.0, 'Exp4':0.0, 'Exp5':0.0}
    end_time = {'Exp1':10.0, 'Exp2':10.0, 'Exp3':10.0, 'Exp4':10.0, 'Exp5':10.0}
    builder2.bound_profile(var = 'S', bounds = (0, 5))
    builder_dict = {'Exp1':builder1, 'Exp2':builder2, 'Exp3':builder2, 'Exp4':builder2, 'Exp5':builder1}
    
    options = dict()
    options['linear_solver'] = 'ma57'
    #options['mu_init']=1e-6
    
    # ============================================================================
    #   USER INPUT SECTION - MULTIPLE EXPERIMENTAL DATASETS       
    # ===========================================================================
    # Here we use the class for Multiple experiments, notice that we add the dictionary
    # Containing the datasets here as an argument
    pest = MultipleExperimentsEstimator(datasets)
    
    nfe = 100
    ncp = 3

    # Now we run the variance estimation on the problem. This is done differently to the
    # single experiment estimation as we now have to solve for variances in each dataset
    # separately these are automatically patched into the main model when parameter estimation is run
    #results_variances = pest.run_variance_estimation(solver = 'ipopt', 
    #                                                 tee=False,
    #                                                 nfe=nfe,
    #                                                 ncp=ncp, 
    #                                                 solver_opts = options,
    #3                                                 start_time=start_time, 
    #3                                                 end_time=end_time, 
    #                                                 builder = builder_dict,
    #                                                 max_iter = 10)
    
    # Finally we run the parameter estimation. This solves each dataset separately first and then
    # links the models and solves it simultaneously
    sigmas = {'A':1e-10,'B':1e-10,'C':1e-10,'device':1e-6}
    sigmas2 = {'A':1e-10,'B':1e-10,'C':1e-10, 'D':1e-10, 'E':1e-10, 'device':1e-6}
    sigmas3 = {'A':1e-10,'B':1e-10,'C':1e-10, 'D':1e-10, 'E':1e-10, 'device':2e-6}
    variances = {'Exp1':sigmas2, 'Exp2':sigmas2, 'Exp3':sigmas3, 'Exp4':sigmas2, 'Exp5':sigmas2}
                 #, 'Exp3':sigmas}
    
    results_pest = pest.run_parameter_estimation(builder = builder_dict,
                                                         solver = 'ipopt_sens', 
                                                         tee=False,
                                                         nfe=nfe,
                                                         ncp=ncp,
                                                         sigma_sq = variances,
                                                         solver_opts = options,
                                                         covariance = True,
                                                         start_time=start_time, 
                                                         end_time=end_time)                                                          
                                                         
    
    # Note here, that with the multiple datasets, we are returning a dictionary cotaining the 
    # results for each block. Since we know that all parameters are shared, we only need to print
    # the parameters from one experiment, however for the plots, they could differ between experiments
    print("The estimated parameters are:")
    #for k,v in six.iteritems(results_pest['Exp1'].P):
    #    print(k, v)
    for k,v in results_pest.items():
        print(results_pest[k].P)
        #print(type(results_pest[k].P))
        #print(k,v)
        #for k,v in results_pest[k].P.items():
        #    print(k,v)
    
    if with_plots:
        for k,v in results_pest.items():

            results_pest[k].C.plot.line(legend=True)
            plt.xlabel("time (s)")
            plt.ylabel("Concentration (mol/L)")
            plt.title("Concentration Profile")
            results_pest[k].Z.plot.line(legend=True)
            plt.xlabel("time (s)")
            plt.ylabel("Concentration (mol/L)")
            plt.title("Concentration Profile")
            results_pest[k].S.plot.line(legend=True)
            plt.xlabel("Wavelength (cm)")
            plt.ylabel("Absorbance (L/(mol cm))")
            plt.title("Absorbance  Profile")
        
            plt.show()
            
