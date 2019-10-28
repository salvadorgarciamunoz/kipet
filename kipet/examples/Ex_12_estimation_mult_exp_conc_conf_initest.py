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
       
    # Load concentration data from the relevant file location. As described in section 4.3.1
    #################################################################################
    dataDirectory = os.path.abspath(
        os.path.join( os.path.dirname( os.path.abspath( inspect.getfile(
            inspect.currentframe() ) ) ), 'data_sets'))
    filename =  os.path.join(dataDirectory,'Ex_1_C_data_withoutA.csv')
    C_frame1 = read_concentration_data_from_csv(filename)
    C_frame2 = add_noise_to_signal(C_frame1, 0.0001)

    #################################################################################    
    builder = TemplateBuilder()
    builder2 = TemplateBuilder()
    components = {'B':0,'C':0}
    builder.add_mixture_component(components)
    builder.add_parameter('k1', init=1.0, bounds=(0.00,10)) 
    #There is also the option of providing initial values: Just add init=... as additional argument as above.
    builder.add_parameter('k2',init = 0.224, bounds=(0.0,10))

    builder2.add_mixture_component(components)
    builder2.add_parameter('k1', init=1.0, bounds=(0.00, 10))
    # There is also the option of providing initial values: Just add init=... as additional argument as above.
    builder2.add_parameter('k2', init=0.224, bounds=(0.0, 10))
    #add complementary state variable:
    extra_states1 = dict()
    extra_states1['A'] = 1e-3
    builder.add_complementary_state_variable(extra_states1)

    extra_states2 = dict()
    extra_states2['A2'] = 1e-3
    builder2.add_complementary_state_variable(extra_states2)
    
    # If you have multiple experiments, you need to add your experimental datasets to a dictionary:
    datasets = {'Exp1': C_frame1, 'Exp2': C_frame2}
    #Notice that we do not add the data to the model as we did in the past, rather we pass this as 
    #an argument into the function later on.
    
    # define explicit system of ODEs
    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.X[t,'A']
        exprs['B'] = m.P['k1']*m.X[t,'A']-m.P['k2']*m.Z[t,'B']
        exprs['C'] = m.P['k2']*m.Z[t,'B']
        return exprs

    def rule_odes2(m,t):
        exprs = dict()
        exprs['A2'] = -m.P['k1']*m.X[t,'A2']
        exprs['B'] = m.P['k1']*m.X[t,'A2']-m.P['k2']*m.Z[t,'B']
        exprs['C'] = m.P['k2']*m.Z[t,'B']
        return exprs
    
    builder.set_odes_rule(rule_odes)
    builder2.set_odes_rule(rule_odes2)
    start_time1=0.0
    start_time2=0.0
    end_time1=10.0
    end_time2=10.0

    modelexp1 = builder.create_pyomo_model(start_time1, end_time1)
    # model4=model3.clone()
    initextra = {'A'}  # to define to be estimated initial values for extra states
    builder.add_init_extra('A', init=1e-3, bounds=(0.0, 0.1))#init=0.97
    builder.set_estinit_extra_species(modelexp1, initextra)  # to define to be estimated initial values for extra states

    builder22 = TemplateBuilder()
    builder22.add_mixture_component(components)
    builder22.add_parameter('k1', init=1.0, bounds=(0.0, 10))
    # There is also the option of providing initial values: Just add init=... as additional argument as above.
    builder22.add_parameter('k2', init=0.224, bounds=(0.0, 10))
    # add complementary state variable:
    builder22.add_complementary_state_variable(extra_states2)

    builder22.set_odes_rule(rule_odes2)
    modelexp2 = builder22.create_pyomo_model(start_time2, end_time2)

    initextra2 = {'A2'}
    builder22.add_init_extra('A2', init=1e-3, bounds=(0.0, 0.1))
    builder22.set_estinit_extra_species(modelexp2, initextra2)  # to define to be estimated initial values for extra states



    #Define our start and end times for the experiment, as we had done in the past, now related to
    #the dataset names
    start_time = {'Exp1':start_time1, 'Exp2':start_time2}
    end_time = {'Exp1':end_time1, 'Exp2':end_time2}
    builder_dict = {'Exp1':builder, 'Exp2':builder22}
    
    options = dict()
    options['linear_solver'] = 'ma57'
    #options['mu_init']=1e-6
    
    # ============================================================================
    #   USER INPUT SECTION - MULTIPLE EXPERIMENTAL DATASETS       
    # ===========================================================================
    # Here we use the class for Multiple experiments, notice that we add the dictionary
    # Containing the datasets here as an argument
    pest = MultipleExperimentsEstimator(datasets)
    
    nfe = 60
    ncp = 3

    sigmas1 = {'device':1e-10,'A':1e-10,'B':1e-10,'C':1e-10}
    sigmas2 = {'device':1e-10,'A2':1e-10,'B':1e-10,'C':1e-10}
    
    variances = {'Exp1':sigmas1, 'Exp2':sigmas2}

    # Different methods for initialization:
    # results_sim = pest.run_simulation(builder = builder_dict,
    #                                                      solver = 'ipopt',
    #                                                      tee=False,
    #                                                      nfe=nfe,
    #                                                      ncp=ncp,
    #                                                      # options=options,
    #                                                      sigma_sq = variances,
    #                                                      #FEsim=True,#FEsim=True,#
    #                                                      solver_opts = options,
    #                                                      # covariance = True,
    #                                                      start_time=start_time,
    #                                                      end_time=end_time)

    # results_var = pest.run_variance_estimation(builder = builder_dict,
    #                                                      solver = 'ipopt',
    #                                                      tee=True,#False,
    #                                                      nfe=nfe,
    #                                                      ncp=ncp,
    #                                                      tolerance=1e-5,
    #                                                      max_iter=15,
    #                                                      # method='alternate',
    #                                                      # secant_point=1e-11,
    #                                                      # initial_sigmas=sigmas, #init_sigmas
    #                                                      # subset_lambdas=A_set,
    #                                                      # options=options,
    #                                                      #sigma_sq = variances,
    #                                                      # FEsim=True,#FEPysim=True,
    #                                                      solver_opts = options,
    #                                                      # covariance = True,
    #                                                      start_time=start_time,
    #                                                      end_time=end_time)


    # Finally we run the parameter estimation. This solves each dataset separately first and then
    # links the models and solves it simultaneously
    results_pest = pest.run_parameter_estimation(solver = 'k_aug', #or ipopt_sens
                                                        tee=True,
                                                         nfe=nfe,
                                                         ncp=ncp,
                                                         covariance = True,
                                                         solver_opts = options,
                                                         start_time=start_time, 
                                                         end_time=end_time,
                                                         spectra_problem = False,
                                                         sigma_sq=variances,
                                                         builder = builder_dict)
    
    # Note here, that with the multiple datasets, we are returning a dictionary cotaining the 
    # results for each block. Since we know that all parameters are shared, we only need to print
    # the parameters from one experiment, however for the plots, they could differ between experiments
    print("The estimated parameters are:")

    for k,v in results_pest.items():
        print(results_pest[k].P)

    for k,v in results_pest.items():
        print(results_pest[k].Pinit)#[0.0,'A'])
    
    if with_plots:
        for k,v in results_pest.items():

            results_pest[k].C.plot.line(legend=True)
            plt.xlabel("time (s)")
            plt.ylabel("Concentration (mol/L)")
            plt.title("Concentration Profile")

            results_pest[k].X.plot.line(legend=True)
            plt.xlabel("time (s)")
            plt.ylabel("Concentration (mol/L)")
            plt.title("Concentration Profile")
    
            results_pest[k].Z.plot.line(legend=True)
            plt.xlabel("time (s)")
            plt.ylabel("Concentration (mol/L)")
            plt.title("Concentration Profile")
        
            plt.show()
            
