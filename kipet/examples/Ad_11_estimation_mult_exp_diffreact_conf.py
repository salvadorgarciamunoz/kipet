#  _________________________________________________________________________
#
#  Kipet: Kinetic parameter estimation toolkit
#  Copyright (c) 2016 Eli Lilly.
#  _________________________________________________________________________

# Sample Problem 
# Estimating confidence intervals from multiple experimental datasets with different reactions present

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
    filename1 =  os.path.join(dataDirectory,'multexp4D.csv')
    filename2 = os.path.join(dataDirectory,'multexp5D.csv')
    D_frame1 = read_spectral_data_from_csv(filename1, negatives_to_zero = True)
    D_frame2 = read_spectral_data_from_csv(filename2, negatives_to_zero = True)

    # We wish to add a third dataset that is just a more noisey version of 2
    D_frame3 = add_noise_to_signal(D_frame2, 0.00001)
    
    
    # If you have multiple experiments, you need to add your experimental datasets to a dictionary:
    datasets = {'Exp1': D_frame1, 'Exp2': D_frame2, 'Exp3': D_frame3}
    

    #=========================================================================
    # EXPERIMENT 1
    #=========================================================================
    #Initial conditions
    builder1 = TemplateBuilder()    
    components1 = dict()
    components1['A'] = 1e-3
    components1['B'] = 0.0
    components1['C'] = 0.0
    components1['D'] = 0.0
    components1['E'] = 0

    builder1.add_mixture_component(components1)

    builder1.add_parameter('k1', init=1.5, bounds=(0.001,10)) 
    builder1.add_parameter('k2',init = 0.2, bounds=(0.0001,5))
    builder1.add_parameter('k3',init =  0.4, bounds=(0.3,2))

    # Notice that, although we will not have any reaction for D and E, we still add this equation
    # This model acts as the main model for all experimental datasets
    
    # define explicit system of ODEs
    # DEFINE MODEL FOR ALL EXPERIMENTS
    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.Z[t,'A']
        exprs['B'] = m.P['k1']*m.Z[t,'A']-m.P['k2']*m.Z[t,'B']
        exprs['C'] = m.P['k2']*m.Z[t,'B']
        exprs['D'] = -m.P['k3']*m.Z[t,'D']
        exprs['E'] = m.P['k3']*m.Z[t,'D']
        return exprs

    # As with other multiple experiments KIPET examples, we have different TemplateBuilders for
    # each of the experiments.    
    builder1.set_odes_rule(rule_odes)
    
    # In this example, it is useful to bound the profiles to stop the components with 0 concentration,
    # i.e. D and E, from having very large absorbances. 
    builder1.bound_profile(var = 'S', bounds = (0, 10))
    
    #=========================================================================
    # EXPERIMENT 2 and 3
    #=========================================================================
    builder2 = TemplateBuilder()    
    components2 = {'A':3e-3,'B':0,'C':0, 'D': 1e-3,'E':0}
    builder2.add_mixture_component(components2)
    builder2.add_parameter('k1', init=1.5, bounds=(0.001,10)) 
    #There is also the option of providing initial values: Just add init=... as additional argument as above.
    builder2.add_parameter('k2',init = 0.2, bounds=(0.0001,1))
    builder2.add_parameter('k3',init =  0.6, bounds=(0.3,2))
    
    # Additionally, we do not add the spectral data to the TemplateBuilder, rather supplying the 
    # TemplateBuilder before data is added as an argument into the function
    
    # define explicit system of ODEs
    builder2.set_odes_rule(rule_odes)

    start_time = {'Exp1':0.0, 'Exp2':0.0, 'Exp3':0.0}
    end_time = {'Exp1':10.0, 'Exp2':10.0, 'Exp3':10.0}
    builder_dict = {'Exp1':builder1, 'Exp2':builder2, 'Exp3':builder2}
    
    options = dict()
    options['linear_solver'] = 'ma57'
    #options['mu_init']=1e-6
    
    # ============================================================================
    #   USER INPUT SECTION - MULTIPLE EXPERIMENTAL DATASETS       
    # ===========================================================================
    # Here we use the class for Multiple experiments, notice that we add the dictionary
    # Containing the datasets here as an argument
    pest = MultipleExperimentsEstimator(datasets)
    
    nfe = 50
    ncp = 3
    
    # If we wish to run the variance estimation, it is advised to run it separately on 
    # each experiment in the case of reactions that are present in certain experiments and 
    # not in others. This is because the variance estimator can obtain nonsensical results.
    # It is then advised to only include the reactions that one expects to be active in order 
    # to obtain variances.
    
    # In this simulated example we will just fix the variances

    sigmas2 = {'A':1e-10,'B':1e-10,'C':1e-10, 'D':1e-10, 'E':1e-10, 'device':1e-8}
    variances = {'Exp1':sigmas2, 'Exp2':sigmas2, 'Exp3':sigmas2}

    # Finally we run the parameter estimation. This solves each dataset separately first and then
    # links the models and solves it simultaneously

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
    for k,v in results_pest.items():
        print(results_pest[k].P)

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
            
