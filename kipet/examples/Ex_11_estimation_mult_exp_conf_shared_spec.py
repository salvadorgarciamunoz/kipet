#  _________________________________________________________________________
#
#  Kipet: Kinetic parameter estimation toolkit
#  Copyright (c) 2016 Eli Lilly.
#  _________________________________________________________________________

# Sample code showing how to estimate parameters from multiple datasets

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
    filename1 =  os.path.join(dataDirectory,'Dij_exp1.txt')
    filename2 = os.path.join(dataDirectory,'Dij_exp3_reduced.txt')
    D_frame1 = read_spectral_data_from_txt(filename1)
    D_frame2 = read_spectral_data_from_txt(filename2)

    #This function can be used to remove a certain number of wavelengths from data
    # in this case only every 2nd wavelength is included
    D_frame1 = decrease_wavelengths(D_frame1,A_set = 3)
    
    #Here we add noise to datasets in order to make our data differenct between experiments
    D_frame2 = add_noise_to_signal(D_frame2, 0.000000011)
    
    D_frame2 = decrease_wavelengths(D_frame2,A_set = 3)

    #################################################################################    
    builder = TemplateBuilder()    
    components = {'A':1e-3,'B':0,'C':0}
    builder.add_mixture_component(components)
    builder.add_parameter('k1', init=1.0, bounds=(0.00,10)) 
    #There is also the option of providing initial values: Just add init=... as additional argument as above.
    builder.add_parameter('k2',init = 0.224, bounds=(0.0,10))
    
    # If you have multiple experiments, you need to add your experimental datasets to a dictionary:
    datasets = {'Exp1': D_frame1, 'Exp2': D_frame2}

    # Additionally, we do not add the spectral data to the TemplateBuilder, rather supplying the 
    # TemplateBuilder before data is added as an argument into the function
    
    # define explicit system of ODEs
    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.Z[t,'A']
        exprs['B'] = m.P['k1']*m.Z[t,'A']-m.P['k2']*m.Z[t,'B']
        exprs['C'] = m.P['k2']*m.Z[t,'B']
        return exprs
    
    builder.set_odes_rule(rule_odes)

    # Note here that the times or sizes of wavelengths need not be the same
    start_time = {'Exp1':0.0, 'Exp2':0.0}

    end_time = {'Exp1':10.0, 'Exp2':9.0}
    
    options = dict()
    options['linear_solver'] = 'ma27'
    #options['mu_init']=1e-6
    
    # ============================================================================
    #   USER INPUT SECTION - MULTIPLE EXPERIMENTAL DATASETS       
    # ===========================================================================
    # Here we use the class for Multiple experiments, notice that we add the dictionary
    # Containing the datasets here as an argument
    pest = MultipleExperimentsEstimator(datasets)
    
    nfe = 100
    ncp = 3
    
    # Finally we run the parameter estimation. This solves each dataset separately first and then
    # links the models and solves it simultaneously
    sigmas = {'A':1e-10,'B':1e-10,'C':1e-11,'device':2e-6}
    
    variances = {'Exp1':sigmas, 'Exp2':sigmas}
    
    results_pest = pest.run_parameter_estimation(builder = builder,
                                                         solver = 'ipopt_sens', 
                                                         tee=False,
                                                         nfe=nfe,
                                                         ncp=ncp,
                                                         sigma_sq = variances,
                                                         solver_opts = options,
                                                         covariance = True,
                                                         start_time=start_time, 
                                                         end_time=end_time,
                                                         shared_spectra = True)                                                          
                                                         
    
    # Note here, that with the multiple datasets, we are returning a dictionary cotaining the 
    # results for each block. Since we know that all parameters are shared, we only need to print
    # the parameters from one experiment, however for the plots, they could differ between experiments
    print("The estimated parameters are:")

    for k,v in results_pest.items():
        print(results_pest[k].P)
    
    if with_plots:
        for k,v in results_pest.items():

            results_pest[k].C['A'].plot.line(legend=True)
            results_pest[k].C['B'].plot.line(legend=True, linestyle="--")
            results_pest[k].C['C'].plot.line(legend=True, linestyle="-.")
            plt.xlabel("time (s)")
            plt.ylabel("Concentration (mol/L)")
            plt.title("Concentration Profile")
    
            results_pest[k].S['A'].plot.line(legend=True)
            results_pest[k].S['B'].plot.line(legend=True, linestyle="--")
            results_pest[k].S['C'].plot.line(legend=True, linestyle="-.")
            plt.xlabel("Wavelength (cm)")
            plt.ylabel("Absorbance (L/(mol cm))")
            plt.title("Absorbance  Profile")
        
            plt.show()
            
