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
    filename1 =  os.path.join(dataDirectory,'ex5data.csv')
    filename2 = os.path.join(dataDirectory,'multexpEx32.csv')
    filename3 =  os.path.join(dataDirectory,'multexpEx33.csv')
    filename4 = os.path.join(dataDirectory,'multexpEx34.csv')
    D_frame1 = read_spectral_data_from_csv(filename1, negatives_to_zero = True)
    D_frame2 = read_spectral_data_from_csv(filename2, negatives_to_zero = True)
    D_frame3 = read_spectral_data_from_csv(filename3, negatives_to_zero = True)
    D_frame4 = read_spectral_data_from_csv(filename4, negatives_to_zero = True)
    #This function can be used to remove a certain number of wavelengths from data
    # in this case only every 2nd wavelength is included
    #D_frame1 = decrease_wavelengths(D_frame1,A_set = 2)
    
    #Here we add noise to datasets in order to make our data differenct between experiments
    #D_frame2 = add_noise_to_signal(D_frame2, 0.005)
    
    #D_frame2 = decrease_wavelengths(D_frame2,A_set = 2)

    #################################################################################    
    builder = TemplateBuilder()    
    builder1 = TemplateBuilder()  
    builder2 = TemplateBuilder()  
    builder3 = TemplateBuilder()  
    
    builder.add_parameter('k1', init=1.5, bounds=(0.3,5)) 
    #There is also the option of providing initial values: Just add init=... as additional argument as above.
    builder.add_parameter('k2',init =17, bounds=(5,50))
    builder.add_parameter('k3', init=0.3, bounds=(0.001,2)) 
    builder.add_parameter('k4', init=0.3, bounds=(0.001,2)) 
    builder1.add_parameter('k1', init=1.5, bounds=(0.3,3)) 
    #There is also the option of providing initial values: Just add init=... as additional argument as above.
    builder1.add_parameter('k2',init =17, bounds=(5,50))
    builder1.add_parameter('k3', init=0.3, bounds=(0.001,2)) 
    builder2.add_parameter('k1', init=1.5, bounds=(0.3,5)) 
    #There is also the option of providing initial values: Just add init=... as additional argument as above.
    builder2.add_parameter('k2',init =17, bounds=(5,50))
    builder2.add_parameter('k3', init=0.3, bounds=(0.001,2)) 
    builder3.add_parameter('k1', init=1.5, bounds=(0.3,5)) 
    #There is also the option of providing initial values: Just add init=... as additional argument as above.
    builder3.add_parameter('k2',init =17, bounds=(5,50))
    builder3.add_parameter('k3', init=0.3, bounds=(0.001,2)) 
    # If you have multiple experiments, you need to add your experimental datasets to a dictionary:
    datasets = {'Exp1': D_frame1, 'Exp2': D_frame2,'Exp3': D_frame3, 'Exp4': D_frame4}

    # Additionally, we do not add the spectral data to the TemplateBuilder, rather supplying the 
    # TemplateBuilder before data is added as an argument into the function
    
    # define explicit system of ODEs
    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.Z[t,'A']
        exprs['B'] = -m.P['k1']*m.Z[t,'A']-m.P['k3']*m.Z[t,'B']
        exprs['C'] = -m.P['k2']*m.Z[t,'C']*m.Z[t,'D'] +m.P['k1']*m.Z[t,'A']
        exprs['D'] = -2*m.P['k2']*m.Z[t,'C']*m.Z[t,'D']
        exprs['E'] = m.P['k2']*m.Z[t,'C']*m.Z[t,'D']
        exprs['F'] = m.P['k3']*m.Z[t,'B']
        return exprs
    
    def rule_odes2(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.Z[t,'A']+m.P['k4']*m.Z[t,'G']
        exprs['B'] = -m.P['k1']*m.Z[t,'A']-m.P['k3']*m.Z[t,'B']
        exprs['C'] = -m.P['k2']*m.Z[t,'C']*m.Z[t,'D'] +m.P['k1']*m.Z[t,'A']
        exprs['D'] = -2*m.P['k2']*m.Z[t,'C']*m.Z[t,'D']
        exprs['E'] = m.P['k2']*m.Z[t,'C']*m.Z[t,'D']
        exprs['F'] = m.P['k3']*m.Z[t,'B']
        exprs['G'] = -m.P['k4']*m.Z[t,'G']
        return exprs
    # Note here that the times or sizes of wavelengths need not be the same
    
    start_time = {'Exp1':0.0, 'Exp2':0.0,'Exp3':0.0, 'Exp4':0.0}

    end_time = {'Exp1':10.0, 'Exp2':10.0,'Exp3':10.0, 'Exp4':10.0}

    components1 = {'A':3e-2,'B':4e-2,'C':0, 'D':2e-2,'E':0,'F':0,'G':0.01}
    components2 = {'A':3e-2,'B':4e-2,'C':0, 'D':2e-2,'E':0,'F':0}
    components3 = {'A':2e-2,'B':6e-2,'C':0, 'D':1e-2,'E':0,'F':0}
    components4 = {'A':2e-2,'B':6e-2,'C':0, 'D':0.0, 'E':0,'F':0}
    
    builder.add_mixture_component(components1)
    builder1.add_mixture_component(components2)
    builder2.add_mixture_component(components3)
    builder3.add_mixture_component(components4)
    
    builder.set_odes_rule(rule_odes2)
    builder.bound_profile(var = 'S', bounds = (0,10))
    builder.bound_profile(var = 'S', comp = 'B', profile_range=(189,191), bounds = (1.5,2.1))
    builder1.set_odes_rule(rule_odes)
    builder1.bound_profile(var = 'S', bounds = (0,10))
    #builder1.bound_profile(var = 'S', comp = 'D', profile_range=(209,211), bounds = (2.5,6))
    builder1.bound_profile(var = 'S', comp = 'B', profile_range=(189,191), bounds = (1.5,2.1))
    builder2.set_odes_rule(rule_odes)
    builder2.bound_profile(var = 'S', bounds = (0,10))
    builder2.bound_profile(var = 'S', comp = 'D', profile_range=(209,211), bounds = (2.5,6))
    #builder2.bound_profile(var = 'S', comp = 'B', profile_range=(189,191), bounds = (1.5,2.1))
    builder3.set_odes_rule(rule_odes)
    builder3.bound_profile(var = 'S', bounds = (0,10))
    #builder3.bound_profile(var = 'S', comp = 'D', profile_range=(209,211), bounds = (2.5,6))
    builder3.bound_profile(var = 'S', comp = 'B', profile_range=(189,191), bounds = (1.5,2.1))
    
    builder5 = {'Exp1':builder, 'Exp2': builder1,'Exp3':builder2, 'Exp4': builder3}
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
    
    # Finally we run the parameter estimation. This solves each dataset separately first and then
    # links the models and solves it simultaneously
    #sigmas = {'A':1e-10,'B':1e-10,'C':1e-10,'device':1.5e-6}
    #sigmas2 = {'A':1e-10,'B':1e-10,'C':1e-10,'device':3e-6}
    #variances = {'Exp1':sigmas, 'Exp2':sigmas}
                                    #1e-08 6e-08
    initsigs = {'Exp1': 1e-08, 'Exp2': 1.5e-08,'Exp3': 1e-08, 'Exp4': 1e-08}
    secantpoint = {'Exp1': 7e-08,'Exp2':5e-08,'Exp3': 6e-08,'Exp4':6e-08}
    tolerance = {'Exp1': 3e-8, 'Exp2': 1e-9,'Exp3': 1e-9, 'Exp4': 1e-9}
    
    # Now we run the variance estimation on the problem. This is done differently to the
    # single experiment estimation as we now have to solve for variances in each dataset
    # separately these are automatically patched into the main model when parameter estimation is run
    results_variances = pest.run_variance_estimation(solver = 'ipopt',
                                                     method = 'alternate',
                                                     initial_sigmas = initsigs,
                                                     secant_point = secantpoint,
                                                     tolerance = tolerance,
                                                     tee=False,
                                                     nfe=nfe,
                                                     ncp=ncp, 
                                                     solver_opts = options,
                                                     start_time=start_time, 
                                                     end_time=end_time, 
                                                     builder = builder5)
    options['linear_solver'] = 'ma57'
    #options['max_iter'] = 10000
    #options['ma57_pivot_order'] = 4
    results_pest = pest.run_parameter_estimation(builder = builder5,
                                                         solver = 'ipopt', 
                                                         tee=True,
                                                         nfe=nfe,
                                                         ncp=ncp,
                                                         shared_spectra = True,
                                                         #sigma_sq = variances,
                                                         solver_opts = options,
                                                         covariance = False,
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

            results_pest[k].C['A'].plot.line(legend=True)
            results_pest[k].C['B'].plot.line(legend=True, linestyle="--")
            results_pest[k].C['C'].plot.line(legend=True, linestyle="-.")
            results_pest[k].C['D'].plot.line(legend=True, linestyle=":")
            results_pest[k].C['E'].plot.line(legend=True, linestyle="-")
            results_pest[k].C['F'].plot.line(legend=True, linestyle="--")
            plt.xlabel("time (s)")
            plt.ylabel("Concentration (mol/L)")
            plt.title("Concentration Profile")
            plt.show()
    
            results_pest[k].S['A'].plot.line(legend=True)
            results_pest[k].S['B'].plot.line(legend=True, linestyle="--")
            results_pest[k].S['C'].plot.line(legend=True, linestyle="-.")
            results_pest[k].S['D'].plot.line(legend=True, linestyle=":")
            results_pest[k].S['E'].plot.line(legend=True, linestyle="-")
            results_pest[k].S['F'].plot.line(legend=True, linestyle="--")
            plt.xlabel("Wavelength (cm)")
            plt.ylabel("Absorbance (L/(mol cm))")
            plt.title("Absorbance  Profile")
        
            plt.show()
            
