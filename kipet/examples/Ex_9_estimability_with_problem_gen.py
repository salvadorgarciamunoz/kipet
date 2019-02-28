
#  _________________________________________________________________________
#
#  Kipet: Kinetic parameter estimation toolkit
#  Copyright (c) 2016 Eli Lilly.
#  _________________________________________________________________________

# Sample Problem 
# This problem atempts to show how a user can generate a simulated problem with
# concentrations of random noise. This problem is then used to test the estimability analysis

from __future__ import print_function
from kipet.library.TemplateBuilder import *
from kipet.library.PyomoSimulator import *
from kipet.library.ParameterEstimator import *
from kipet.library.VarianceEstimator import *
from kipet.library.data_tools import *
from kipet.library.EstimabilityAnalyzer import *
import matplotlib.pyplot as plt
import os
import sys
import inspect
import six
import pandas as pd

if __name__ == "__main__":
    with_plots = True
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False       
    #=========================================================================
    #     SIMULATION MODEL
    #=========================================================================
    # First we will look to generate our data for each of our components in order
    # to build the C-matrix that we desire. We will look to give our species different
    # noise levels in order to also test whether the scaling for multiple levels works
    builder = TemplateBuilder()    
    builder.add_mixture_component('A',0.5)
    builder.add_mixture_component('B',0.0)
    builder.add_mixture_component('C',0.0)
    builder.add_mixture_component('D',0.01)
    builder.add_mixture_component('E',0.0)
    builder.add_mixture_component('F',0.3)
    builder.add_mixture_component('G',0.5)
    builder.add_mixture_component('H',0.0)
    
    #Following this we add the kinetic parameters
    builder.add_parameter('k1',0.3)
    builder.add_parameter('k2',0.1)
    builder.add_parameter('k3',0.1)
    builder.add_parameter('k4',0.4)
    builder.add_parameter('k5',0.02)
    builder.add_parameter('k6',0.5)
    # define explicit system of ODEs
    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.Z[t,'A']-m.P['k4']*m.Z[t,'A']- m.P['k5']*m.Z[t,'E']*m.Z[t,'A']
        exprs['B'] = m.P['k1']*m.Z[t,'A']-m.P['k2']*m.Z[t,'B']-m.P['k3']*m.Z[t,'B']
        exprs['C'] = m.P['k2']*m.Z[t,'B']-m.P['k4']*m.Z[t,'C']
        exprs['D'] = m.P['k4']*m.Z[t,'A']-m.P['k3']*m.Z[t,'D']
        exprs['E'] = m.P['k3']*m.Z[t,'B'] - m.P['k5']*m.Z[t,'E']*m.Z[t,'A']
        
        exprs['F'] = m.P['k5']*m.Z[t,'E']*m.Z[t,'A'] - m.P['k6']*m.Z[t,'G']**2*m.Z[t,'F']
        exprs['G'] = -m.P['k6']*m.Z[t,'G']**2*m.Z[t,'F']
        exprs['H'] = m.P['k6']*m.Z[t,'G']**2*m.Z[t,'F']
        return exprs
    #builder.add_concentration_data(D_frame)
    #Add these ODEs to our model template
    builder.set_odes_rule(rule_odes)
    opt_model = builder.create_pyomo_model(0.0,20.0) 

    # Once the model is described we run the simulator
    
    # call FESimulator
    # FESimulator re-constructs the current TemplateBuilder into fe_factory syntax
    # there is no need to call PyomoSimulator any more as FESimulator is a child class 
    sim = PyomoSimulator(opt_model)
    
    # defines the discrete points wanted in the concentration profile
    sim.apply_discretization('dae.collocation', nfe=50, ncp=3, scheme='LAGRANGE-RADAU')
    
    #this will allow for the fe_factory to run the element by element march forward along 
    #the elements and also automatically initialize the PyomoSimulator model, allowing
    #for the use of the run_sim() function as before. We only need to provide the inputs 
    #to this function as an argument dictionary
    init = sim.run_sim(solver = 'ipopt', tee = True)
    
    if with_plots:
        
        init.Z.plot.line(legend=True)
        plt.xlabel("time (min)")
        plt.ylabel("Concentration ((mol /cm3))")
        plt.title("Concentration  Profile")
        
        plt.show()
        
    print("Simulation is done")
    
    data = add_noise_to_signal(init.Z,0.05)
    if with_plots:
        
        data.plot.line(legend=True)
        plt.xlabel("time (min)")
        plt.ylabel("Concentration ((mol /cm3))")
        plt.title("Concentration  Profile")
        
        plt.show()
    
    write_concentration_data_to_csv('sim_data.csv',data)
    
    # Load spectral data from the file location. 
    #################################################################################

    D_frame = read_concentration_data_from_csv('sim_data.csv')

    # Then we build dae block for as described in the section 4.2.1. Note the addition
    # of the data using .add_concentration_data
    #################################################################################    
    builder = TemplateBuilder()    
    builder.add_mixture_component('A',0.5)
    builder.add_mixture_component('B',0.0)
    builder.add_mixture_component('C',0.0)
    builder.add_mixture_component('D',0.01)
    builder.add_mixture_component('E',0.0)
    builder.add_mixture_component('F',0.3)
    builder.add_mixture_component('G',0.5)
    builder.add_mixture_component('H',0.0)
    
    #Following this we add the kinetic parameters
    builder.add_parameter('k1',bounds=(0.0,1))
    builder.add_parameter('k2',bounds=(0.0,1))
    builder.add_parameter('k3',bounds=(0.0,1))
    builder.add_parameter('k4',bounds=(0.0,1))
    builder.add_parameter('k5',bounds=(0.0,1))
    builder.add_parameter('k6',bounds=(0.0,1))
    # define explicit system of ODEs
    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.Z[t,'A']-m.P['k4']*m.Z[t,'A']- m.P['k5']*m.Z[t,'E']*m.Z[t,'A']
        exprs['B'] = m.P['k1']*m.Z[t,'A']-m.P['k2']*m.Z[t,'B']-m.P['k3']*m.Z[t,'B']
        exprs['C'] = m.P['k2']*m.Z[t,'B']-m.P['k4']*m.Z[t,'C']
        exprs['D'] = m.P['k4']*m.Z[t,'A']-m.P['k3']*m.Z[t,'D']
        exprs['E'] = m.P['k3']*m.Z[t,'B'] - m.P['k5']*m.Z[t,'E']*m.Z[t,'A']
        
        exprs['F'] = m.P['k5']*m.Z[t,'E']*m.Z[t,'A'] - m.P['k6']*m.Z[t,'G']**2*m.Z[t,'F']
        exprs['G'] = -m.P['k6']*m.Z[t,'G']**2*m.Z[t,'F']
        exprs['H'] = m.P['k6']*m.Z[t,'G']**2*m.Z[t,'F']
        
        return exprs
    builder.add_concentration_data(D_frame)
    #Add these ODEs to our model template
    builder.set_odes_rule(rule_odes)
    opt_model = builder.create_pyomo_model(0.0,20.0) 

    #=========================================================================
    # USER INPUT SECTION - ESTIMABILITY ANALYSIS
    #=========================================================================
    # In order to run the estimability analysis we create a pyomo model as described in section 4.3.4

    # Here we use the estimability analysis tools
    e_analyzer = EstimabilityAnalyzer(opt_model)
    # Problem needs to be discretized first
    e_analyzer.apply_discretization('dae.collocation',nfe=60,ncp=1,scheme='LAGRANGE-RADAU')
    # define the uncertainty surrounding each of the parameters
    # This is used for scaling the variables (i.e. 0.01 means that we are sure that the initial 
    # value ofthat parameter is within 1 % of the real value)
    param_uncertainties = {'k1':0.09,'k2':0.01,'k3':0.02,'k4':0.01, 'k5':0.5,'k6':0.8}
    # sigmas, as before, represent the variances in regard to component
    sigmas = {'A':1e-10,'B':1e-10,'C':1e-11, 'D':1e-11,'E':1e-11,'F':1e-11,'G':1e-11,'H':1e-11,'device':3e-9}
    # measurement scaling
    meas_uncertainty = 0.1
    # The rank_params_yao function ranks parameters from most estimable to least estimable 
    # using the method of Yao (2003). Notice the required arguments. Returns a dictionary of rankings.
    listparams = e_analyzer.rank_params_yao(meas_scaling = meas_uncertainty, param_scaling = param_uncertainties, sigmas =sigmas)
    print(listparams)
    # Now we can run the analyzer using the list of ranked parameters
    params_to_select = e_analyzer.run_analyzer(method = 'Wu', parameter_rankings = listparams,meas_scaling = meas_uncertainty, variances =sigmas)
    # We can then use this information to fix certain parameters and run the parameter estimation
    print(params_to_select)
    
    #=========================================================================
    # USER INPUT SECTION - Parameter Estimation
    #=========================================================================
    # Finally we are able to run the parameter estimator on our new model.
    builder = TemplateBuilder()    
    builder.add_mixture_component('A',0.5)
    builder.add_mixture_component('B',0.0)
    builder.add_mixture_component('C',0.0)
    builder.add_mixture_component('D',0.01)
    builder.add_mixture_component('E',0.0)
    builder.add_mixture_component('F',0.3)
    builder.add_mixture_component('G',0.5)
    builder.add_mixture_component('H',0.0)
    
    #Following this we add the kinetic parameters
    builder.add_parameter('k1',bounds=(0.0,1))
    builder.add_parameter('k2',bounds=(0.0,1))
    builder.add_parameter('k3',0.5)
    builder.add_parameter('k4',0.5)
    builder.add_parameter('k5',bounds=(0.0,1))
    builder.add_parameter('k6',bounds=(0.0,1))
    # define explicit system of ODEs
    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.Z[t,'A']-m.P['k4']*m.Z[t,'A']- m.P['k5']*m.Z[t,'E']*m.Z[t,'A']
        exprs['B'] = m.P['k1']*m.Z[t,'A']-m.P['k2']*m.Z[t,'B']-m.P['k3']*m.Z[t,'B']
        exprs['C'] = m.P['k2']*m.Z[t,'B']-m.P['k4']*m.Z[t,'C']
        exprs['D'] = m.P['k4']*m.Z[t,'A']-m.P['k3']*m.Z[t,'D']
        exprs['E'] = m.P['k3']*m.Z[t,'B'] - m.P['k5']*m.Z[t,'E']*m.Z[t,'A']
        
        exprs['F'] = m.P['k5']*m.Z[t,'E']*m.Z[t,'A'] - m.P['k6']*m.Z[t,'G']**2*m.Z[t,'F']
        exprs['G'] = -m.P['k6']*m.Z[t,'G']**2*m.Z[t,'F']
        exprs['H'] = m.P['k6']*m.Z[t,'G']**2*m.Z[t,'F']
        
        return exprs
    builder.add_concentration_data(D_frame)
    #Add these ODEs to our model template
    builder.set_odes_rule(rule_odes)
    opt_model = builder.create_pyomo_model(0.0,20.0) 
    p_estimator = ParameterEstimator(opt_model)
    p_estimator.apply_discretization('dae.collocation',nfe=60,ncp=3,scheme='LAGRANGE-RADAU')
    results_p = p_estimator.run_opt('k_aug',
                                    tee=True,
                                    variances=sigmas,
                                    covariance=True)

    print("The estimated parameters are:")
    for k,v in six.iteritems(opt_model.P):
        print(k, v.value)

    
    if with_plots:
        # display concentration and absorbance results
        results_p.C.plot.line(legend=True)

        plt.xlabel("time (s)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile")

        results_p.Z.plot.line(legend=True)

        plt.xlabel("time (s)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile")
        plt.show()
