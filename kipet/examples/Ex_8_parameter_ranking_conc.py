
#  _________________________________________________________________________
#
#  Kipet: Kinetic parameter estimation toolkit
#  Copyright (c) 2016 Eli Lilly.
#  _________________________________________________________________________

# Sample Problem 
# using simulated data for a concentration-only problem for 5 components and
# 4 kinetic parameters this problem demonstrates the estimability parameter ranking
# algorithm

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
    #USER INPUT SECTION - REQUIRED MODEL BUILDING ACTIONS
    #=========================================================================
       
    
    # Load spectral data from the relevant file location. As described in section 4.3.1
    #################################################################################
    dataDirectory = os.path.abspath(
        os.path.join( os.path.dirname( os.path.abspath( inspect.getfile(
            inspect.currentframe() ) ) ), 'data_sets'))
    filename =  os.path.join(dataDirectory,'new_estim_problem_conc.csv')
    D_frame = read_concentration_data_from_csv(filename)

    # Then we build dae block for as described in the section 4.2.1. Note the addition
    # of the data using .add_concentration_data
    #################################################################################    
    builder = TemplateBuilder()    
    builder.add_mixture_component('A',0.3)
    builder.add_mixture_component('B',0.0)
    builder.add_mixture_component('C',0.0)
    builder.add_mixture_component('D',0.01)
    builder.add_mixture_component('E',0.0)
    
    #Following this we add the kinetic parameters
    builder.add_parameter('k1',bounds=(0.1,1.5))
    builder.add_parameter('k2',bounds=(0.0,0.3))
    builder.add_parameter('k3',bounds=(0.0,0.4))
    builder.add_parameter('k4',bounds=(0.0,0.6))
    # define explicit system of ODEs
    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.Z[t,'A']-m.P['k4']*m.Z[t,'A']
        exprs['B'] = m.P['k1']*m.Z[t,'A']-m.P['k2']*m.Z[t,'B']-m.P['k3']*m.Z[t,'B']
        exprs['C'] = m.P['k2']*m.Z[t,'B']-m.P['k4']*m.Z[t,'C']
        exprs['D'] = m.P['k4']*m.Z[t,'A']-m.P['k3']*m.Z[t,'D']
        exprs['E'] = m.P['k3']*m.Z[t,'B']
        
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
    e_analyzer.apply_discretization('dae.collocation',nfe=60,ncp=3,scheme='LAGRANGE-RADAU')
    # define the uncertainty surrounding each of the parameters
    # This is used for scaling the variables (i.e. 0.01 means that we are sure that the initial 
    # value ofthat parameter is within 1 % of the real value)
    param_uncertainties = {'k1':0.09,'k2':0.01,'k3':0.02,'k4':0.01}
    # sigmas, as before, represent the variances in regard to component
    sigmas = {'A':1e-10,'B':1e-10,'C':1e-11, 'D':1e-11,'E':1e-11,'device':3e-9}
    # measurement scaling
    meas_uncertainty = 0.01
    # The rank_params_yao function ranks parameters from most estimable to least estimable 
    # using the method of Yao (2003). Notice the required arguments. Returns a dictionary of rankings.
    listparams = e_analyzer.rank_params_yao(meas_scaling = meas_uncertainty, param_scaling = param_uncertainties, sigmas =sigmas)

    # We can then use this information to fix certain parameters and run the parameter estimation