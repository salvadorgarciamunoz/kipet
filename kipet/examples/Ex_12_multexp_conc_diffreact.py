
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
from kipet.library.MultipleExperimentsEstimator import *
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
    #     SIMULATION MODEL   1
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
    
    data = add_noise_to_signal(init.Z,0.02)
    if with_plots:
        
        data.plot.line(legend=True)
        plt.xlabel("time (min)")
        plt.ylabel("Concentration ((mol /cm3))")
        plt.title("Concentration  Profile")
        
        plt.show()
    
    write_concentration_data_to_csv('sim_data.csv',data)

    #=========================================================================
    #     SIMULATION MODEL   2
    #=========================================================================
    # First we will look to generate our data for each of our components in order
    # to build the C-matrix that we desire. We will look to give our species different
    # noise levels in order to also test whether the scaling for multiple levels works
    builder1 = TemplateBuilder()    
    builder1.add_mixture_component('A',0.3)
    builder1.add_mixture_component('B',0.0)
    builder1.add_mixture_component('C',0.0)
    builder1.add_mixture_component('D',0.1)
    builder1.add_mixture_component('E',0.0)
    builder1.add_mixture_component('F',0.5)
    builder1.add_mixture_component('G',0.8)
    builder1.add_mixture_component('H',0.0)  
    
    #Following this we add the kinetic parameters
    builder1.add_parameter('k1',0.3)
    builder1.add_parameter('k2',0.1)
    builder1.add_parameter('k3',0.1)
    builder1.add_parameter('k4',0.4)
    builder1.add_parameter('k5',0.02)
    builder1.add_parameter('k6',0.5)
    # define explicit system of ODEs
    #builder.add_concentration_data(D_frame)
    #Add these ODEs to our model template
    builder1.set_odes_rule(rule_odes)
    opt_model1 = builder1.create_pyomo_model(0.0,20.0) 

    # Once the model is described we run the simulator
    
    # call FESimulator
    # FESimulator re-constructs the current TemplateBuilder into fe_factory syntax
    # there is no need to call PyomoSimulator any more as FESimulator is a child class 
    sim1 = PyomoSimulator(opt_model1)
    
    # defines the discrete points wanted in the concentration profile
    sim1.apply_discretization('dae.collocation', nfe=50, ncp=3, scheme='LAGRANGE-RADAU')
    
    #this will allow for the fe_factory to run the element by element march forward along 
    #the elements and also automatically initialize the PyomoSimulator model, allowing
    #for the use of the run_sim() function as before. We only need to provide the inputs 
    #to this function as an argument dictionary
    init1 = sim1.run_sim(solver = 'ipopt', tee = True)
    
    if with_plots:
        
        init1.Z.plot.line(legend=True)
        plt.xlabel("time (min)")
        plt.ylabel("Concentration ((mol /cm3))")
        plt.title("Concentration  Profile")
        
        plt.show()
        
    print("Simulation 2 is done")
    
    data2 = add_noise_to_signal(init1.Z,0.02)
    if with_plots:
        
        data2.plot.line(legend=True)
        plt.xlabel("time (min)")
        plt.ylabel("Concentration ((mol /cm3))")
        plt.title("Concentration  Profile")
        
        plt.show()
    
    write_concentration_data_to_csv('sim_data1.csv',data2)
    
    # Load spectral data from the file location. 
    #################################################################################

    D_frame1 = read_concentration_data_from_csv('sim_data.csv')
    D_frame2 = read_concentration_data_from_csv('sim_data1.csv')

    # Then we build dae block for as described in the section 4.2.1. Note the addition
    # of the data using .add_concentration_data
    #################################################################################  
    
    #=========================================================================
    # USER INPUT SECTION - Parameter Estimation
    #=========================================================================
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
    builder.add_parameter('k5',init=0.2,bounds=(0.0,1))
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
    #builder.add_concentration_data(D_frame1)
    
    #Add these ODEs to our model template
    builder.set_odes_rule(rule_odes)

    builder1 = TemplateBuilder()    

    builder1.add_mixture_component('A',0.3)
    builder1.add_mixture_component('B',0.0)
    builder1.add_mixture_component('C',0.0)
    builder1.add_mixture_component('D',0.1)
    builder1.add_mixture_component('E',0.0)
    builder1.add_mixture_component('F',0.5)
    builder1.add_mixture_component('G',0.8)
    builder1.add_mixture_component('H',0.0)   
    #Following this we add the kinetic parameters
    builder1.add_parameter('k1',init=0.2,bounds=(0.0,1))
    builder1.add_parameter('k2',init=0.2,bounds=(0.0,1))
    builder1.add_parameter('k3',init=0.05,bounds=(0.0,1))
    builder1.add_parameter('k4',init=0.5,bounds=(0.0,1))
    builder1.add_parameter('k5',init=0.2,bounds=(0.0,1))
    builder1.add_parameter('k6',init=0.45,bounds=(0.0,1))

    #builder.add_concentration_data(D_frame2)
    def rule_odes1(m,t):
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
    #Add these ODEs to our model template
    builder1.set_odes_rule(rule_odes1)
    
    start_time = {'Exp1':0.0, 'Exp2':0.0}
    end_time = {'Exp1':20.0, 'Exp2':20.0}    
    datasets = {'Exp1': D_frame1, 'Exp2': D_frame2}
    sigmas = {'A':1e-10,'B':1e-10,'C':1e-11, 'D':1e-11,'E':1e-11,'F':1e-11,'G':1e-11,'H':1e-11,'device':0.02}
    variances = {'Exp1':sigmas, 'Exp2':sigmas}
    builder_dict = {'Exp1':builder, 'Exp2':builder1}

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
    results_pest = pest.run_parameter_estimation(solver = 'ipopt_sens', 
                                                        tee=False,
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
        
            plt.show()
            

