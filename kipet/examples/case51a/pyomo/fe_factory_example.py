#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  _________________________________________________________________________
#
#  Kipet: Kinetic parameter estimation toolkit
#  Copyright (c) 2016 Eli Lilly.
#  _________________________________________________________________________

# Sample Problem 
# Estimation with know variances of spectral data using pyomo discretization
#
#		\frac{dZ_a}{dt} = -k_1*Z_a	                Z_a(0) = 1
#		\frac{dZ_b}{dt} = k_1*Z_a - k_2*Z_b		Z_b(0) = 0
#               \frac{dZ_c}{dt} = k_2*Z_b	                Z_c(0) = 0
#               C_k(t_i) = Z_k(t_i) + w(t_i)    for all t_i in measurement points
#               D_{i,j} = \sum_{k=0}^{Nc}C_k(t_i)S(l_j) + \xi_{i,j} for all t_i, for all l_j 



from __future__ import print_function
from __future__ import division
from kipet.model.TemplateBuilder import *
from kipet.sim.PyomoSimulator import *
from kipet.opt.ParameterEstimator import *
import matplotlib.pyplot as plt
from kipet.utils.fe_factory import *

from kipet.utils.data_tools import *
from pyomo.opt import SolverFactory
import inspect
import sys
import os

__author__ = "David Thierry @dthierry" #: April 2018

if __name__ == "__main__":


    with_plots = True
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False
    
    # read 300x100 spectra matrix D_{i,j}
    # this defines the measurement points t_i and l_j as well
    dataDirectory = os.path.abspath(
        os.path.join( os.path.dirname( os.path.abspath( inspect.getfile(
            inspect.currentframe() ) ) ), '..','..','data_sets'))
    filename =  os.path.join(dataDirectory,'Dij.txt')
    D_frame = read_spectral_data_from_txt(filename)
    

    ######################################
    builder = TemplateBuilder()    
    components = {'A': 1e-3,'B': 0,'C': 0}
    builder.add_mixture_component(components)
    # note the parameter is not fixed
    builder.add_parameter('k1',bounds=(0.0,5.0))
    builder.add_parameter('k2',bounds=(0.0,1.0))
    builder.add_spectral_data(D_frame)
    
    # define explicit system of ODEs
    def rule_odes2(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.Z[t,'A']
        exprs['B'] = m.P['k1']*m.Z[t,'A']-m.P['k2']*m.Z[t,'B']
        exprs['C'] = m.P['k2']*m.Z[t,'B']
        return exprs

    builder.set_odes_rule(rule_odes2)

    pyomo_model2 = builder.create_pyomo_model(0.0,10.0)
    src = pyomo_model2.clone()

    optimizer = ParameterEstimator(pyomo_model2)

    optimizer.apply_discretization('dae.collocation',nfe=30,ncp=3,scheme='LAGRANGE-RADAU')
    optimizer.model.time.pprint()

    # Provide good initial guess
    p_guess = {'k1':2.0, 'k2':0.5}

    #: @dthierry: regular stuff for fe_factory
    param_dict = {}
    param_dict["P", "k1"] = 2.0
    param_dict["P", "k2"] = 0.5
    model = optimizer.model
    #: @dthierry: gracefully call fe_factory
    fe_factory = fe_initialize(model, src, init_con="init_conditions_c", param_name="P", param_values=param_dict)
    fe_factory.run()

    optimizer.model.P['k1'].set_value(p_guess['k1'])
    optimizer.model.P['k2'].set_value(p_guess['k2'])
    optimizer.model.P.fix()
    ip = SolverFactory('ipopt')
    ip.solve(optimizer.model, tee=True)
    #: all done
