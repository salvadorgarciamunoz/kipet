#  _________________________________________________________________________
#
#  Kipet: Kinetic parameter estimation toolkit
#  Copyright (c) 2016 Eli Lilly.
#  _________________________________________________________________________

# Sample Problem 2 (From Sawall et.al.)
# First example from WF paper simulation of ODE system using pyomo discretization and IPOPT
#
#		\frac{dC_a}{dt} = -k_1*C_a*C_b	                                C_a(0) = 1
#		\frac{dC_b}{dt} = -k_1*C_a*C_b                   		C_b(0) = 1
#               \frac{dC_c}{dt} = k_1*C_a*C_b-2*k_2*C_c^2	                C_c(0) = 0
#               \frac{dC_d}{dt} = k_2*C_c^2             	                C_c(0) = 0

from kipet.model.TemplateBuilder import *
from kipet.sim.PyomoSimulator import *
import matplotlib.pyplot as plt
import sys
import os

if __name__ == "__main__":

    with_plots = True
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False
            
    # create template model 
    builder = TemplateBuilder()
    components = {'A':1,'B':0.8,'C':0,'D':0}
    builder.add_mixture_component(components)
    builder.add_parameter('k1',2.0)
    builder.add_parameter('k2',1.0)
    
    # define explicit system of ODEs
    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.C[t,'A']*m.C[t,'B']
        exprs['B'] = -m.P['k1']*m.C[t,'A']*m.C[t,'B']
        exprs['C'] = m.P['k1']*m.C[t,'A']*m.C[t,'B']-2*m.P['k2']*m.C[t,'C']**2
        exprs['D'] = m.P['k2']*m.C[t,'C']**2
        return exprs

    builder.set_rule_ode_expressions_dict(rule_odes)

    # create an instance of a pyomo model template
    # the template includes
    #      - C variables indexed over time and components names e.g. m.C[t,'A']
    #      - P parameters indexed over the parameter names e.g. m.P['k']
    pyomo_model = builder.create_pyomo_model(0.0,10.0)

    # create instance of simulator
    simulator = PyomoSimulator(pyomo_model)
    # defines the discrete points wanted in the concentration profile
    simulator.apply_discretization('dae.collocation',nfe=200,ncp=3,scheme='LAGRANGE-RADAU')

    # simulate
    results_pyomo = simulator.run_sim('ipopt',tee=True)

    # display concentration results
    if with_plots:
        results_pyomo.C.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile")
        
        plt.show()
