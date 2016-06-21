#  _________________________________________________________________________
#
#  Kipet: Kinetic parameter estimation toolkit
#  Copyright (c) 2016 Eli Lilly.
#  _________________________________________________________________________

# Sample Problem 1 (From Sawall et.al.)
# First example from WF paper simulation of ODE system using multistep-integrator
#
#		\frac{dZ_a}{dt} = -k_1*Z_a	                Z_a(0) = 1
#		\frac{dZ_b}{dt} = k_1*Z_a - k_2*Z_b		Z_b(0) = 0
#               \frac{dZ_c}{dt} = k_2*Z_b	                Z_c(0) = 0


from kipet.model.TemplateBuilder import *
from kipet.sim.CasadiSimulator import *
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
    builder.add_mixture_component('A',1)
    builder.add_mixture_component('B',0)
    builder.add_mixture_component('C',0)
    builder.add_parameter('k1',0.3)
    builder.add_parameter('k2',0.05)

    def rule_mass_balances(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.Z[t,'A']
        exprs['B'] = m.P['k1']*m.Z[t,'A']-m.P['k2']*m.Z[t,'B']
        exprs['C'] = m.P['k2']*m.Z[t,'B']
        return exprs

    builder.set_mass_balances_rule(rule_mass_balances)
    
    # create an instance of a casadi model template
    casadi_model = builder.create_casadi_model(0.0,12.0)    

    # create instance of simulator
    sim = CasadiSimulator(casadi_model)
    # defines the discrete points wanted in the concentration profile
    sim.apply_discretization('integrator',nfe=200)
    # simulate
    results_casadi = sim.run_sim("cvodes")

    # display concentration results
    if with_plots:
        results_casadi.Z.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile")
        
        plt.show()
    
