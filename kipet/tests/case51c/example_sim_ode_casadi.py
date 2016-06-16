#  _________________________________________________________________________
#
#  Kipet: Kinetic parameter estimation toolkit
#  Copyright (c) 2016 Eli Lilly.
#  _________________________________________________________________________

# Sample Problem 2 (From Sawall et.al.)
# First example from WF paper simulation of ODE system using pyomo discretization and IPOPT
#
#		\frac{dC_a}{dt} = -k_1*C_a*C_b	                                C_a(0) = 1
#		\frac{dC_b}{dt} = -k_1*C_a*C_b                   		C_b(0) = 0.8
#               \frac{dC_c}{dt} = k_1*C_a*C_b-2*k_2*C_c^2	                C_c(0) = 0
#               \frac{dC_d}{dt} = k_2*C_c^2             	                C_c(0) = 0

from kipet.model.TemplateBuilder import *
from kipet.sim.CasadiSimulator import *
import matplotlib.pyplot as plt

if __name__ == "__main__":

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
    
    # create an instance of a casadi model template
    casadi_model = builder.create_casadi_model(0.0,10.0)    

    # create instance of simulator
    sim = CasadiSimulator(casadi_model)
    # defines the discrete points wanted in the concentration profile
    sim.apply_discretization('integrator',nfe=200)
    # simulate
    results_casadi = sim.run_sim("cvodes")

    # display concentration results
    results_casadi.C.plot.line(legend=True)
    plt.xlabel("time (s)")
    plt.ylabel("Concentration (mol/L)")
    plt.title("Concentration Profile")

    plt.show()
