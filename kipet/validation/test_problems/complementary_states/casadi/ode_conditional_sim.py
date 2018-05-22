#  _________________________________________________________________________
#
#  Kipet: Kinetic parameter estimation toolkit
#  Copyright (c) 2016 Eli Lilly.
#  _________________________________________________________________________

# Sample Problem 2 
# Simulation of ODE system using pyomo discretization and IPOPT
#
#		\frac{dZ_a}{dt} = -k(T)*Z_a*Z_b	        Z_a(0) = 6.7
#		\frac{dZ_b}{dt} = -k(T)*Z_a*Z_b		Z_b(0) = 20.2
#		\frac{dZ_c}{dt} = 2*k(T)*Z_a*Z_b        Z_c(0) = 0.0
#               \frac{dT}{dt} = Tedot + Tsdot           T(0) = 293.2



from kipet.library.TemplateBuilder import *
from kipet.library.CasadiSimulator import *
import matplotlib.pyplot as plt
import casadi as ca
import sys

if __name__ == "__main__":

    with_plots = True
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False

    
    # create template model 
    builder = TemplateBuilder()    
    builder.add_mixture_component('A',6.7)
    builder.add_mixture_component('B',20.0)
    builder.add_mixture_component('C',0.0)
        
    builder.add_complementary_state_variable('T',290.0)
    
    builder.add_parameter('k_p',3.734e7)

    # define explicit system of ODEs
    def rule_odes(m,t):
        r = -m.P['k_p']*ca.exp(-15400.0/(1.987*m.X[t,'T']))*m.Z[t,'A']*m.Z[t,'B']
        T1 = 45650.0*(-r*0.01)/28.0
        #T2 = ca.if_else(m.X[t,'T']>328.0,0.0,2.0)
        T2 = 1+(328.0-m.X[t,'T'])/((328.0-m.X[t,'T'])**2+1e-5**2)**0.5
        exprs = dict()
        exprs['A'] = r
        exprs['B'] = r
        exprs['C'] = -r
        exprs['T'] = T1+T2

        return exprs
    
    builder.set_odes_rule(rule_odes)

    # create an instance of a casadi model template
    casadi_model = builder.create_casadi_model(0.0,20.0)    
    
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

        results_casadi.X.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Temperature (K)")
        plt.title("Temperature Profile")
        
        plt.show()
