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
from kipet.library.PyomoSimulator import *
import matplotlib.pyplot as plt
from pyomo.core import *
import sys

if __name__ == "__main__":

    with_plots = True
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False
    
    # create template model 
    species = {'A':6.7,'B':20.2,'C':0.0}
    params = {'k_p':3.734e7}
    builder = TemplateBuilder(concentrations=species,
                              parameters=params)    
    
    builder.add_complementary_state_variable('T',290.0)

    # define explicit system of ODEs
    def rule_odes(m,t):
        r = m.P['k_p']*exp(-15400.0/(1.987*m.X[t,'T']))*m.Z[t,'A']*m.Z[t,'B']
        T1 = 45650.0*(r*0.01)/28.0
        #T2 = Expr_if(IF=m.X[t,'T']>328.0, THEN=0.0, ELSE=2.0)
        T2 = 1+(328.0-m.X[t,'T'])/((328.0-m.X[t,'T'])**2+1e-5**2)**0.5
        exprs = dict()
        exprs['A'] = -r
        exprs['B'] = -r
        exprs['C'] = r
        exprs['T'] = T1+T2
        return exprs

    builder.set_odes_rule(rule_odes)
    
    # create an instance of a pyomo model template
    # the template includes
    #      - Z variables indexed over time and components names e.g. m.Z[t,'A']
    #      - P parameters indexed over the parameter names e.g. m.P['k']
    pyomo_model = builder.create_pyomo_model(0.0,20.0)

    # create instance of simulator
    simulator = PyomoSimulator(pyomo_model)
    # defines the discrete points wanted in the concentration profile
    simulator.apply_discretization('dae.collocation',nfe=40,ncp=3,scheme='LAGRANGE-RADAU')

    # simulate
    results_pyomo = simulator.run_sim('ipopt',tee=True)

    if with_plots:
        # display concentration results
        results_pyomo.Z.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile")

        
        results_pyomo.X.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Temperature (K)")
        plt.title("Temperature Profile")
        
        plt.show()
