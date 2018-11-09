#  _________________________________________________________________________
#
#  Kipet: Kinetic parameter estimation toolkit
#  Copyright (c) 2016 Eli Lilly.
#  _________________________________________________________________________

# Sample Problem 
# Example from Chen et al. (2016) paper with simulation of ODE system using pyomo discretization and IPOPT
#
#		\frac{dZ_a}{dt} = -k_1*Z_a	                Z_a(0) = 1
#		\frac{dZ_b}{dt} = k_1*Z_a - k_2*Z_b		    Z_b(0) = 0
#       \frac{dZ_c}{dt} = k_2*Z_b	                    Z_c(0) = 0

from kipet.library.TemplateBuilder import *
from kipet.library.PyomoSimulator import *

import matplotlib.pyplot as plt
import sys

if __name__ == "__main__":

    with_plots = True
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False
    
    #=========================================================================
    #USER INPUT SECTION - REQUIRED MODEL BUILDING ACTIONS
    #=========================================================================
    
    # Create Template model (Section 4.2.1 of documentation)
    builder = TemplateBuilder()  
    
    #First we define the components present in the mixture
    builder.add_mixture_component('A',0.3)
    builder.add_mixture_component('B',0.0)
    builder.add_mixture_component('C',0.0)
    builder.add_mixture_component('D',0.01)
    builder.add_mixture_component('E',0.0)
    
    #Following this we add the kinetic parameters
    builder.add_parameter('k1',1.0)
    builder.add_parameter('k2',0.1)
    builder.add_parameter('k3',0.2)
    builder.add_parameter('k4',0.3)
    # define explicit system of ODEs
    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.Z[t,'A']-m.P['k4']*m.Z[t,'A']
        exprs['B'] = m.P['k1']*m.Z[t,'A']-m.P['k2']*m.Z[t,'B']-m.P['k3']*m.Z[t,'B']
        exprs['C'] = m.P['k2']*m.Z[t,'B']-m.P['k4']*m.Z[t,'C']
        exprs['D'] = m.P['k4']*m.Z[t,'A']-m.P['k3']*m.Z[t,'D']
        exprs['E'] = m.P['k3']*m.Z[t,'B']
        
        return exprs

    #Add these ODEs to our model template
    builder.set_odes_rule(rule_odes)

    # create an instance of a pyomo model template
    # the template includes
    #      - Z variables indexed over time and components names e.g. m.Z[t,'A']
    #      - P parameters indexed over the parameter names e.g. m.P['k']
    # The arguments here are the start and end time of the simulation
    pyomo_model = builder.create_pyomo_model(0.0, 20.0)

    #=========================================================================
    # USER INPUT SECTION - SPECIFIC USE SECTION
    #=========================================================================
    # Since in this example we wish to simulate the reaction system defined above,
    # we call the PyomoSimulator class as described in Section 4.2.2 of the documentation
     
    # create instance of simulator with the created pyomo_model as input
    simulator = PyomoSimulator(pyomo_model)
    
    # Then we define the discrete points wanted in the concentration profile and 
    # define our discretization scheme and which collocation roots to deploy
    # further details and advice is included in Section 4.2.2 of the documentation
    simulator.apply_discretization('dae.collocation', ncp = 3, nfe = 100, scheme = 'LAGRANGE-RADAU')

    # Finally we can define our results and run the simulation
    results_pyomo = simulator.run_sim('ipopt',tee=True)
    clean_sig = results_pyomo.Z
    
    noise = np.random.normal(0,0.003,clean_sig.shape)
    sig = clean_sig+noise
    C_Dataframe = pd.DataFrame(data=sig)
    C_Dataframe[C_Dataframe<0]=0
    #for i,r in C_Dataframe.iterrows():
    #    print(i,r) 
    #    if r <=0:
    #        r=0
    
    write_concentration_data_to_csv('new_estim_problem_conc.csv',C_Dataframe)
    C_Dataframe.plot.line(legend=True)
    # display concentration results as in Section 4.2.3
    if with_plots:
        results_pyomo.Z.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile")
        
        plt.show()
