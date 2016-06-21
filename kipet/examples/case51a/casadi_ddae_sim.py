#  _________________________________________________________________________
#
#  Kipet: Kinetic parameter estimation toolkit
#  Copyright (c) 2016 Eli Lilly.
#  _________________________________________________________________________

# Sample Problem 2 (From Sawall et.al.)
# Basic simulation of ODE with spectral data using multistep-integrator 
#
#		\frac{dZ_a}{dt} = -k_1*Z_a	                Z_a(0) = 1
#		\frac{dZ_b}{dt} = k_1*Z_a - k_2*Z_b		Z_b(0) = 0
#               \frac{dZ_c}{dt} = k_2*Z_b	                Z_c(0) = 0
#               Z_k(t_i) = Z_k(t_i) + w(t_i)    for all t_i in measurement points
#               D_{i,j} = \sum_{k=0}^{Nc}C_k(t_i)S(l_j) + \xi_{i,j} for all t_i, for all l_j 

from kipet.model.TemplateBuilder import *
from kipet.sim.CasadiSimulator import *
from kipet.utils.data_tools import *
import matplotlib.pyplot as plt
import os
import sys
import inspect

if __name__ == "__main__":

    with_plots = True
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False
    
    # read 300x100 spectra matrix D_{i,j}
    # this defines the measurement points t_i and l_j as well
    dataDirectory = os.path.abspath(
        os.path.join( os.path.dirname( os.path.abspath( inspect.getfile(
            inspect.currentframe() ) ) ), '..','data_sets'))
    filename =  os.path.join(dataDirectory,'Dij_case51a.csv')
    D_frame = read_spectral_data_from_csv(filename)
    
    # create template model 
    builder = TemplateBuilder()    
    builder.add_mixture_component('A',1e-3)
    builder.add_mixture_component('B',0)
    builder.add_mixture_component('C',0)
    builder.add_parameter('k1',2.0)
    builder.add_parameter('k2',0.2)
    # includes spectra data in the template and defines measurement sets
    builder.add_spectral_data(D_frame)

    # define explicit system of ODEs
    def rule_mass_balances(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.Z[t,'A']
        exprs['B'] = m.P['k1']*m.Z[t,'A']-m.P['k2']*m.Z[t,'B']
        exprs['C'] = m.P['k2']*m.Z[t,'B']
        return exprs
    
    builder.set_mass_balances_rule(rule_mass_balances)

    # create an instance of a casadi model template
    # the template includes
    #   - Z variables indexed over time and components names e.g. m.Z[t,'A']
    #   - C variables indexed over measurement t_i and components names e.g. m.C[t_i,'A']
    #   - P parameters indexed over the parameter names m.P['k']
    #   - D spectra data indexed over the t_i, l_j measurement points m.D[t_i,l_j]
    casadi_model = builder.create_casadi_model(0.0,10.0)

    # create instance of simulator
    sim = CasadiSimulator(casadi_model)
    # defines the discrete points wanted in the profiles (does not include measurement points)
    sim.apply_discretization('integrator',nfe=500)
    # simulate
    results_casadi = sim.run_sim("cvodes")

    # displary concentrations and absorbances results
    if with_plots:
        results_casadi.C.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile")

        results_casadi.S.plot.line(legend=True)
        plt.xlabel("Wavelength (cm)")
        plt.ylabel("Absorbance (L/(mol cm))")
        plt.title("Absorbance  Profile")

        plt.show()    
