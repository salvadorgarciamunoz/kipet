#  _________________________________________________________________________
#
#  Kipet: Kinetic parameter estimation toolkit
#  Copyright (c) 2016 Eli Lilly.
#  _________________________________________________________________________

# Sample Problem 2 (From Sawall et.al.)
# Basic simulation of ODE with spectral data using multistep-integrator 
#
#		\frac{dZ_a}{dt} = -k*Z_a	Z_a(0) = 1
#		\frac{dZ_b}{dt} = k*Z_a		Z_b(0) = 0
#
#               C_k(t_i) = Z_k(t_i) + w_k(t_i)    for all t_i in measurement points
#               D_{i,j} = \sum_{k=0}^{Nc}C_k(t_i)S_k(l_j) + \xi_{i,j} for all t_i, for all l_j 


from kipet.model.TemplateBuilder import *
from kipet.sim.CasadiSimulator import *
from kipet.utils.data_tools import *
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":

    # read 200x500 D matrix
    # this defines the measurement points t_i and l_j as well
    filename = 'data_sets{}Dij_basic.txt'.format(os.sep)
    D_frame = read_spectral_data_from_txt(filename)

    # take a look at the data
    plot_spectral_data(D_frame.T)
    plt.xlabel("Wavelength")
    plt.ylabel("Absorption")
    plt.title("Series of spectra (2D plot)")
    plt.show()
    
    
    # create template model 
    builder = TemplateBuilder()    
    builder.add_mixture_component({'A':1,'B':0})
    builder.add_parameter('k',0.01)
    # includes spectra data in the template and defines measurement sets
    builder.add_spectral_data(D_frame)

    # define explicit system of ODEs
    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k']*m.C[t,'A']
        exprs['B'] = m.P['k']*m.C[t,'A']
        return exprs

    builder.set_rule_ode_expressions_dict(rule_odes)

    # create an instance of a casadi model template
    # the template includes
    #   - C variables indexed over time and components names e.g. m.C[t,'A']
    #   - C_noise variables indexed over measurement t_i and components names e.g. m.C_noise[t_i,'A']
    #   - P parameters indexed over the parameter names m.P['k']
    #   - D spectra data indexed over the t_i, l_j measurement points m.D[t_i,l_j]
    casadi_model = builder.create_casadi_model(0.0,200.0)

    # create instance of simulator
    sim = CasadiSimulator(casadi_model)
    # defines the discrete points wanted in the profiles (does not include measurement points)
    sim.apply_discretization('integrator',nfe=500)
    # simulate
    results_casadi = sim.run_sim("cvodes")

    # displary concentrations and absorbances results
    plt.xlabel("time (s)")
    plt.ylabel("Concentration (mol/L)")
    plt.title("Concentration Profile")

    plt.plot(results_casadi.C_noise)

    plt.figure()
    plt.plot(results_casadi.S)
    plt.xlabel("Wavelength (cm)")
    plt.ylabel("Absorbance (L/(mol cm))")
    plt.title("Absorbance  Profile")

    plt.show()

    #write_absorption_data_to_csv("data_sets/Slk_basic.csv",results_casadi.S)
    #write_absorption_data_to_txt("data_sets/Slk_basic.txt",results_casadi.S)
    
    #write_spectral_data_to_csv('Dij_basic.csv',D_frame)
    #red = read_spectral_data_from_csv('Dij_basic.csv')

    #write_spectral_data_to_txt('Dij_basic.txt',D_frame)
    #red2 = read_spectral_data_from_txt('Dij_basic.txt')
    
