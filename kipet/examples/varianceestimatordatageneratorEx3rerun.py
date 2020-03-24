#  _________________________________________________________________________
#
#  Kipet: Kinetic parameter estimation toolkit
#  Copyright (c) 2016 Eli Lilly.
#  _________________________________________________________________________

# Sample Problem
# Tutorial problem data generation for variance
#


from kipet.library.TemplateBuilder import *
from kipet.library.PyomoSimulator import *
from kipet.library.data_tools import *
from kipet.library.ParameterEstimator import *
from kipet.library.VarianceEstimator import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import inspect
from kipet.library.FESimulator import *
#from pyomo.core.base.expr import Expr_if
from pyomo.core import *
from pyomo.opt import *
import pickle
import os
#import decimal
from itertools import count, takewhile

# =========================================================================
# USER INPUT SECTION - Parameters for the absorption generation from Lorentzian parameters
# =========================================================================


def Lorentzian_parameters():
    params_a = dict()
    params_a['alphas'] = [0.8, 0, 1.8, 1.2]
    params_a['betas'] = [150.0, 170.0, 190.0, 250.0]
    params_a['gammas'] = [100.0, 30000.0, 10.0, 100.0]

    params_b = dict()
    params_b['alphas'] = [1.0, 0, 4.0, 1.0]
    params_b['betas'] = [150.0, 170.0, 210.0, 250.0]
    params_b['gammas'] = [100.0, 30000.0, 10.0, 100.0]

    params_c = dict()
    params_c['alphas'] = [0.5, 0, 1.5, 1.0]
    params_c['betas'] = [150.0, 170.0, 220.0, 250.0]
    params_c['gammas'] = [100.0, 30000.0, 10.0, 100.0]

    params_d = dict()
    params_d['alphas'] = [1.2, 0, 2.8, 1.1]
    params_d['betas'] = [150.0, 170.0, 183.0, 250.0]
    params_d['gammas'] = [100.0, 30000.0, 10.0, 100.0]

    params_e = dict()
    params_e['alphas'] = [1, 0, 6.0, 1]
    params_e['betas'] = [150.0, 170.0, 220.0, 250.0]
    params_e['gammas'] = [100.0, 30000.0, 10.0, 100.0]

    params_f = dict()
    params_f['alphas'] = [0.9, 0, 2.1, 1.1]
    params_f['betas'] = [150.0, 170.0, 200.0, 250.0]
    params_f['gammas'] = [100.0, 30000.0, 10.0, 100.0]


    params_g = dict()
    params_g['alphas'] = [0.2, 0, 1.2, 0.2]
    params_g['betas'] = [150.0, 170.0, 205.0, 250.0]
    params_g['gammas'] = [100.0, 30000.0, 20.0, 100.0]


    return {'A': params_d,'B':params_a,'C': params_c, 'D': params_b, 'E': params_g,'F': params_f,}

def frange(start, stop, step):
    return takewhile(lambda x: x< stop, count(start, step))

if __name__ == "__main__":

    with_plots = True
    if len(sys.argv) == 2:
        if int(sys.argv[1]):
            with_plots = False
    # =========================================================================
    # USER INPUT SECTION - MODEL BUILDING - See user guide for how this section functions
    # =========================================================================

    # read S matrix
    wl_span = np.arange(180,230, 1)#1599,1852, 5)
    S_parameters = Lorentzian_parameters()
    S_frame = generate_absorbance_data(wl_span, S_parameters)

    # create template model
    builder = TemplateBuilder()
    
   # components
    components = dict()
    components['A'] = 3e-2
    components['B'] = 4e-2
    components['C'] = 0.0
    components['D'] = 2e-2
    components['E'] = 0.0
    components['F'] = 0.0

    builder.add_mixture_component(components)


    params = dict()
    params['k1'] = 1
    params['k2'] = 20
    params['k3'] = 0.05

    builder.add_parameter(params)

    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.Z[t,'A']
        exprs['B'] = -m.P['k1']*m.Z[t,'A']-m.P['k3']*m.Z[t,'B']
        exprs['C'] = -m.P['k2']*m.Z[t,'C']*m.Z[t,'D'] +m.P['k1']*m.Z[t,'A']
        exprs['D'] = -2*m.P['k2']*m.Z[t,'C']*m.Z[t,'D']
        exprs['E'] = m.P['k2']*m.Z[t,'C']*m.Z[t,'D']
        exprs['F'] = m.P['k3']*m.Z[t,'B']
        return exprs


    builder.set_odes_rule(rule_odes)

    builder.add_measurement_times([i for i in frange(0., 10., 0.2)])
    
    #Add time points where feed as discrete jump should take place:
    #feed_times=[101.035, 303.126]#, 400.
    #builder.add_feed_times(feed_times)

    model = builder.create_pyomo_model(0, 10)
    
    builder.add_absorption_data(S_frame)
    write_absorption_data_to_txt('Sij_varest.txt', S_frame)

    model = builder.create_pyomo_model(0., 10.) 

    #=========================================================================
    #USER INPUT SECTION - FE Factory
    #=========================================================================
     
    # call FESimulator
    # FESimulator re-constructs the current TemplateBuilder into fe_factory syntax
    # there is no need to call PyomoSimulator any more as FESimulator is a child class 
    sim = FESimulator(model)
    
    # defines the discrete points wanted in the concentration profile
    sim.apply_discretization('dae.collocation', nfe=50, ncp=3, scheme='LAGRANGE-RADAU')

    init = sim.call_fe_factory()
    #=========================================================================
    #USER INPUT SECTION - SIMULATION
    #=========================================================================
     
    #: now the final run, just as before
    # simulate
    
    options = {}
    #results_sim = sim.run_sim('ipopt',
                          #tee=True,
                          #solver_opts=options)
    sigmas={'A': 6e-7,
            'B':8e-9,
    'C': 9e-08,
    'D': 1e-08,
    'E': 1e-7,
    'F': 1e-6,
    'device':1e-7}

    results_sim = sim.run_sim('ipopt',variances=sigmas,seed=123453256,
                          tee=True)
    #Load data:
    dataDirectory = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(
        inspect.currentframe()))), 'data_sets'))
    
    D_Dataframe = pd.DataFrame(data=results_sim.D)
    S_Dataframe = pd.DataFrame(data=results_sim.S)
    C_Dataframe = pd.DataFrame(data=results_sim.C)
    Z_Dataframe = pd.DataFrame(data=results_sim.Z)
    dZ_Dataframe = pd.DataFrame(data=results_sim.dZdt)

    #write_absorption_data_to_csv(os.path.join(dataDirectory, 'multexp4S.csv'), S_Dataframe)        
    write_spectral_data_to_csv(os.path.join(dataDirectory, 'varestjytsek27.csv'), D_Dataframe)
    #write_concentration_data_to_csv(os.path.join(dataDirectory, 'multexp4C.csv'), C_Dataframe)
    #write_concentration_data_to_csv(os.path.join(dataDirectory, 'multexp4Z.csv'), Z_Dataframe)
    #write_concentration_data_to_csv(os.path.join(dataDirectory, 'multexp4dZdT'), dZ_Dataframe)
    
    if with_plots:
        #plot_spectral_data(D_Dataframe,dimension='3D')
        results_sim.C['A'].plot.line(legend=True)
        results_sim.C['B'].plot.line(legend=True, linestyle="--")
        results_sim.C['C'].plot.line(legend=True, linestyle="-.")
        results_sim.C['D'].plot.line(legend=True, linestyle=":")
        results_sim.C['E'].plot.line(legend=True, linestyle="-")
        results_sim.C['F'].plot.line(legend=True, linestyle="--")
        plt.xlabel("time (h)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile")
        plt.show()

        results_sim.S['A'].plot.line(legend=True)
        results_sim.S['B'].plot.line(legend=True, linestyle="--")
        results_sim.S['C'].plot.line(legend=True, linestyle="-.")
        results_sim.S['D'].plot.line(legend=True, linestyle=":")
        results_sim.S['E'].plot.line(legend=True, linestyle="-")
        results_sim.S['F'].plot.line(legend=True, linestyle="--")
        plt.xlabel("Wavelength (cm)")# (cm)")
        plt.ylabel("Absorbance (L/(mol cm))")
        plt.title("Absorbance  Profile")


        plt.show()
