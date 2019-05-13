#  _________________________________________________________________________
#
#  Kipet: Kinetic parameter estimation toolkit
#  Copyright (c) 2016 Eli Lilly.
#  _________________________________________________________________________

# Sample Problem
# Christina's project application
# Estimation with known variances of spectral data using pyomo discretization
#
#               C_k(t_i) = Z_k(t_i) + w(t_i)    for all t_i in measurement points
#               D_{i,j} = \sum_{k=0}^{Nc}C_k(t_i)S(l_j) + \xi_{i,j} for all t_i, for all l_j


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
    params_a['alphas'] = [0.8, 0.1, 1.8, 1.2]
    params_a['betas'] = [150.0, 170.0, 200.0, 250.0]
    params_a['gammas'] = [100.0, 30000.0, 100.0, 100.0]

    params_b = dict()
    params_b['alphas'] = [1.0, 0.2, 2.0, 1.0]
    params_b['betas'] = [150.0, 170.0, 200.0, 250.0]
    params_b['gammas'] = [100.0, 30000.0, 100.0, 100.0]

    params_c = dict()
    params_c['alphas'] = [0.5, 0.1, 1.5, 1.0]
    params_c['betas'] = [150.0, 170.0, 200.0, 250.0]
    params_c['gammas'] = [100.0, 30000.0, 100.0, 100.0]

    params_d = dict()
    params_d['alphas'] = [1.2, 0.4, 1.9, 1.1]
    params_d['betas'] = [150.0, 170.0, 200.0, 250.0]
    params_d['gammas'] = [100.0, 30000.0, 100.0, 100.0]

    params_e = dict()
    params_e['alphas'] = [0.4, 0.2, 1.0, 0.4]
    params_e['betas'] = [150.0, 170.0, 200.0, 250.0]
    params_e['gammas'] = [100.0, 30000.0, 100.0, 100.0]

    params_f = dict()
    params_f['alphas'] = [0.9, 0.5, 2.1, 1.1]
    params_f['betas'] = [150.0, 170.0, 200.0, 250.0]
    params_f['gammas'] = [100.0, 30000.0, 100.0, 100.0]

    params_e = dict()
    params_e['alphas'] = [0.4, 0.2, 1.0, 0.4]
    params_e['betas'] = [150.0, 170.0, 200.0, 250.0]
    params_e['gammas'] = [100.0, 30000.0, 100.0, 100.0]

    params_f = dict()
    params_f['alphas'] = [0.9, 0.5, 2.1, 1.1]
    params_f['betas'] = [150.0, 170.0, 200.0, 250.0]
    params_f['gammas'] = [100.0, 30000.0, 100.0, 100.0]

    params_g = dict()
    params_g['alphas'] = [0.2, 0.1, 0.4, 0.2]
    params_g['betas'] = [150.0, 170.0, 200.0, 250.0]
    params_g['gammas'] = [100.0, 30000.0, 100.0, 100.0]


    return {'AH': params_d,'B':params_a,'C': params_c, 'BH+': params_b, 'A-': params_f, 'P':params_e, 'AC-': params_g}

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
    components['AH'] = 0.395555
    components['B'] = 0.0351202
    components['C'] = 0.0
    components['BH+'] = 0.0
    components['A-'] = 0.0
    components['AC-'] = 0.0
    components['P'] = 0.0

    builder.add_mixture_component(components)

    # add algebraics
    algebraics = ['0', '1', '2', '3', '4', '5']  # the indices of the rate rxns
    # note the fifth component. Which basically work as input

    builder.add_algebraic_variable(algebraics)


    params = dict()
    params['k0'] = 49.7796
    params['k1'] = 8.93156
    params['k2'] = 1.31765
    params['k3'] = 0.310870
    params['k4'] = 3.87809

    builder.add_parameter(params)

    # add additional state variables
    extra_states = dict()
    extra_states['V'] = 0.0629418

    builder.add_complementary_state_variable(extra_states)

    # stoichiometric coefficients
    gammas = dict()
    gammas['AH'] = [-1, 0, 0, -1, 0]
    gammas['B'] = [-1, 0, 0, 0, 1]
    gammas['C'] = [0, -1, 1, 0, 0]
    gammas['BH+'] = [1, 0, 0, 0, -1]
    gammas['A-'] = [1, -1, 1, 1, 0]
    gammas['AC-'] = [0, 1, -1, -1, -1]
    gammas['P'] = [0, 0, 0, 1, 1]

    def rule_algebraics(m, t):
        r = list()
        r.append(m.Y[t, '0'] - m.P['k0'] * m.Z[t, 'AH'] * m.Z[t, 'B'])
        r.append(m.Y[t, '1'] - m.P['k1'] * m.Z[t, 'A-'] * m.Z[t, 'C'])
        r.append(m.Y[t, '2'] - m.P['k2'] * m.Z[t, 'AC-'])
        r.append(m.Y[t, '3'] - m.P['k3'] * m.Z[t, 'AC-'] * m.Z[t, 'AH'])
        r.append(m.Y[t, '4'] - m.P['k4'] * m.Z[t, 'AC-'] * m.Z[t, 'BH+'])
        
        return r
    #: there is no ae for Y[t,5] because step equn under rule_odes functions as the switch for the "C" equation

    builder.set_algebraics_rule(rule_algebraics)


    def rule_odes(m, t):
        exprs = dict()
        eta = 1e-2
        step = 0.5 * ((m.Y[t, '5'] + 1) / ((m.Y[t, '5'] + 1) ** 2 + eta ** 2) ** 0.5 + (210.0 - m.Y[t,'5']) / ((210.0 - m.Y[t, '5']) ** 2 + eta ** 2) ** 0.5)
        exprs['V'] = 7.27609e-05 * step
        V = m.X[t, 'V']
        # mass balances
        for c in m.mixture_components:
            exprs[c] = gammas[c][0] * m.Y[t, '0'] + gammas[c][1] * m.Y[t, '1']+gammas[c][2] * m.Y[t, '2']+gammas[c][3] * m.Y[t, '3']+ gammas[c][4] * m.Y[t, '4'] - exprs['V'] / V * m.Z[t, c]
            if c == 'C':
                exprs[c] += 0.02247311828 / (m.X[t, 'V'] * 210) * step
        return exprs


    builder.set_odes_rule(rule_odes)
    builder.add_measurement_times([i for i in frange(0., 600., 2.)])
    
    #Add time points where feed as discrete jump should take place:
    feed_times=[101.035, 303.126]#, 400.
    builder.add_feed_times(feed_times)

    model = builder.create_pyomo_model(0, 600)
    
    builder.add_absorption_data(S_frame)
    write_absorption_data_to_txt('Sij_FEcaseexample5.txt', S_frame)

    model = builder.create_pyomo_model(0., 600.) 

    #=========================================================================
    #USER INPUT SECTION - FE Factory
    #=========================================================================
     
    # call FESimulator
    # FESimulator re-constructs the current TemplateBuilder into fe_factory syntax
    # there is no need to call PyomoSimulator any more as FESimulator is a child class 
    sim = FESimulator(model)
    
    # defines the discrete points wanted in the concentration profile
    sim.apply_discretization('dae.collocation', nfe=50, ncp=3, scheme='LAGRANGE-RADAU')

    #Since the model cannot discriminate inputs from other algebraic elements, we still
    #need to define the inputs as inputs_sub
    inputs_sub = {}
    inputs_sub['Y'] = ['5']
    
    fixedy = True
    yfix={}
    yfix['Y']=['5']#needed in case of different input fixes
    #since these are inputs we need to fix this
    for key in sim.model.time.value:
        sim.model.Y[key, '5'].set_value(key)
        sim.model.Y[key, '5'].fix()

    #this will allow for the fe_factory to run the element by element march forward along 
    #the elements and also automatically initialize the PyomoSimulator model, allowing
    #for the use of the run_sim() function as before. We only need to provide the inputs 
    #to this function as an argument dictionary
    
    #New Inputs for discrete feeds
    Z_step = {'AH': .03} #Which component and which amount is added
    X_step = {'V': .01}
    jump_states = {'Z': Z_step, 'X': X_step}
    jump_points1 = {'AH': 101.035}#Which component is added at which point in time
    jump_points2 = {'V': 303.126}
    jump_times = {'Z': jump_points1, 'X': jump_points2}

    init = sim.call_fe_factory(inputs_sub, jump_states, jump_times, feed_times)
    #=========================================================================
    #USER INPUT SECTION - SIMULATION
    #=========================================================================
     
    #: now the final run, just as before
    # simulate
    
    options = {}
    #results_sim = sim.run_sim('ipopt',
                          #tee=True,
                          #solver_opts=options)
    sigmas={'AH': 1e-10,
            'B':1e-10,
    'C': 1e-10,
    'BH+': 1e-10,
    'A-':1e-10,
    'AC-':1e-10,
    'P':1e-10,
    'device':1e-10}

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

    write_absorption_data_to_csv(
        os.path.join(dataDirectory, 'FeCaseexamplewithoutTemp_S_data_input_noiselesspoints.csv'), S_Dataframe)
    write_spectral_data_to_csv(os.path.join(dataDirectory, 'FeCaseexamplewithoutTemp_D_data_input_noiselesspoints.csv'), D_Dataframe)
    write_concentration_data_to_csv(os.path.join(dataDirectory, 'FeCaseexamplewithoutTemp_C_data_input_noiselesspoints.csv'), C_Dataframe)
    write_concentration_data_to_csv(os.path.join(dataDirectory, 'FeCaseexamplewithoutTemp_Z_data_input_noiselesspoints.csv'), Z_Dataframe)
    write_concentration_data_to_csv(
        os.path.join(dataDirectory, 'FeCaseexamplewithoutTemp_dZ_data_input_noiselesspoints.csv'), dZ_Dataframe)

    if with_plots:

        results_sim.C.plot.line(legend=True)
        plt.xlabel("time (h)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile")

        results_sim.S.plot.line(legend=True)
        plt.xlabel("Wavelength (cm)")# (cm)")
        plt.ylabel("Absorbance (L/(mol cm))")
        plt.title("Absorbance  Profile")


        plt.show()
