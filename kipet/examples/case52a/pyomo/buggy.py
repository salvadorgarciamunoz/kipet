#!/usr/bin/env python
# -*- coding: utf-8 -*-

from kipet.model.TemplateBuilder import *
from kipet.sim.PyomoSimulator import *
from kipet.opt.ParameterEstimator import *
import matplotlib.pyplot as plt

from kipet.utils.data_tools import *
import inspect
import sys
import os, six

if __name__ == "__main__":

    with_plots = True
    if len(sys.argv) == 2:
        if int(sys.argv[1]):
            with_plots = False

    # Load spectral data
    #################################################################################
    dataDirectory = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(
            inspect.currentframe()))), '..', '..', 'data_sets'))
    filename = os.path.join(dataDirectory, 'Dij_case52a.txt')
    D_frame = read_spectral_data_from_txt(filename)

    # build dae block for optimization problems
    #################################################################################
    builder = TemplateBuilder()
    components = {'A': 211.45e-3, 'B': 180.285e-3, 'C': 3.187e-3}
    builder.add_mixture_component(components)

    # note the parameter is not fixed
    builder.add_parameter('k1', bounds=(0.0, 1.0))
    builder.add_spectral_data(D_frame)


    # define explicit system of ODEs
    def rule_odes(m, t):
        exprs = dict()
        exprs['A'] = -m.P['k1'] * m.Z[t, 'A'] * m.Z[t, 'B']
        exprs['B'] = -m.P['k1'] * m.Z[t, 'A'] * m.Z[t, 'B']
        exprs['C'] = m.P['k1'] * m.Z[t, 'A'] * m.Z[t, 'B']
        return exprs


    builder.set_odes_rule(rule_odes)

    pyomo_model = builder.create_pyomo_model(0.0, 200.0)

    optimizer = ParameterEstimator(pyomo_model)

    optimizer.apply_discretization('dae.collocation', nfe=60, ncp=3, scheme='LAGRANGE-RADAU')

    A_set = [l for i, l in enumerate(pyomo_model.meas_lambdas) if (i % 4 == 0)]

    # Provide good initial guess
    p_guess = {'k1': 0.006655}
    raw_results = optimizer.run_lsq_given_P('ipopt', p_guess, tee=False)

    optimizer.initialize_from_trajectory('Z', raw_results.Z)
    optimizer.initialize_from_trajectory('S', raw_results.S)
    optimizer.initialize_from_trajectory('dZdt', raw_results.dZdt)
    optimizer.initialize_from_trajectory('C', raw_results.C)

    options = dict()
    # options['mu_strategy'] = 'adaptive'
    # fixes the variances for now
    sigmas = {'device': 1.94554e-5,
              'A': 2.45887e-6,
              'B': 2.45887e-6,
              'C': 3.1296e-11}
    results_pyomo = optimizer.run_opt('ipopt_sens',
                                      tee=True,
                                      solver_options=options,
                                      variances=sigmas,
                                      tolerance=1e-4,
                                      max_iter=40,
                                      subset_lambdas=A_set,
                                      covariance=True)

    print("The estimated parameters are:")
    for k, v in six.iteritems(results_pyomo.P):
        print(k, v)

    tol = 1e-2
    assert (abs(results_pyomo.P['k1'] - 0.00665) < tol)

    # display results
    if with_plots:
        results_pyomo.C.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile")

        results_pyomo.S.plot.line(legend=True)
        plt.xlabel("Wavelength (cm)")
        plt.ylabel("Absorbance (L/(mol cm))")
        plt.title("Absorbance  Profile")

        plt.show()
