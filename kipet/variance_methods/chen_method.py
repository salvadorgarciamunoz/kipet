"""
Initialization for method from Chen et al. 2016 
"""
# Standard library imports
import os

# Third party imports
import numpy as np
import pandas as pd
from pyomo.environ import Objective, SolverFactory
from scipy.optimize import least_squares

# KIPET library imports
from kipet.calculation_tools.beer_lambert import S_from_DC
from kipet.estimator_tools.results_object import ResultsObject
from kipet.variance_methods.chen_method_pyomo import solve_C, solve_S, solve_Z
from kipet.variance_methods.chen_method_scipy import (build_c_model,
                                                      build_s_model, solve_c_scipy, solve_s_scipy)


def run_method(var_est_object, solver, run_opt_kwargs):
    """This is the original method for estimating variances from Chen et
    al. 2016

    This is an improved method for determining the component variances. This method has been removed from the
    VarianceEstimator class for simplification.

    :param VarianceEstimator var_est_object: The variance estimation object
    :param str solver: The solver being used (currently not used)
    :param dict run_opt_kwargs: The dict of user settings passed on from the ReactionModel

    :return results: The results from the variance estimation
    :rtype: ResultsObject

    """
    solver_opts = run_opt_kwargs.pop('solver_opts', dict())
    max_iter = run_opt_kwargs.pop('max_iter', 400)
    tol = run_opt_kwargs.pop('tolerance', 5.0e-5)
    tee = run_opt_kwargs.pop('tee', False)
    norm_order = run_opt_kwargs.pop('norm', np.inf)
    A = run_opt_kwargs.pop('subset_lambdas', None)
    init_C = run_opt_kwargs.pop('init_C', None)
    lsq_ipopt = run_opt_kwargs.pop('lsq_ipopt', False)
    species_list = run_opt_kwargs.pop('subset_components', None)
    fixed_device_var = run_opt_kwargs.pop('fixed_device_variance', None)

    if init_C is None:
        solve_initalization(var_est_object, 
                            solver,
                            subset_lambdas=A,
                            solver_opts=solver_opts,
                            tee=tee)
    else:
        for t in var_est_object.model.times_spectral:
            for k in var_est_object.comps['unknown_absorbance']:
                var_est_object.model.C[t, k].value = init_C[k][t]
                var_est_object.model.Z[t, k].value = init_C[k][t]

        # This comes from Optimizer
        s_array = S_from_DC(var_est_object.model, init_C)
        S_frame = pd.DataFrame(data=s_array,
                               columns=var_est_object.comps['unknown_absorbance'],
                               index=var_est_object._meas_lambdas)

        if hasattr(var_est_object, '_abs_components'):
            component_set = var_est_object._abs_components
        else:
            component_set = var_est_object._mixture_components

        for l in var_est_object._meas_lambdas:
            for k in component_set:
                var_est_object.model.S[l, k].value = S_frame[k][l]  # 1e-2
                if hasattr(var_est_object.model, 'known_absorbance'):
                    if k in var_est_object.model.known_absorbance:
                        var_est_object.model.S[l, k].value = var_est_object.model.known_absorbance_data[k][l]

    print("{: >11} {: >20}".format('Iter', '|Zi-Zi+1|'))
    logiterfile = "iterations.log"
    if os.path.isfile(logiterfile):
        os.remove(logiterfile)

    if lsq_ipopt:
        build_s_model(var_est_object)
        build_c_model(var_est_object)
    else:
        if species_list is None:
            build_scipy_lsq_arrays(var_est_object)
        else:
            lsq_ipopt = True
            build_s_model(var_est_object)
            build_c_model(var_est_object)

    for it in range(max_iter):

        rb = ResultsObject()

        # vars_to_load = ['Z', 'C', 'Cs', 'S', 'Y']
        # if not hasattr(var_est_object, '_abs_components'):
        #     vars_to_load.remove('Cs')
        rb.load_from_pyomo_model(var_est_object.model)#, to_load=vars_to_load)

        solve_Z(var_est_object, solver)

        if lsq_ipopt:
            solve_S(var_est_object, solver)
            solve_C(var_est_object, solver)
        else:
            solved_s = solve_s_scipy(var_est_object)
            solved_c = solve_c_scipy(var_est_object)

        ra = ResultsObject()
        ra.load_from_pyomo_model(var_est_object.model)#, to_load=vars_to_load)

        r_diff = compute_diff_results(rb, ra)
        Z_norm = r_diff.compute_var_norm('Z', norm_order)

        if it > 0:
            print("{: >11} {: >20}".format(it, Z_norm))
        _log_iterations(var_est_object, logiterfile, it)
        if Z_norm < tol and it >= 1:
            break

    results = ResultsObject()

    results.load_from_pyomo_model(var_est_object.model)

    print('Iterative optimization converged. Estimating variances now')
    solved_variances = _solve_variances(var_est_object,
                                        results,
                                        fixed_dev_var=fixed_device_var)

    compute_D_given_SC(var_est_object, results)

    results.P = {name: var_est_object.model.P[name].value for name in var_est_object.model.parameter_names}

    # removes temporary files. This needs to be changes to work with pyutilib
    if os.path.exists(var_est_object._tmp2):
        os.remove(var_est_object._tmp2)
    if os.path.exists(var_est_object._tmp3):
        os.remove(var_est_object._tmp3)
    if os.path.exists(var_est_object._tmp4):
        os.remove(var_est_object._tmp4)

    return results


def solve_initalization(var_est_object, solver, **kwds):
    """Solves formulation 19 in Chen. et al 2016

    :param VarianceEstimator var_est_object: The variance estimation object
    :param str solver: The solver being used (currently not used)
    :param dict kwds: The dict of user settings passed on from the ReactionModel

    :return: None

    """
    solver_opts = kwds.pop('solver_opts', dict())
    tee = kwds.pop('tee', True)
    set_A = kwds.pop('subset_lambdas', list())
    profile_time = kwds.pop('profile_time', False)
    sigmas_sq = kwds.pop('variances', dict())

    if not set_A:
        set_A = var_est_object.model.meas_lambdas

    keys = sigmas_sq.keys()
    # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):

    for k in var_est_object.comps['unknown_absorbance']:
        if k not in keys:
            sigmas_sq[k] = 0.0

    print("Solving Initialization Problem\n")

    # Check this!!!!
    var_est_object._warn_if_D_negative()

    obj = 0.0
    for t in var_est_object.model.times_spectral:
        for l in set_A:
            D_bar = sum(
                var_est_object.model.Z[t, k] * var_est_object.model.S[l, k] for k in var_est_object.comps['unknown_absorbance'])
            obj += (var_est_object.model.D[t, l] - D_bar) ** 2

    var_est_object.model.init_objective = Objective(expr=obj)

    opt = SolverFactory(solver)

    for key, val in solver_opts.items():
        opt.options[key] = val
    solver_results = opt.solve(var_est_object.model,
                               tee=tee,
                               report_timing=profile_time)

    for t in var_est_object.model.times_spectral:
        for k in var_est_object._mixture_components:
            if k in sigmas_sq and sigmas_sq[k] > 0.0:
                var_est_object.model.C[t, k].value = np.random.normal(var_est_object.model.Z[t, k].value, sigmas_sq[k])
            else:
                var_est_object.model.C[t, k].value = var_est_object.model.Z[t, k].value

    var_est_object.model.del_component('init_objective')

    return None


def _log_iterations(var_est_object, filename, iteration):
    """Log solution of each sub-problem in Chen et al. 2016

    :param VarianceEstimator var_est_object: The variance estimation object
    :param str filename: The filename of the tmp files
    :param int iteration: The iteration number

    :return: None

    """
    with open(filename, "a") as f:
        f.write("\n#######################Iteration {}#######################\n".format(iteration))
        with open(var_est_object._tmp2, 'r') as tf:
            f.write("\n#######################Solve Z {}#######################\n".format(iteration))
            f.write(tf.read())
        with open(var_est_object._tmp3, 'r') as tf:
            f.write("\n#######################Solve S {}#######################\n".format(iteration))
            f.write(tf.read())
        with open(var_est_object._tmp4, 'r') as tf:
            f.write("\n#######################Solve C {}#######################\n".format(iteration))
            f.write(tf.read())

    return None


def _solve_variances(var_est_object, results, fixed_dev_var=None):
    """Solves Eq. 23 in Chen et al. 2016

    :param VarianceEstimator var_est_object: The variance estimation object
    :param ResultsObject results: The results object
    :param float fixed_dev_var: Optional device variance

    :return: Indication if variances were estimated successfully.
    :rtype: bool

    """
    nl = len(var_est_object.model.meas_lambdas)
    nt = len(var_est_object.model.times_spectral)
    b = np.zeros((nl, 1))

    variance_dict = dict()
    n_val = len(var_est_object.comps['unknown_absorbance'])

    A = np.ones((nl, n_val + 1))
    reciprocal_nt = 1.0 / nt
    for i, l in enumerate(var_est_object.model.meas_lambdas):
        for j, t in enumerate(var_est_object.model.times_spectral):
            D_bar = 0.0
            for w, k in enumerate(var_est_object.comps['unknown_absorbance']):
                A[i, w] = results.S[k][l] ** 2
                D_bar += results.S[k][l] * results.Z[k][t]
            b[i] += (var_est_object.model.D[t, l] - D_bar) ** 2
        b[i] *= reciprocal_nt

    if fixed_dev_var == None:
        res_lsq = np.linalg.lstsq(A, b, rcond=None)
        all_nonnegative = True
        n_vars = n_val + 1

        for i in range(n_vars):
            if res_lsq[0][i] < 0.0:
                if res_lsq[0][i] < -1e-5:
                    all_nonnegative = False
                else:
                    res_lsq[0][i] = abs(res_lsq[0][i])
            res_lsq[0][i]

        if not all_nonnegative:
            x0 = np.zeros(n_val + 1) + 1e-2
            bb = np.zeros(nl)
            for i in range(nl):
                bb[i] = b[i]

            def F(x, M, rhs):
                return rhs - M.dot(x)

            def JF(x, M, rhs):
                return -M

            res_lsq = least_squares(F, x0, JF,
                                    bounds=(0.0, np.inf),
                                    verbose=2, args=(A, bb))
            for i, k in enumerate(var_est_object.comps['unknown_absorbance']):
                variance_dict[k] = res_lsq.x[i]
            variance_dict['device'] = res_lsq.x[n_val]
            results.sigma_sq = variance_dict
            return res_lsq.success

        else:
            for i, k in enumerate(var_est_object.comps['unknown_absorbance']):
                variance_dict[k] = res_lsq[0][i][0]
            variance_dict['device'] = res_lsq[0][n_val][0]
            results.sigma_sq = variance_dict

    if fixed_dev_var:
        bp = np.zeros((nl, 1))
        Ap = np.zeros((nl, n_val))
        for i, l in enumerate(var_est_object.model.meas_lambdas):
            bp[i] = b[i] - fixed_dev_var
            for j, t in enumerate(var_est_object.model.times_spectral):
                for w, k in enumerate(var_est_object.comps['unknown_absorbance']):
                    Ap[i, w] = results.S[k][l] ** 2

        res_lsq = np.linalg.lstsq(Ap, bp, rcond=None)
        all_nonnegative = True
        n_vars = n_val
        for i in range(n_vars):
            if res_lsq[0][i] < 0.0:
                if res_lsq[0][i] < -1e-5:
                    all_nonnegative = False
                else:
                    res_lsq[0][i] = abs(res_lsq[0][i])
            res_lsq[0][i]

        for i, k in enumerate(var_est_object.comps['unknown_absorbance']):
            variance_dict[k] = res_lsq[0][i][0]
        variance_dict['device'] = fixed_dev_var
        results.sigma_sq = variance_dict

    return 1


def build_scipy_lsq_arrays(var_est_object):
    """Creates arrays for scipy solvers

    :param VarianceEstimator var_est_object: The variance estimation object

    :return: None

    """
    var_est_object._d_array = np.zeros((len(var_est_object.model.times_spectral), len(var_est_object.model.meas_lambdas)))
    for i, t in enumerate(var_est_object.model.times_spectral):
        for j, l in enumerate(var_est_object.model.meas_lambdas):
            var_est_object._d_array[i, j] = var_est_object.model.D[t, l]

    #if hasattr(var_est_object, '_abs_components'):
    #    n_val = var_est_object._nabs_components
    #else:
    n_val = len(var_est_object.comps['unknown_absorbance'])

    var_est_object._s_array = np.ones(len(var_est_object.model.meas_lambdas) * n_val)
    var_est_object._z_array = np.ones(len(var_est_object.model.times_spectral) * n_val)
    var_est_object._c_array = np.ones(len(var_est_object.model.times_spectral) * n_val)

    return None


def compute_diff_results(results1, results2):
    """Calculate differences between results

    :param results1: The first results object
    :param results2: The second results object

    :return: The ResultsObject containing the differences
    :rtype: ResultsObject

    """
    diff_results = ResultsObject()
    diff_results.Z = results1.Z - results2.Z
    diff_results.S = results1.S - results2.S
    diff_results.C = results1.C - results2.C
    return diff_results


def compute_D_given_SC(var_est_object, results, sigma_d=0):
    """Solves Eq. 23 in Chen et al. 2016

    :param VarianceEstimator var_est_object: The variance estimation object
    :param ResultsObject results: The results object
    :param float sigma_d: Device variance

    :return: None

    """
    d_results = []
    
    for i, t in enumerate(var_est_object.model.times_spectral):
        if t in var_est_object.model.times_spectral:
            for j, l in enumerate(var_est_object.model.meas_lambdas):
                suma = 0.0
                for w, k in enumerate(var_est_object.comps['unknown_absorbance']):
                    Cs = results.C[k][t]  # just the absorbing ones
                    Ss = results.S[k][l]
                    suma += Cs * Ss
                if sigma_d:
                    suma += np.random.normal(0.0, sigma_d)
                d_results.append(suma)

    # if hasattr(var_est_object, '_abs_components'):  # added for removing non_abs ones from first term in obj CS
    #     for i, t in enumerate(var_est_object.model.times_spectral):
    #         if t in var_est_object.model.times_spectral:
    #             for j, l in enumerate(var_est_object.model.meas_lambdas):
    #                 suma = 0.0
    #                 for w, k in enumerate(var_est_object._abs_components):
    #                     Cs = results.C[k][t]  # just the absorbing ones
    #                     Ss = results.S[k][l]
    #                     suma += Cs * Ss
    #                 if sigma_d:
    #                     suma += np.random.normal(0.0, sigma_d)
    #                 d_results.append(suma)

    # else:
    #     for i, t in enumerate(var_est_object.model.times_spectral):
    #         if t in var_est_object.model.times_spectral:
    #             for j, l in enumerate(var_est_object.model.meas_lambdas):
    #                 suma = 0.0
    #                 for w, k in enumerate(var_est_object._mixture_components):
    #                     C = results.C[k][t]
    #                     S = results.S[k][l]
    #                     suma += C * S
    #                 if sigma_d:
    #                     suma += np.random.normal(0.0, sigma_d)
    #                 d_results.append(suma)

    d_array = np.array(d_results).reshape((len(var_est_object.model.times_spectral), len(var_est_object.model.meas_lambdas)))
    results.D = pd.DataFrame(data=d_array,
                             columns=var_est_object.model.meas_lambdas,
                             index=var_est_object.model.times_spectral)

    return None
