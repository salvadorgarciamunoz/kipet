"""
Module to hold the method developed by Chen et al. 2016 based on Scipy
"""
# Standard library imports
from contextlib import contextmanager
from io import StringIO
import sys
import time

# Third party imports
import numpy as np
from pyomo.environ import ConcreteModel, Var
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix

def build_s_model(var_est_object):
    """Builds s_model to solve formulation 22 with ipopt

    :param VarianceEstimator var_est_object: The variance estimation object

    :return: None

    """
    var_est_object.S_model = ConcreteModel()
    
    if var_est_object._is_D_deriv:
        lower_bound = 0.0
    else:
        lower_bound = None
        
    var_est_object.S_model.S = Var(
        var_est_object.model.meas_lambdas,
        var_est_object.comps['unknown_absorbance'],
        bounds=(lower_bound, None),
        initialize=1.0
    )

    for l in var_est_object.model.meas_lambdas:
        for k in var_est_object.comps['unknown_absorbance']:
            var_est_object.S_model.S[l, k].value = var_est_object.model.S[l, k].value
            if hasattr(var_est_object.model, 'known_absorbance'):
                if k in var_est_object.model.known_absorbance:
                    if var_est_object.model.S[l, k].value != var_est_object.model.known_absorbance_data[k][l]:
                        var_est_object.model.S[l, k].set_value(var_est_object.model.known_absorbance_data[k][l])
                        var_est_object.S_model.S[l, k].fix()
                        

def solve_s_scipy(var_est_object, **kwds):
    """Solves formulation 22 in Chen et al 2016 (using scipy least_squares)

    This method is not intended to be used by users directly

    :param VarianceEstimator var_est_object: The variance estimation object

    :Keyword Args:

        - tee (bool,optional): flag to tell the optimizer whether to stream output
          to the terminal or not
        - profile_time (bool,optional): flag to tell pyomo to time the construction and solution of the model.
          Default False
        - ftol (float, optional): Tolerance for termination by the change of the cost function. Default is 1e-8
        - xtol (float, optional): Tolerance for termination by the change of the independent variables. Default is 1e-8
        - gtol (float, optional): Tolerance for termination by the norm of the gradient. Default is 1e-8.
        - loss (str, optional): Determines the loss function. The following keyword values are allowed:
        
            - 'linear' (default) : rho(z) = z. Gives a standard least-squares problem.
            - 'soft_l1' : rho(z) = 2 * ((1 + z)**0.5 - 1). The smooth approximation of l1 (absolute value) loss. Usually a good choice for robust least squares.
            - 'huber' : rho(z) = z if z <= 1 else 2*z**0.5 - 1. Works similarly to 'soft_l1'.
            - 'cauchy' : rho(z) = ln(1 + z). Severely weakens outliers influence, but may cause difficulties in optimization process.
            - 'arctan' : rho(z) = arctan(z). Limits a maximum loss on a single residual, has properties similar to 'cauchy'.
        - f_scale (float, optional): Value of soft margin between inlier and outlier residuals, default is 1.0
        - max_nfev (int, optional): Maximum number of function evaluations before the termination

    :return res.success: Success attribute from least_squares

    """
    method = kwds.pop('method', 'trf')
    def_tol = 1.4901161193847656e-07
    ftol = kwds.pop('ftol', def_tol)
    xtol = kwds.pop('xtol', def_tol)
    gtol = kwds.pop('gtol', def_tol)
    x_scale = kwds.pop('x_scale', 1.0)
    loss = kwds.pop('loss', 'linear')
    f_scale = kwds.pop('f_scale', 1.0)
    max_nfev = kwds.pop('max_nfev', None)
    verbose = kwds.pop('verbose', 2)
    profile_time = kwds.pop('profile_time', False)
    tee = kwds.pop('tee', False)

    if profile_time:
        print('-----------------Solve_S--------------------')
        t0 = time.time()

    n = len(var_est_object.comps['unknown_absorbance'])

    for j, l in enumerate(var_est_object.model.meas_lambdas):
        for k, c in enumerate(var_est_object.comps['unknown_absorbance']):
            if var_est_object.model.S[l, c].value < 0.0 and var_est_object._is_D_deriv == False:  #: only less thant zero for non-absorbing
                var_est_object._s_array[j * n + k] = 1e-2
            else:
                var_est_object._s_array[j * n + k] = var_est_object.model.S[l, c].value

    for j, t in enumerate(var_est_object.model.times_spectral):
        for k, c in enumerate(var_est_object.comps['unknown_absorbance']):
            var_est_object._z_array[j * n + k] = var_est_object.model.Z[t, c].value

    def F(x, z_array, d_array, nl, nt, nc):
        diff = np.zeros(nt*nl)
        for i in range(nt):
            for j in range(nl):
                diff[i*nl+j] = d_array[i, j]-sum(z_array[i*nc+k]*x[j*nc+k] for k in range(nc))
        return diff

    def JF(x, z_array, d_array, nl, nt, nc):
        row = []
        col = []
        data = []
        for i in range(nt):
            for j in range(nl):
                for k in range(nc):
                    row.append(i*nl+j)
                    col.append(j*nc+k)
                    data.append(-z_array[i*nc+k])
        return coo_matrix((data, (row, col)),
                          shape=(nt*nl, nc*nl))


    if var_est_object._is_D_deriv == False:
        lower_bound = 0.0
    else:
        lower_bound = -np.inf
            
    if tee:
        res = least_squares(F,
                            var_est_object._s_array,
                            JF,
                            (lower_bound ,np.inf),
                            method,
                            ftol,
                            xtol,
                            gtol,
                            x_scale,
                            loss,
                            f_scale,
                            max_nfev=max_nfev,
                            verbose=verbose,
                            args=(var_est_object._z_array,
                                  var_est_object._d_array,
                                  len(var_est_object.model.meas_lambdas),
                                  len(var_est_object.model.times_spectral),
                                  n)
                            )
        
    else:
        f = StringIO()
        with stdout_redirector(f):
            res = least_squares(F,
                                var_est_object._s_array,
                                JF,
                                (lower_bound ,np.inf),
                                method,
                                ftol,
                                xtol,
                                gtol,
                                x_scale,
                                loss,
                                f_scale,
                                max_nfev=max_nfev,
                                verbose=verbose,
                                args=(var_est_object._z_array,
                                      var_est_object._d_array,
                                      len(var_est_object.model.meas_lambdas),
                                      len(var_est_object.model.times_spectral),
                                      n)
                                )
            
        with open(var_est_object._tmp3,'w') as tf:
            tf.write(f.getvalue())
    
    if profile_time:
        t1 = time.time()
        print("Scipy.optimize.least_squares time={:.3f} seconds".format(t1-t0))

    for j, l in enumerate(var_est_object.model.meas_lambdas):
        for k, c in enumerate(var_est_object.comps['unknown_absorbance']):
            var_est_object.model.S[l, c].value = res.x[j * n + k]
            if hasattr(var_est_object.model, 'known_absorbance'):
                if c in var_est_object.model.known_absorbance:
                    var_est_object.model.S[l, c].set_value(var_est_object.model.known_absorbance_data[c][c][l])

    return res.success


def build_c_model(var_est_object):
    """Builds s_model to solve formulation 25 with ipopt

    :param VarianceEstimator var_est_object: The variance estimation object

    :return: None

    """
    var_est_object.C_model = ConcreteModel()
    
    var_est_object.C_model.C = Var(
        var_est_object.model.times_spectral,
        var_est_object.comps['unknown_absorbance'],
        bounds=(0.0, None),
        initialize=1.0
    )

    for l in var_est_object.model.times_spectral:
        for k in var_est_object.comps['unknown_absorbance']:
            var_est_object.C_model.C[l, k].value = var_est_object.model.C[l, k].value
            if hasattr(var_est_object.model, 'non_absorbing'):
                var_est_object.C_model.C[l, k].fix()


def solve_c_scipy(var_est_object, **kwds):
    """Solves formulation 25 in weifengs paper (using scipy least_squares)

    :param VarianceEstimator var_est_object: The variance estimation object

    :Keyword Args:

        - tee (bool,optional): flag to tell the optimizer whether to stream output
          to the terminal or not
        - profile_time (bool,optional): flag to tell pyomo to time the construction and solution of the model.
          Default False
        - ftol (float, optional): Tolerance for termination by the change of the cost function. Default is 1e-8
        - xtol (float, optional): Tolerance for termination by the change of the independent variables. Default is 1e-8
        - gtol (float, optional): Tolerance for termination by the norm of the gradient. Default is 1e-8.
        - loss (str, optional): Determines the loss function. The following keyword values are allowed:

            - 'linear' (default) : rho(z) = z. Gives a standard least-squares problem.
            - 'soft_l1' : rho(z) = 2 * ((1 + z)**0.5 - 1). The smooth approximation of l1 (absolute value) loss. Usually a good choice for robust least squares.
            - 'huber' : rho(z) = z if z <= 1 else 2*z**0.5 - 1. Works similarly to 'soft_l1'.
            - 'cauchy' : rho(z) = ln(1 + z). Severely weakens outliers influence, but may cause difficulties in optimization process.
            - 'arctan' : rho(z) = arctan(z). Limits a maximum loss on a single residual, has properties similar to 'cauchy'.
        - f_scale (float, optional): Value of soft margin between inlier and outlier residuals, default is 1.0
        - max_nfev (int, optional): Maximum number of function evaluations before the termination

    :return res.success: Success attribute from least_squares

    """
    method = kwds.pop('method','trf')
    def_tol = 1.4901161193847656e-07
    ftol = kwds.pop('ftol',def_tol)
    xtol = kwds.pop('xtol',def_tol)
    gtol = kwds.pop('gtol',def_tol)
    x_scale = kwds.pop('x_scale',1.0)
    loss = kwds.pop('loss','linear')
    f_scale = kwds.pop('f_scale',1.0)
    max_nfev = kwds.pop('max_nfev',None)
    verbose = kwds.pop('verbose',2)
    profile_time = kwds.pop('profile_time',False)
    tee =  kwds.pop('tee',False)
    
    if profile_time:
        print('-----------------Solve_C--------------------')
        t0 = time.time()
   
    n = len(var_est_object.comps['unknown_absorbance'])
        
    count = 0
    for i, t in enumerate(var_est_object.model.times_spectral):
        for k, c in enumerate(var_est_object.comps['unknown_absorbance']):

            if getattr(var_est_object.model, 'C')[t, c].value <= 0.0:
                var_est_object._c_array[i * n + k] = 1e-15
            else:
                var_est_object._c_array[i * n + k] = getattr(var_est_object.model, 'C')[t, c].value

        count += 1
    
    count = 0
    for j, l in enumerate(var_est_object.model.meas_lambdas):
        for k, c in enumerate(var_est_object.comps['unknown_absorbance']):
            var_est_object._s_array[j * n + k] = var_est_object.model.S[l, c].value
            count += 1

    def F(x, s_array, d_array, nl, nt, nc):
        diff = np.zeros(nt * nl)
        for i in range(nt):
            for j in range(nl):
                diff[i * nl + j] = d_array[i, j] - sum(s_array[j * nc + k] * x[i * nc + k] for k in range(nc))
        return diff

    def JF(x, s_array, d_array, nl, nt, nc):
        row = []
        col = []
        data = []
        
        for i in range(nt):
            for j in range(nl):
                for k in range(nc):
                    row.append(i*nl+j)
                    col.append(i*nc+k)
                    data.append(-s_array[j*nc+k])

        return coo_matrix((data, (row, col)),
                          shape=(nt*nl,nc*nt))

    if tee:
        res = least_squares(F, 
                            var_est_object._c_array, 
                            JF,
                            (0.0, np.inf), 
                            method,
                            ftol, 
                            xtol, 
                            gtol,
                            x_scale, 
                            loss, 
                            f_scale,
                            max_nfev=max_nfev,
                            verbose=verbose,
                            args=(var_est_object._s_array,
                                  var_est_object._d_array,
                                  len(var_est_object.model.meas_lambdas),
                                  len(var_est_object.model.times_spectral),
                                  n)
                            )
        
    else:
        f = StringIO()
        with stdout_redirector(f):
            res = least_squares(F, 
                                var_est_object._c_array, 
                                JF,
                                (0.0, np.inf), 
                                method,
                                ftol, 
                                xtol, 
                                gtol,
                                x_scale, 
                                loss, 
                                f_scale,
                                max_nfev=max_nfev,
                                verbose=verbose,
                                args=(var_est_object._s_array,
                                      var_est_object._d_array,
                                      len(var_est_object.model.meas_lambdas),
                                      len(var_est_object.model.times_spectral),
                                      n)
                                )
                
        with open(var_est_object._tmp4, 'w') as tf:
            tf.write(f.getvalue())

    if profile_time:
        t1 = time.time()
        print("Scipy.optimize.least_squares time={:.3f} seconds".format(t1-t0))

    for j, t in enumerate(var_est_object.model.times_spectral):
        for k,c in enumerate(var_est_object.comps['unknown_absorbance']):
            getattr(var_est_object.model, 'C')[t,c].value = res.x[j*n+k]
   
    return res.success

@contextmanager
def stdout_redirector(stream):
    old_stdout = sys.stdout
    sys.stdout = stream
    try:
        yield
    finally:
        sys.stdout = old_stdout
