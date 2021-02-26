"""
Module to hold the method developed by Chen et al. 2016
"""
import numpy as np
import scipy
from scipy.optimize import least_squares

from pyomo.environ import (
    Objective,
    SolverFactory,
    )

def solve_C(var_est_object, solver, **kwds):
     """Solves formulation 23 from Weifengs procedure with ipopt

        This method is not intended to be used by users directly

     Args:

     Returns:
         None

     """
     solver_opts = kwds.pop('solver_opts', dict())
     tee = kwds.pop('tee', False)
     update_nl = kwds.pop('update_nl', False)
     profile_time = kwds.pop('profile_time', False)

     # D_bar obj
     obj = 0.0
     for t in var_est_object._meas_times:
         for l in var_est_object._meas_lambdas:
             D_bar = sum(var_est_object.model.S[l, k].value*var_est_object.C_model.C[t, k] for k in var_est_object.component_set)
             obj += (var_est_object.model.D[t, l]-D_bar)**2
     
     var_est_object.C_model.objective = Objective(expr=obj)
             
     if profile_time:
         print('-----------------Solve_C--------------------')

     opt = SolverFactory(solver)

     for key, val in solver_opts.items():
         opt.options[key]=val
         
     solver_results = opt.solve(var_est_object.C_model,
                                logfile=var_est_object._tmp4,
                                tee=tee,
                                #keepfiles=True,
                                report_timing=profile_time)

     var_est_object.C_model.del_component('objective')
     
     for t in var_est_object._allmeas_times:
         for c in var_est_object.component_set:
             var_est_object.model.C[t, c].value = var_est_object.C_model.C[t, c].value

def solve_S(var_est_object, solver, **kwds):
    """Solves formulation 23 from Weifengs procedure with ipopt

       This method is not intended to be used by users directly

    Args:

    Returns:
        None

    """
    solver_opts = kwds.pop('solver_opts', dict())
    tee = kwds.pop('tee', False)
    update_nl = kwds.pop('update_nl', False)
    profile_time = kwds.pop('profile_time', False)
        
    for l in var_est_object._meas_lambdas:
        for c in var_est_object.component_set:
            var_est_object.S_model.S[l, c].value = var_est_object.model.S[l, c].value
            if hasattr(var_est_object.model, 'known_absorbance'):
                if c in var_est_object.model.known_absorbance:
                    if var_est_object.model.S[l, c].value != var_est_object.model.known_absorbance_data[c][l]:
                        var_est_object.model.S[l, c].set_value(var_est_object.model.known_absorbance_data[c][l])
                        var_est_object.S_model.S[l, c].fix()
    
    #D_bar obj
    obj = 0.0
    for t in var_est_object._meas_times:
        for l in var_est_object._meas_lambdas:
            D_bar = sum(var_est_object.S_model.S[l, k] * var_est_object.model.Z[t, k].value for k in var_est_object.component_set)
            obj += (D_bar - var_est_object.model.D[t, l]) ** 2
                
    var_est_object.S_model.objective = Objective(expr=obj)

    if profile_time:
        print('-----------------Solve_S--------------------')

    opt = SolverFactory(solver)

    for key, val in solver_opts.items():
        opt.options[key]=val
    solver_results = opt.solve(var_est_object.S_model,
                               logfile=var_est_object._tmp3,
                               tee=tee,
                               #keepfiles=True,
                               #show_section_timing=True,
                               report_timing=profile_time)

    var_est_object.S_model.del_component('objective')
    
    for l in var_est_object._meas_lambdas:
        for c in var_est_object.component_set:
            var_est_object.model.S[l, c].value = var_est_object.S_model.S[l, c].value
            if hasattr(var_est_object.model, 'known_absorbance'):
                if c in var_est_object.model.known_absorbance:
                    if var_est_object.model.S[l, c].value != var_est_object.model.known_absorbance_data[c][l]:
                        var_est_object.model.S[l, c].set_value(var_est_object.model.known_absorbance_data[c][l])
                        var_est_object.S_model.S[l, c].fix()
               

def solve_Z(var_est_object, solver, **kwds):
    """Solves formulation 20 in weifengs paper

       This method is not intended to be used by users directly

    Args:
    
        solver_opts (dict, optional): options passed to the nonlinear solver

        tee (bool,optional): flag to tell the optimizer whether to stream output
        to the terminal or not
    
        profile_time (bool,optional): flag to tell pyomo to time the construction and solution of the model. 
        Default False
    
        subset_lambdas (array_like,optional): Set of wavelengths to used in initialization problem 
        (Weifeng paper). Default all wavelengths.

    Returns:

        None

    """
    solver_opts = kwds.pop('solver_opts', dict())
    tee = kwds.pop('tee', False)
    profile_time = kwds.pop('profile_time', False)
    
    for t in var_est_object._allmeas_times:
        for k in var_est_object._sublist_components:
            if hasattr(var_est_object.model, 'non_absorbing'):
                if k not in var_est_object.model.non_absorbing:
                    pass
                else:
                    var_est_object.model.C[t, k].fixed = True

    obj = 0.0
    
    # Conc obj no sigma - will this work with missing data?
    for k in var_est_object._sublist_components:
        obj += sum((var_est_object.model.C[t, k]-var_est_object.model.Z[t, k])**2 for t in var_est_object._meas_times)
        

    var_est_object.model.z_objective = Objective(expr=obj)
    if profile_time:
        print('-----------------Solve_Z--------------------')
        
    opt = SolverFactory(solver)

    for key, val in solver_opts.items():
        opt.options[key]=val

    from pyomo.opt import ProblemFormat
    solver_results = opt.solve(var_est_object.model,
                               logfile=var_est_object._tmp2,
                               tee=tee,
                               #show_section_timing=True,
                               report_timing=profile_time)

    var_est_object.model.del_component('z_objective')

def solve_S_from_DC(var_est_object, C_dataFrame, tee=False, with_bounds=False, max_iter=200):
    """Solves a basic least squares problems with SVD.
    
    Args:
        C_dataFrame (DataFrame) data frame with concentration values
    
    Returns:
        DataFrame with estimated S_values 

    """
    D_data = var_est_object.model.D
    if var_est_object._n_meas_lambdas:
        # build Dij vector
        D_vector = np.zeros(var_est_object._n_meas_times*var_est_object._n_meas_lambdas)
        
        row  = []
        col  = []
        data = []    
        for i,t in enumerate(var_est_object._meas_times):
            for j,l in enumerate(var_est_object._meas_lambdas):
                for k,c in enumerate(var_est_object._mixture_components):
                    row.append(i*var_est_object._n_meas_lambdas+j)
                    col.append(j*var_est_object._n_components+k)
                    data.append(C_dataFrame[c][t])
                D_vector[i*var_est_object._n_meas_lambdas+j] = D_data[t,l]    
            
                    
        Bd = scipy.sparse.coo_matrix((data, (row, col)),
                                     shape=(var_est_object._n_meas_times*var_est_object._n_meas_lambdas,
                                            var_est_object._n_components*var_est_object._n_meas_lambdas))

        if not with_bounds:
            if var_est_object._n_meas_times == var_est_object._n_components:
                s_array = scipy.sparse.linalg.spsolve(Bd, D_vector)
            elif var_est_object._n_meas_times>var_est_object._n_components:
                result_ls = scipy.sparse.linalg.lsqr(Bd, D_vector,show=tee)
                s_array = result_ls[0]
            else:
                raise RuntimeError('Need n_t_meas >= var_est_object._n_components')
        else:
            nl = var_est_object._n_meas_lambdas
            nt = var_est_object._n_meas_times
            nc = var_est_object._n_components
            x0 = np.zeros(nl*nc)+1e-2
            M = Bd.tocsr()
            
            def F(x,M,rhs):
                return  rhs-M.dot(x)

            def JF(x,M,rhs):
                return -M

            if tee == True:
                verbose = 2
            else:
                verbose = 0
                
            res_lsq = least_squares(F,x0,JF,
                                    bounds=(0.0,np.inf),
                                    max_nfev=max_iter,
                                    verbose=verbose,args=(M,D_vector))
            s_array = res_lsq.x
            
        s_shaped = s_array.reshape((var_est_object._n_meas_lambdas,var_est_object._n_components))
    else:
        s_shaped = np.empty((var_est_object._n_meas_lambdas,var_est_object._n_components))

    return s_shaped