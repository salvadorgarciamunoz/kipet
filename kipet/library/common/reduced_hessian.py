"""
Reduced Hessian Generation

This module creates the reduced Hessian for use in various KIPET modules
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.environ import (
    Constraint,
    Param,
    Set,
    SolverFactory,
    Suffix,
    )
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from kipet.library.common.parameter_handling import (
    set_scaled_parameter_bounds,
    )

def get_kkt_info(model_object, parameter_set, method='pynumero'):
    
    if method == 'pynumero':
        return pynumero_kkt(model_object, parameter_set)
    elif method == 'k_aug':
        return k_aug_kkt(model_object, parameter_set)
    else:
        return pynumero_kkt(model_object, parameter_set)
     
def k_aug_kkt(model_object, parameter_set):
    """Returns the KKT matrix components using k_aug"""

    kaug = SolverFactory('k_aug')
    tmpfile_i = "ipopt_output"
    
    #model_object.ipopt_zL_in.update(model_object.ipopt_zL_out)
    #model_object.ipopt_zU_in.update(model_object.ipopt_zU_out)
    
    with open(tmpfile_i, 'r') as f:
        output_string = f.read()
    
    stub = output_string.split('\n')[0].split(',')[1][2:-4]
    col_file = Path(stub + '.col')
    row_file = Path(stub + '.row')
    
    kaug.options["deb_kkt"] = ""  
    kaug.solve(model_object, tee=False)
    
    hess = pd.read_csv('hess_debug.in', delim_whitespace=True, header=None, skipinitialspace=True)
    hess.columns = ['irow', 'jcol', 'vals']
    hess.irow -= 1
    hess.jcol -= 1
    #os.unlink('hess_debug.in')
    
    jac = pd.read_csv('jacobi_debug.in', delim_whitespace=True, header=None, skipinitialspace=True)
    m = jac.iloc[0,0]
    n = jac.iloc[0,1]
    jac.drop(index=[0], inplace=True)
    jac.columns = ['irow', 'jcol', 'vals']
    jac.irow -= 1
    jac.jcol -= 1
    #os.unlink('jacobi_debug.in')
    
    Jac_coo = coo_matrix((jac.vals, (jac.irow, jac.jcol)), shape=(m, n)) 
    Hess_coo = coo_matrix((hess.vals, (hess.irow, hess.jcol)), shape=(n, n)) 
    Jac = Jac_coo.todense()
    #Hess = Hess_coo.tocsr()
    Hess = Hess_coo.todense()
    
    for i in range(Hess.shape[0]):
        for j in range(Hess.shape[1]):
            if i == j:
                continue
            Hess[j, i] = Hess[i, j]
    
    # Hess = Hess + Hess.T - np.diag(np.asarray(Hess.diagonal()).flatten())
    
    var_index_names = pd.read_csv(col_file, sep = ';', header=None) # dummy sep
    con_index_names = pd.read_csv(row_file, sep = ';', header=None) # dummy sep
    
    var_index_names = [var_name for var_name in var_index_names[0]]
    con_index_names = [con_name for con_name in con_index_names[0].iloc[:-1]]
    
    J_df = pd.DataFrame(Jac, columns=var_index_names, index=con_index_names)
    H_df = pd.DataFrame(Hess, columns=var_index_names, index=var_index_names)
    
    var_index_names = pd.DataFrame(var_index_names)
    
    KKT_up = pd.merge(H_df, J_df.transpose(), left_index=True, right_index=True)
    KKT = pd.concat((KKT_up, J_df))
    KKT = KKT.fillna(0)
    
    duals = None
    
    return KKT, H_df, J_df, var_index_names, con_index_names, duals
 
def pynumero_kkt(model_object, parameter_set):
    
    """Takes the model and uses PyNumero to get the jacobian and Hessian
    information as dataframes
    
    Args:
        model (pyomo ConcreteModel): A pyomo model instance of the current
        problem (used in calculating the reduced Hessian)

    Returns:
        
        KKT (pd.DataFrame): the KKT matrix as a dataframe
        
        H_df (pd.DataFrame): the Hessian as a dataframe
        
        J_df (pd.DataFrame): the jacobian as a dataframe
        
        var_index_names (list): the index of variables
        
        con_index_names (list): the index of constraints
        
    """
    nlp = PyomoNLP(model_object)
    varList = nlp.get_pyomo_variables()
    conList = nlp.get_pyomo_constraints()
    duals = nlp.get_duals()
    
    #print(f'duals: {duals}')
    #print(len(duals))
    
    J = nlp.extract_submatrix_jacobian(pyomo_variables=varList, pyomo_constraints=conList)
    H = nlp.extract_submatrix_hessian_lag(pyomo_variables_rows=varList, pyomo_variables_cols=varList)
    
    var_index_names = [v.name for v in varList]
    con_index_names = [v.name for v in conList]

    J_df = pd.DataFrame(J.todense(), columns=var_index_names, index=con_index_names)
    H_df = pd.DataFrame(H.todense(), columns=var_index_names, index=var_index_names)
    
    var_index_names = pd.DataFrame(var_index_names)
    
    
    # You will have to use scarce matrices in order to do this.
    
    KKT_up = pd.merge(H_df, J_df.transpose(), left_index=True, right_index=True)
    KKT = pd.concat((KKT_up, J_df))
    KKT = KKT.fillna(0)
    
    return KKT, H_df, J_df, var_index_names, con_index_names, duals
    

# def prep_model_for_k_aug(model_object):
#     """This function prepares the optimization models with required
#     suffixes. This is here because I don't know if this is already in 
#     KIPET somewhere else.
    
#     Args:
#         model (pyomo model): The model of the system
        
#     Retuns:
#         None
        
#     """
#     #model_object.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
#     # model_object.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
#     # model_object.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
#     # model_object.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
#     # model_object.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)
#     #model_object.red_hessian = Suffix(direction=Suffix.EXPORT)
#     # model_object.dof_v = Suffix(direction=Suffix.EXPORT)
#     #model_object.rh_name = Suffix(direction=Suffix.IMPORT)
    
#     # count_vars = 1
#     # for k, v in model_object.P.items():
#     #     model_object.dof_v[k] = count_vars
#     #     count_vars += 1
    
#     #model_object.npdp = Suffix(direction=Suffix.EXPORT)
    
#    return None
def calculate_reduced_hessian(model_object, parameter_set=None, **kwargs): #method='k_aug', calc_method='global'):
    
    calc_method = kwargs.get('calc_method', 'fixed')  
    method = kwargs.get('method', 'k_aug')
    scaled = kwargs.get('scaled', True)
    rho = kwargs.get('rho', 10)

    if calc_method == 'global':
        return _calculate_reduced_hessian_global(model_object, parameter_set, method, scaled, rho)
    elif calc_method == 'fixed':
        return _calculate_reduced_hessian_fixed(model_object, parameter_set, method)
    else:
        raise ValueError('Unknown method for reduced Hessian calculation.')


def _calculate_reduced_hessian_global(model_object, parameter_set, method, scaled, rho):
    """Calculate the reduced Hessian
    
    Args:
        model_object (pyomo model): the pyomo model of the reaction
        
        parameter_set (list): list of current model parameters
        
    Returns:
        reduced_hessian (numpy array): reduced hessian of the model
    
    """
    ipopt = SolverFactory('ipopt')
    tmpfile_i = "ipopt_output"
    
    #rh_model = copy.deepcopy(self.model)
    add_global_constraints(model_object, parameter_set, scaled=scaled)
    set_scaled_parameter_bounds(model_object, parameter_set=parameter_set, rho=rho, scaled=scaled)

    # for k, v in self.solver_opts.items():
    #     ipopt.options[k] = v
    
    ipopt.solve(model_object, 
                symbolic_solver_labels=True, 
                keepfiles=True, 
                tee=True,
                logfile=tmpfile_i,
                #solver_options=self.solver_opts,
                )
    
    if parameter_set is None:
        parameter_set = [p for p in model_object.P]

    # Get the KKT information from the model object
    kkt_df, hess, jac, var_ind, con_ind_new, duals = get_kkt_info(model_object, parameter_set=parameter_set, method=method)
    
    M_size = len(model_object.P)
    M = pd.DataFrame(np.zeros((M_size, M_size))) #, index=parameter_names, columns=parameter_names)
    col_ind  = [var_ind.loc[var_ind[0] == f'P[{v}]'].index[0] for v in parameter_set]
    dummy_constraints = [f'fix_params_to_global[{k}]' for k in parameter_set]
    dc = [d for d in dummy_constraints]
    
    K = kkt_df.drop(index=dc, columns=dc)
    E = np.zeros((len(dummy_constraints), K.shape[1]))
    
    for i, indx in enumerate(col_ind):
        E[i, indx] = 1

    # Make square matrix (A) of Eq. 14
    top = (K, E.T)
    bot = (E, np.zeros((len(dummy_constraints), len(dummy_constraints))))
    top = np.hstack(top)
    bot = np.hstack(bot)
    A = np.vstack((top, bot))
    
    # Make the rhs (b) of Eq. 14
    b = np.vstack((np.zeros((K.shape[0], len(dummy_constraints))), -1*np.eye(len(dummy_constraints))))

    # Solve for Qi and Si
    rhs = np.linalg.solve(A, b)
    Si = rhs[-rhs.shape[1]:, :]
        
    # This is the small matrix - how can you reconcile this with the NSD and RHPS?
    Mi = pd.DataFrame(Si, index=parameter_set, columns=parameter_set)
    reduced_hessian = Mi.values

    return reduced_hessian 

def _calculate_reduced_hessian_fixed(model_object, parameter_set=None, method='k_aug', verbose=False):
        """This function solves an optimization with very restrictive bounds
        on the paramters in order to get the reduced hessian at fixed 
        conditions
        
        Args:
            Se (list): The current list of estimable parameters.
            
            Sf (list): The current list of fixed parameters.
            
            verbose (bool): Defaults to False, option to show the output from
                the solver (solver option 'tee').
            
        Returns:
            reduced_hessian (np.ndarray): The resulting reduced hessian matrix.
            
        """
        delta = 1e-20
        n_free = len(parameter_set)
        ipopt = SolverFactory('ipopt')
        kaug = SolverFactory('k_aug')
        tmpfile_i = "ipopt_output"
        
        #self._run_simulation()
        
        if hasattr(model_object, 'fix_params_to_global'):
            model_object.del_component('fix_params_to_global')  
        
        for k, v in model_object.P.items():
            if k in parameter_set:
                ub = model_object.P[k].value
                lb = model_object.P[k].value - delta
                model_object.P[k].setlb(lb)
                model_object.P[k].setub(ub)
                model_object.P[k].unfix()
        
            else:
                model_object.P[k].fix()
        
        ipopt.solve(model_object, symbolic_solver_labels=True, keepfiles=True, tee=verbose, logfile=tmpfile_i)

        with open(tmpfile_i, 'r') as f:
            output_string = f.read()
        
        stub = output_string.split('\n')[0].split(',')[1][2:-4]
        col_file = Path(stub + '.col')
        
        kaug.options["deb_kkt"] = ""  
        kaug.solve(model_object, tee=False)#verbose)
        
        hess = pd.read_csv('hess_debug.in', delim_whitespace=True, header=None, skipinitialspace=True)
        hess.columns = ['irow', 'jcol', 'vals']
        hess.irow -= 1
        hess.jcol -= 1
        
        jac = pd.read_csv('jacobi_debug.in', delim_whitespace=True, header=None, skipinitialspace=True)
        m = jac.iloc[0,0]
        n = jac.iloc[0,1]
        jac.drop(index=[0], inplace=True)
        jac.columns = ['irow', 'jcol', 'vals']
        jac.irow -= 1
        jac.jcol -= 1
        
        Jac_coo = coo_matrix((jac.vals, (jac.irow, jac.jcol)), shape=(m, n)) 
        Hess_coo = coo_matrix((hess.vals, (hess.irow, hess.jcol)), shape=(n, n)) 
        Jac = Jac_coo.todense()
        
        var_ind = pd.read_csv(col_file, sep = ';', header=None) # dummy sep
        col_ind = [var_ind.loc[var_ind[0] == f'P[{v}]'].index[0] for v in parameter_set]
        
        Jac_f = Jac[:, col_ind]
        Jac_l = np.delete(Jac, col_ind, axis=1)
        X = spsolve(coo_matrix(np.mat(Jac_l)).tocsc(), coo_matrix(np.mat(-Jac_f)).tocsc())
        
        col_ind_left = list(set(range(n)).difference(set(col_ind)))
        col_ind_left.sort()
        
        Z = np.zeros([n, n_free])
        Z[col_ind, :] = np.eye(n_free)
        Z[col_ind_left, :] = X.todense()
    
        Z_mat = coo_matrix(np.mat(Z)).tocsr()
        Z_mat_T = coo_matrix(np.mat(Z).T).tocsr()
        Hess = Hess_coo.tocsr()
        red_hessian = Z_mat_T * Hess * Z_mat
        
        return red_hessian.todense()

def add_global_constraints(model_object, parameter_set=None, scaled=False):
     """This adds the dummy constraints to the model forcing the local
     parameters to equal the current global parameter values
     
     """
     if parameter_set is None:
        parameter_set = [p for p in model_object.P]
     
     if scaled:
         global_param_init = {p: 1 for p in parameter_set}
     else:
         global_param_init = {p: model_object.P[p].value for p in parameter_set}
        
     if hasattr(model_object, 'fix_params_to_global'):
         model_object.del_component('fix_params_to_global')   
     
     global_param_name = 'd'
     global_constraint_name = 'fix_params_to_global'
     param_set_name = 'parameter_names'
     
     setattr(model_object, 'current_p_set', Set(initialize=parameter_set))

     setattr(model_object, global_param_name, Param(getattr(model_object, param_set_name),
                           initialize=global_param_init,
                           mutable=True,
                           ))
     
     def rule_fix_global_parameters(m, k):
         
         return getattr(m, 'P')[k] - getattr(m, global_param_name)[k] == 0
         
     setattr(model_object, global_constraint_name, 
     Constraint(getattr(model_object, 'current_p_set'), rule=rule_fix_global_parameters))

     print('constraints added')

     return None