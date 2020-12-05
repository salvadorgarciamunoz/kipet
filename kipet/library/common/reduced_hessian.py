"""
Reduced Hessian Generation

This module creates the reduced Hessian for use in various KIPET modules
"""
import os
from pathlib import Path
import time

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
from scipy.sparse import coo_matrix, triu
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
    
from kipet.library.common.parameter_handling import (
    set_scaled_parameter_bounds,
    )
 
def get_kkt_info(model_object, method='k_aug'):
    
    """Takes the model and uses PyNumero to get the jacobian and Hessian
    information as dataframes
    
    Args:
        model_object (pyomo ConcreteModel): A pyomo model instance of the current
            problem (used in calculating the reduced Hessian)

        method (str): defaults to k_aug, method by which to obtain optimization
            results

    Returns:
        
        kkt_data (dict): dictionary with the following structure:
            
                {
                'J': J,   # Jacobian
                'H': H,   # Hessian
                'var_ind': var_index_names, # Variable index
                'con_ind': con_index_names, # Constraint index
                'duals': duals, # Duals
                }
        
    """
    if method == 'pynumero':
    
        nlp = PyomoNLP(model_object)
        varList = nlp.get_pyomo_variables()
        conList = nlp.get_pyomo_constraints()
        duals = nlp.get_duals()
        
        J = nlp.extract_submatrix_jacobian(pyomo_variables=varList, pyomo_constraints=conList)
        H = nlp.extract_submatrix_hessian_lag(pyomo_variables_rows=varList, pyomo_variables_cols=varList)
        J = csc_matrix(J)
        
        var_index_names = [v.name for v in varList]
        con_index_names = [v.name for v in conList]
        
    elif method == 'k_aug':
    
        kaug = SolverFactory('k_aug')
        tmpfile_i = "ipopt_output"
        
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
        
        try:
            duals = read_duals(stub + '.sol')
        except:
            duals = None
        
        J = coo_matrix((jac.vals, (jac.irow, jac.jcol)), shape=(m, n)) 
        Hess_coo = coo_matrix((hess.vals, (hess.irow, hess.jcol)), shape=(n, n)) 
        H = Hess_coo + triu(Hess_coo, 1).T
        
        var_index_names = pd.read_csv(col_file, sep = ';', header=None) # dummy sep
        con_index_names = pd.read_csv(row_file, sep = ';', header=None) # dummy sep
        
        var_index_names = [var_name for var_name in var_index_names[0]]
        con_index_names = [con_name for con_name in con_index_names[0].iloc[:-1]]
        con_index_number = {v: k for k, v in enumerate(con_index_names)}
    
    kkt_data = {
                'J': J,
                'H': H,
                'var_ind': var_index_names,
                'con_ind': con_index_names,
                'duals': duals,
                }
    
    return kkt_data

# def get_duals_info(model_object, sol_file, method='k_aug'):
    
#     """Takes the model and uses PyNumero to get the jacobian and Hessian
#     information as dataframes
    
#     Args:
#         model_object (pyomo ConcreteModel): A pyomo model instance of the current
#             problem (used in calculating the reduced Hessian)

#         method (str): defaults to k_aug, method by which to obtain optimization
#             results

#     Returns:
        
#         kkt_data (dict): dictionary with the following structure:
            
#                 {
#                 'J': J,   # Jacobian
#                 'H': H,   # Hessian
#                 'var_ind': var_index_names, # Variable index
#                 'con_ind': con_index_names, # Constraint index
#                 'duals': duals, # Duals
#                 }
        
#     """
#     if method == 'pynumero':
    
#         nlp = PyomoNLP(model_object)
#         conList = nlp.get_pyomo_constraints()
#         duals = nlp.get_duals()
#         con_index_names = [v.name for v in conList]
        
#     elif method == 'k_aug':
    
#         kaug = SolverFactory('k_aug')
#         tmpfile_i = "ipopt_output"
        
#         with open(tmpfile_i, 'r') as f:
#             output_string = f.read()
        
#         stub = output_string.split('\n')[0].split(',')[1][2:-4]
        
#         col_file = Path(stub + '.col')
#         row_file = Path(stub + '.row')
        
#         kaug.options["deb_kkt"] = ""  
#         kaug.solve(model_object, tee=False)
        
#         hess = pd.read_csv('hess_debug.in', delim_whitespace=True, header=None, skipinitialspace=True)
#         hess.columns = ['irow', 'jcol', 'vals']
#         hess.irow -= 1
#         hess.jcol -= 1
#         #os.unlink('hess_debug.in')
        
#         jac = pd.read_csv('jacobi_debug.in', delim_whitespace=True, header=None, skipinitialspace=True)
#         m = jac.iloc[0,0]
#         n = jac.iloc[0,1]
#         jac.drop(index=[0], inplace=True)
#         jac.columns = ['irow', 'jcol', 'vals']
#         jac.irow -= 1
#         jac.jcol -= 1
#         #os.unlink('jacobi_debug.in')
        
#         try:
#             duals = read_duals(stub + '.sol')
#         except:
#             duals = None
        
#         J = coo_matrix((jac.vals, (jac.irow, jac.jcol)), shape=(m, n)) 
#         Hess_coo = coo_matrix((hess.vals, (hess.irow, hess.jcol)), shape=(n, n)) 
#         H = Hess_coo + triu(Hess_coo, 1).T
        
#         var_index_names = pd.read_csv(col_file, sep = ';', header=None) # dummy sep
#         con_index_names = pd.read_csv(row_file, sep = ';', header=None) # dummy sep
        
#         var_index_names = [var_name for var_name in var_index_names[0]]
#         con_index_names = [con_name for con_name in con_index_names[0].iloc[:-1]]
#         con_index_number = {v: k for k, v in enumerate(con_index_names)}
    
#     kkt_data = {
#                 'J': J,
#                 'H': H,
#                 'var_ind': var_index_names,
#                 'con_ind': con_index_names,
#                 'duals': duals,
#                 }
    
#     return kkt_data
    
def prep_model_for_k_aug(model_object):
    """This function prepares the optimization models with required
    suffixes.
    
    Args:
        model_object (pyomo model): The model of the system
        
    Retuns:
        None
        
    """
    model_object.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
    model_object.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
    model_object.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
    model_object.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
    model_object.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)
    model_object.red_hessian = Suffix(direction=Suffix.EXPORT)
    model_object.dof_v = Suffix(direction=Suffix.EXPORT)
    model_object.rh_name = Suffix(direction=Suffix.IMPORT)
    
    count_vars = 1
    for k, v in model_object.P.items():
        model_object.dof_v[k] = count_vars
        count_vars += 1
    
    model_object.npdp = Suffix(direction=Suffix.EXPORT)
    
    return None

# def calculate_m(scenarios, parameter_names):
    
#     m = pd.DataFrame(np.zeros((len(parameter_names), 1)), index=parameter_names, columns=['dual'])
    
#     for model_opt in scenarios:
    
#         duals = calculate_duals(model_opt) 
        
#         for param in m.index:
#             if param in duals.keys():
#                 m.loc[param] = m.loc[param] + duals[param]

#     return -1*m.values

# def calculate_M(scenarios, parameter_names, **kwargs):
#     """
#     scenarions = models
#     parameter_names = names of all parameters (global)
#     needed: globals for the options, because this won't work otherwise
#     """
#     M_size = len(parameter_names)
#     M = pd.DataFrame(np.zeros((M_size, M_size)), index=parameter_names, columns=parameter_names)
        
#     for model in scenarios:
        
#         reduced_hessian = calculate_reduced_hessian(model, parameter_set=parameter_names, **kwargs)
#         M = M.add(reduced_hessian).combine_first(M)
#         M = M[parameter_names]
#         M = M.reindex(parameter_names)
    
#     return M.values

def calculate_duals(model_object, **kwargs):
    
    global_param_name = 'd'
    global_constraint_name = 'fix_params_to_global'
    
    duals = {key: model_object.dual[getattr(model_object, global_constraint_name)[key]] for key, val in getattr(model_object, global_param_name).items()}

    return duals              

# def inner_problem(model_list, parameter_set=None, **kwargs):
    
#     objective_value = 0
#     for model in model_list:
#         optimize_model(model, parameter_set=None, **kwargs)
#         objective_value += model.objective.expr()
        
#     return objective_value

    
def optimize_model(model_object, d=None, parameter_set=None, **kwargs):
    """Takes the model object and performs the optimization
    
    Args:
        model_object (pyomo model): the pyomo model of the reaction
        
        parameter_set (list): list of current model parameters
        
    Returns:
        reduced_hessian (numpy array): reduced hessian of the model
    
    """
    calc_method = kwargs.get('calc_method', 'global')  
    method = kwargs.get('method', 'k_aug')
    scaled = kwargs.get('scaled', False)
    rho = kwargs.get('rho', 10)
    set_param_bounds = kwargs.get('set_param_bounds', False)

    ipopt = SolverFactory('ipopt')
    tmpfile_i = "ipopt_output"

    if calc_method == 'global':
        
        if d is not None:
            param_val_dict = {p: d[i] for i, p in enumerate(parameter_set)}
            for k, v in model_object.P.items():
                v.set_value(param_val_dict[k])
    
        add_global_constraints(model_object, parameter_set=parameter_set, scaled=scaled)
    
    elif calc_method == 'fixed':
        if hasattr(model_object, 'fix_params_to_global'):
            model_object.del_component('fix_params_to_global')  
    
        delta = 1e-20  
        for k, v in model_object.P.items():
            if k in parameter_set:
                ub = model_object.P[k].value
                lb = model_object.P[k].value - delta
                model_object.P[k].setlb(lb)
                model_object.P[k].setub(ub)
                model_object.P[k].unfix()
        
            else:
                model_object.P[k].fix()
                
    if set_param_bounds:
        set_scaled_parameter_bounds(model_object, parameter_set=parameter_set, rho=rho, scaled=scaled)  
                
    ipopt.solve(model_object, 
                symbolic_solver_labels=True, 
                keepfiles=True, 
                tee=False,
                logfile=tmpfile_i,
                )
    
    return None

def calculate_reduced_hessian(model_object, parameter_set=None, **kwargs): 
    """Calculate the reduced Hessian
    
    Args:
        model_object (pyomo model): the pyomo model of the reaction
        
        parameter_set (list): list of current model parameters
        
    Returns:
        reduced_hessian (numpy array): reduced hessian of the model
    
    """
    calc_method = kwargs.get('calc_method', 'global')  
    method = kwargs.get('method', 'k_aug')
    scaled = kwargs.get('scaled', True)
    rho = kwargs.get('rho', 10)
    use_duals = kwargs.get('use_duals', False)
    set_up_constraints = kwargs.get('set_up_constraints', True)
    set_param_bounds = kwargs.get('set_param_bounds', False)
    
    kkt_data = get_kkt_info(model_object, method)
    H = kkt_data['H']
    J = kkt_data['J']
    var_ind = kkt_data['var_ind']
    con_ind_new = kkt_data['con_ind']
    duals = kkt_data['duals']

    col_ind = [var_ind.index(f'P[{v}]') for v in parameter_set]
    m, n = J.shape  
 
    if calc_method == 'global':
        
        dummy_constraints = [f'fix_params_to_global[{k}]' for k in parameter_set]
        jac_row_ind = [con_ind_new.index(d) for d in dummy_constraints] 
        duals_imp = [duals[i] for i in jac_row_ind]
        
        #print(J.shape, len(duals_imp))

        J_c = delete_from_csr(J.tocsr(), row_indices=jac_row_ind).tocsc()
        row_indexer = SparseRowIndexer(J_c)
        J_f = row_indexer[col_ind]
        J_f = delete_from_csr(J_f.tocsr(), row_indices=jac_row_ind, col_indices=[])
        J_l = delete_from_csr(J_c.tocsr(), col_indices=col_ind)  

    elif calc_method == 'fixed':
        
        jac_row_ind = []
        duals_imp = None
    
        J_c = J.tocsc()# delete_from_csr(J.tocsr(), row_indices=jac_row_ind).tocsc()
        row_indexer = SparseRowIndexer(J_c.T)
        J_f = row_indexer[col_ind].T
        #J_f = delete_from_csr(J_f.tocsr(), row_indices=jac_row_ind, col_indices=[]) 
        J_l = delete_from_csr(J_c.tocsr(), col_indices=col_ind)  
        
    else:
        None
    
    #print(f'J: {J.shape}, J_l: {J_l.shape}, J_f: {J_f.shape}')
    
    r_hess = reduced_hessian_matrix(J_f, J_l, H, col_ind)
   
    if use_duals:
        return r_hess.todense()
    else:
        return r_hess.todense()

def reduced_hessian_matrix(F, L, H, col_ind):
    """This calculates the reduced hessian by calculating the null-space based
    on the constraints
    
    Args:
        F (csr_matrix): Rows of the Jacobian related to fixed parameters
        
        L (csr_matrix): The remainder of the Jacobian without parameters
        
        H (csr_matrix): The sparse Hessian
        
        col_ind (list): indicies of columns with fixed parameters
    
    Returns:
        reduced_hessian (csr_matrix): sparse version of the reduced Hessian
        
    """
    n = H.shape[0]
    n_free = n - F.shape[0]
    
    X = spsolve(L.tocsc(), -F.tocsc())
    
    col_ind_left = list(set(range(n)).difference(set(col_ind)))
    col_ind_left.sort()
    
    Z = np.zeros([n, n_free])
    Z[col_ind, :] = np.eye(n_free)
    
    if isinstance(X, csc_matrix):
        Z[col_ind_left, :] = X.todense()
    else:
        Z[col_ind_left, :] = X.reshape(-1, 1)
        
    Z_mat = coo_matrix(np.mat(Z)).tocsr()
    Z_mat_T = coo_matrix(np.mat(Z).T).tocsr()
    Hess = H.tocsr()
    reduced_hessian = Z_mat_T * Hess * Z_mat
    
    return reduced_hessian

def add_global_constraints(model_object, parameter_set=None, scaled=False):
     """This adds the dummy constraints to the model forcing the local
     parameters to equal the current global parameter values
     
     Args:
         model_object (pyomo ConcreteModel): Pyomo model to add constraints to
         
         parameter_set (list): List of parameters to fix using constraints
         
         scaled (bool): True if scaled, False if not scaled
         
     Returns:
         None
     
     """
     if parameter_set is None:
        parameter_set = [p for p in model_object.P]
     
     if scaled:
         global_param_init = {p: 1 for p in parameter_set}
     else:
         global_param_init = {p: model_object.P[p].value for p in parameter_set}
        
     # if d is not None:
     #     global_param_init = {p: d[i] for i, p in enumerate(parameter_set)}
        
     if hasattr(model_object, 'fix_params_to_global'):
         model_object.del_component('fix_params_to_global')   
     
     global_param_name = 'd'
     global_constraint_name = 'fix_params_to_global'
     param_set_name = 'parameter_names'
     
     if hasattr(model_object, 'current_p_set'):
         model_object.del_component('current_p_set')
     
     setattr(model_object, 'current_p_set', Set(initialize=parameter_set))

     if hasattr(model_object, global_param_name):
         model_object.del_component(global_param_name)

     setattr(model_object, global_param_name, Param(getattr(model_object, param_set_name),
                           initialize=global_param_init,
                           mutable=True,
                           ))
     
     def rule_fix_global_parameters(m, k):
         
         return getattr(m, 'P')[k] - getattr(m, global_param_name)[k] == 0
         
     setattr(model_object, global_constraint_name, 
     Constraint(getattr(model_object, 'current_p_set'), rule=rule_fix_global_parameters))

     return None
 
def read_duals(sol_file):
    """Reads the duals from the sol file after solving the problem
    
    Args:
        sol_file (str): The absolute path to the sol file
        
    Returns:
        duals (list): The list of duals values taken from the sol file
    
    """
    sol_file_abs = Path(sol_file)
    
    duals = []
    with sol_file_abs.open() as f: 
        lines = f.readlines()
       
    lines_found = True
    num_of_vars = int(lines[9])
      
    for ln, line in enumerate(lines):
        line = line.rstrip('\n')
        line = line.lstrip('\t').lstrip(' ')
        
        if ln >= 12 and ln <= (num_of_vars + 11):
            duals.append(float(line))
            
    return duals

def delete_from_csr(mat, row_indices=[], col_indices=[]):
    """
    Remove the rows (denoted by ``row_indices``) and columns (denoted by 
    ``col_indices``) from the CSR sparse matrix ``mat``.
    WARNING: Indices of altered axes are reset in the returned matrix
    
    Args:
        mat (csr_matrix): Sparse matrix to delete rows and cols from
        
        row_indicies (list): rows to delete
        
        col_indicies (list): cols to delete
        
    Returns:
        mat (csr_matrix): The sparse matrix with the rows and cols removed
    
    """
    if not isinstance(mat, csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")

    rows = []
    cols = []
    if row_indices:
        rows = list(row_indices)
    if col_indices:
        cols = list(col_indices)

    if len(rows) > 0 and len(cols) > 0:
        row_mask = np.ones(mat.shape[0], dtype=bool)
        row_mask[rows] = False
        col_mask = np.ones(mat.shape[1], dtype=bool)
        col_mask[cols] = False
        return mat[row_mask][:,col_mask]
    elif len(rows) > 0:
        mask = np.ones(mat.shape[0], dtype=bool)
        mask[rows] = False
        return mat[mask]
    elif len(cols) > 0:
        mask = np.ones(mat.shape[1], dtype=bool)
        mask[cols] = False
        return mat[:,mask]
    else:
        return mat

class SparseRowIndexer:
    """Class used to splice sparse matrices"""
    
    def __init__(self, matrix):
        data = []
        indices = []
        indptr = []
        
        _class = 'csr'
        #print(f'LOOK HERE: {type(matrix)}')
        if isinstance(matrix, csc_matrix):
            _class = 'csc'
       #     print(_class)
        
        self._class = _class
        # Iterating over the rows this way is significantly more efficient
        # than matrix[row_index,:] and matrix.getrow(row_index)
        for row_start, row_end in zip(matrix.indptr[:-1], matrix.indptr[1:]):
             data.append(matrix.data[row_start:row_end])
             indices.append(matrix.indices[row_start:row_end])
             indptr.append(row_end-row_start) # nnz of the row

        self.data = np.array(data)
        self.indices = np.array(indices)
        self.indptr = np.array(indptr)
        self.n_columns = matrix.shape[1]

    def __getitem__(self, row_selector):
        data = np.concatenate(self.data[row_selector])
        indices = np.concatenate(self.indices[row_selector])
        indptr = np.append(0, np.cumsum(self.indptr[row_selector]))

        shape = [indptr.shape[0]-1, self.n_columns]
        
        if self._class == 'csr':
            return csr_matrix((data, indices, indptr), shape=shape)
        else:
            return csr_matrix((data, indices, indptr), shape=shape).T.tocsc()