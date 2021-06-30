"""
Reduced Hessian methods renewed.

This module is to hold generic methods for parameter estimator and mee for determining the
reduced hessian. This will eventually replace the reduced hessian module.
"""
# Third party imports
import numpy as np
import pandas as pd
from pathlib import Path
from pyomo.environ import SolverFactory, Suffix
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, triu

# Kipet library imports
from kipet.general_settings.variable_names import VariableNames
from kipet.model_tools.pyomo_model_tools import convert
    
__var = VariableNames()


def define_free_parameters(models_dict, global_params=None, kind='full'):
    """Identifies the variables in the model used to define the reduced hessian  
       
    :param str kind: full, variable, simple
    
    :return: None
    
    .. todo::
        
        This works with PE too, just supply the r1.p_model
        In MEE, you provide a dict of p_models:
            {k: v.p_model for k, v in self.reaction_models.items()}

    """
    global_params = [] if global_params is None else global_params
    param_names = []
    model_count = False
    
    methods = {'variable': 'getname',
               'simple': 'simple',
               'full': 'to_string'}
    
    if kind not in methods:
        print('Wrong variable naming convention')
    
    def name(method, p, v):
        
        if method in ['to_string', 'getname']:
            return getattr(v, method)()
        else:
            return p
    
    for exp, model in models_dict.items():
        for parameter_kind in __var.optimization_variables:
            if hasattr(model, parameter_kind):
                for p, v in getattr(model, parameter_kind).items():
                    if v.is_fixed() or model_count and v.getname() in global_params:
                        continue
                    param_names.append(name(methods[kind], p, v))
                    
        model_count = True
    
    return param_names


def define_reduce_hess_order(models_dict, component_set, param_names_full):
    """
    This sets up the suffixes of the reduced hessian for use with SIpopt
    
    :param dict models_dict: The dict of parameter estimator models
    :param component_set list: The list of components used
    :param param_names_full: The global parameters as a list (var name)

    :return index_to_variable: Mapping of an index to the model variable
    :rtype: dict

    """
    index_to_variable = dict()
    count_vars = 1
    exp_count = True
    
    for exp, model in models_dict.items():
        if hasattr(model, 'C'):
            var = 'C'
            for index in getattr(model, var).index_set():
                if index[1] in component_set:
                    v = getattr(model, var)[index]
                    index_to_variable[count_vars] = v
                    count_vars += 1
                
    for exp, model in models_dict.items():
        if hasattr(model, 'S') and exp_count:
            var = 'S'
            for index in getattr(model, var).index_set():
                if index[1] in component_set:
                    v = getattr(model, var)[index]
                    index_to_variable[count_vars] = v
                    count_vars += 1
       # exp_count = False
        
    for exp, model in models_dict.items():
        
        for parameter_kind in __var.optimization_variables:
            if hasattr(model, parameter_kind):
                for v in getattr(model, parameter_kind).values():
                    if v.to_string() in param_names_full:
                        index_to_variable[count_vars] = v
                        count_vars += 1

    return index_to_variable


def compute_covariance(models_dict, hessian, free_params, all_variances):
    """This method calculates the covariance matrix for spectral problems
    
    :param dict models_dict: The models in dict form
    :param numpy.ndarray hessian: The reduced hessian
    :param int free_params: The number of free parameters
    :param dict all_variances: The dict of variances (as dict)
    
    :return V_theta: The parameter covariance matrix
    :rtype: scipy.sparse.coo_matrix
    
    """
    H = hessian
    B = make_B_matrix(models_dict, free_params, all_variances)
    Vd = make_Vd_matrix(models_dict, all_variances)
    
    print(f'H: {H.shape = }')
    print(f'B: {B.shape = }')
    print(f'Vd: {Vd.shape = }')
    
    R = B.T @ H.T
    A = Vd @ R
    L = H @ B
    V_theta = 4*(A.T @ L.T).T 
    
    return V_theta
        
    
def make_B_matrix(models_dict, free_params, all_variances):

    """This method generates a B matrix for the covariance calculations in both
    single and multiple experiments.

    :param dict models_dict: The models in dict form
    :param int free_params: The number of free parameters
    
    :return B_matrix: The B matrix
    :rtype: scipy.sparse.coo_matrix
    
    """
    S_part = merge_blocks('S', models_dict, all_variances)
    C_part = merge_blocks('C', models_dict, all_variances)
    
    sy, sx = S_part.shape
    M = np.zeros((C_part.size + S_part.size + free_params, C_part.shape[1]*S_part.shape[1]))
    c_index_start = C_part.size
    
    for i in range(C_part.shape[1]):
    
        M[i*sy:(i + 1)*sy, i*sx:(i + 1)*sx] = S_part
        Ct = C_part[:, i].reshape(-1, 1)
        cy, cx = Ct.shape

        for ci in range(sx):
            M[c_index_start + ci*cy : c_index_start + (ci + 1)*cy, cx*ci + i*sx : cx*ci + i*sx + cx] = Ct
    
    B_matrix = coo_matrix(M)
    
    return B_matrix


def merge_blocks(var, models_dict, all_variances):
    """Merges the defined data types into single blocks
    
    :param str var: The model variable
    
    :return: s_mat: the array of model data
    :rtype: numpy.ndarray
    
    """
    # TODO: Make sure blocks of different sizes (components) can be handled
    s_mat = None
    for e, exp in models_dict.items():
        
        comp_vars = [k for k, v in all_variances[e].items() if k != 'device']    
        
        if s_mat is None:
            s_mat = convert(getattr(exp, var)).loc[:, comp_vars].T.values / all_variances[e]['device']
        else:
            s_mat_append = convert(getattr(exp, var)).loc[:, comp_vars].T.values / all_variances[e]['device']
            s_mat = np.hstack((s_mat, s_mat_append))

    return s_mat


def make_Vd_matrix(models_dict, all_variances):
    """Builds d covariance matrix

    This method is not intended to be used by users directly

    :param dict models_dict: Either a pyomo ConcreteModel or a dict of ReactionModels
    :param dict all_variances: variances

    :return: None

    """
    from kipet.model_tools.pyomo_model_tools import convert
    
    Vd_dict = {}
    M_dict = {}
    total_shape = 0
    n_models = len(models_dict)
    
    for name, model in models_dict.items():

        variances = all_variances[name]
        times = model.allmeas_times.ordered_data()
        waves = model.meas_lambdas.ordered_data()
        n_waves = len(waves)
        n_times = len(times)
        Vd = np.zeros((n_models * n_times * n_waves, n_models * n_times * n_waves))
    
        S = convert(model.S)
        comp_vars = [k for k, v in variances.items() if k != 'device']
        S = S.loc[:, comp_vars]
        
        device_variance = variances['device']
        M = np.array([v for k, v in variances.items() if k != 'device']) * S.values @ S.values.T
        M_diag = np.einsum('ii->i', M)
        M_diag += device_variance
        M_dict[name] = M

        for t in range(n_models*n_times):
            Vd[t*n_waves: (t+1)*n_waves, t*n_waves: (t+1)*n_waves] = M
            
        total_shape += Vd.shape[0]
        Vd_dict[name] = Vd
    
    if n_models > 1:
        
        Vd_combined = np.zeros((total_shape, total_shape))
        start_index = 0
        for model, Vd in Vd_dict.items():        
            Vd_combined[start_index:Vd.shape[0]+start_index, start_index:Vd.shape[1]+start_index] = Vd
            start_index = Vd.shape[0]
        
        return coo_matrix(Vd_combined)
    
    return coo_matrix(Vd)


def index_variable_mapping(model_dict, components, parameter_names, mee_obj=None):
    """This adds the needed suffixes for the reduced hessian to the model object
    used in covariance predictions
    
    :param ConcreteModel model_dict: The model from the parameter estimator or the MEE
    :param list components: The list of absorbing components
    :param list parameter_names: The list of parameter names
    :param mee_obj: An MEE instance, default None
    
    :return: index_to_variables
    :rtype: dict
    
    """
    if mee_obj is not None:
        model_obj = mee_obj
    else:
        model_obj = model_dict
        
    model_obj.red_hessian = Suffix(direction=Suffix.IMPORT_EXPORT)
    index_to_variables = define_reduce_hess_order(model_dict, components, parameter_names)
    
    for k, v in index_to_variables.items():
        model_obj.red_hessian[v] = k
        
    return index_to_variables


def covariance_sipopt(model_obj, solver_factory, components, parameters, mee_obj=None):
    """Generalize the covariance optimization with IPOPT Sens

    :param ConcreteModel model: The Pyomo model used in parameter fitting
    :param SolverFactory optimizer: The SolverFactory currently being used
    :param bool tee: Display option
    :param dict all_sigma_specified: The provided variances
    :param mee_obj: An MEE instance, default None

    :return hessian: The covariance matrix
    :rtype: numpy.ndarray

    """
    from kipet.input_output.read_hessian import split_sipopt_string, read_reduce_hessian
            
    _tmpfile = "ipopt_hess"
    index_to_variable = index_variable_mapping(model_obj, components, parameters, mee_obj)
    
    optimization_model = model_obj if mee_obj is None else mee_obj
    
    solver_results = solver_factory.solve(
        optimization_model,
        tee=False,
        logfile=_tmpfile,
        report_timing=True,
        symbolic_solver_labels=True,
        keepfiles=True)
    
    with open(_tmpfile, 'r') as f:
        output_string = f.read()
    
    ipopt_output, hessian_output = split_sipopt_string(output_string)
    covariance_matrix = read_reduce_hessian(hessian_output, len(index_to_variable))
    covariance_matrix_reduced = covariance_matrix[-len(parameters):, :]
    
    return covariance_matrix, covariance_matrix_reduced


def covariance_k_aug(model_obj, solver_factory, components, parameters, mee_obj=None):
    """Generalize the covariance optimization with IPOPT Sens

    :param ConcreteModel model: The Pyomo model used in parameter fitting
    :param SolverFactory optimizer: The SolverFactory currently being used
    :param bool tee: Display option
    :param dict all_sigma_specified: The provided variances

    :return hessian: The covariance matrix
    :rtype: numpy.ndarray

    """
    _tmpfile = "k_aug_hess"
    optimization_model = model_obj if mee_obj is None else mee_obj
    add_warm_start_suffixes(optimization_model, use_k_aug=True)   
    
    ip = SolverFactory('ipopt')
    solver_results = ip.solve(
        optimization_model,
        tee=False,
        logfile=_tmpfile,
        report_timing=True,
        symbolic_solver_labels=True,
        keepfiles=True
        )

    update_warm_start(optimization_model)
    
    k_aug = SolverFactory('k_aug')
    k_aug.options["print_kkt"] = ""
    k_aug.solve(optimization_model, tee=True)
    stub = ip._problem_files[0][:-3]
    
    var_index_names, con_index_names = var_con_data(stub)
    size = (len(con_index_names), len(var_index_names))
    col_ind, col_ind_param_hr = free_variables(model_obj, components, parameters, var_index_names)
    covariance_matrix = calculate_inverse_hr(size, col_ind)
    covariance_matrix_reduced = covariance_matrix[col_ind_param_hr, :]

    return covariance_matrix, covariance_matrix_reduced


def var_con_data(file_stub):
    """
    This prepares the indecies and sizes for the reduced Hessian strucuturing.
    
    The variables are arranged such that the parameters need to be in the right order
    and entered in last (for k_aug).
    
    The commented code lets you place the variables in any order, as long as they are
    still located at the end (important for B and Vd matrix building.)
    
    """
    var_index = pd.read_csv(Path(file_stub + '.col'), sep=';', header=None)  # dummy sep
    con_index = pd.read_csv(Path(file_stub + '.row'), sep=';', header=None)  # dummy sep
    var_index_names = [var_name for var_name in var_index[0]]
    con_index_names = [con_name for con_name in con_index[0].iloc[:-1]]

    return var_index_names, con_index_names


def free_variables(model, components, parameters, var_index_names): 
    """
    This prepares the indecies and sizes for the reduced Hessian strucuturing.
    
    The variables are arranged such that the parameters need to be in the right order
    and entered in last (for k_aug).
    
    The commented code lets you place the variables in any order, as long as they are
    still located at the end (important for B and Vd matrix building.)
    
    """
    spectral_vars = []
    
    if components is not None:
        if hasattr(model, 'C') and hasattr(model, 'S'):
            for var in ['C', 'S']:
                spectral_vars += [f'{var}[{k[0]},{k[1]}]' for k in getattr(model, var) if k[1] in components]
                
    col_ind = [var_index_names.index(v) for v in spectral_vars + parameters]
    col_ind_P = [var_index_names.index(name) for name in parameters]
    col_ind_param_hr = [col_ind.index(p) for p in col_ind_P]
    
    return col_ind, col_ind_param_hr


def _build_raw_J_and_H(size):
    """Given the size of the variables and constraints, the Hessian and Jacobian
    can be built using the output files from k_aug
    
    :param tuple size: The m (con) and n (var) size of the Jacobian
    
    :return: The Hessian and Jacobian as a tuple of sparse (coo) matrices
    
    """
    m, n = size
    kaug_files = Path('GJH')
    
    hess_file = kaug_files.joinpath('H_print.txt')
    hess = pd.read_csv(hess_file, delim_whitespace=True, header=None, skipinitialspace=True)
    hess.columns = ['irow', 'jcol', 'vals']
    hess.irow -= 1
    hess.jcol -= 1

    jac_file = kaug_files.joinpath('A_print.txt')
    jac = pd.read_csv(jac_file, delim_whitespace=True, header=None, skipinitialspace=True)
    jac.columns = ['irow', 'jcol', 'vals']
    jac.irow -= 1
    jac.jcol -= 1
    
    J = coo_matrix((jac.vals, (jac.jcol, jac.irow)), shape=(m, n))
    Hess_coo = coo_matrix((hess.vals, (hess.irow, hess.jcol)), shape=(n, n))
    H = Hess_coo + triu(Hess_coo, 1).T
    
    return H, J

def _build_reduced_hessian(size, col_ind, con_ind=None, parameter_set=None, delete_fixed_constraints=False):
    """Constructs the reduced Hessian used in various methods.
    
    
    """
    H, J = _build_raw_J_and_H(size)

    if delete_fixed_constraints:
        dummy_constraints = [f'fix_params_to_global[{k}]' for k in parameter_set]
        jac_row_ind = [con_ind.index(d) for d in dummy_constraints]
        # duals_imp = [duals[i] for i in jac_row_ind]

        J_c = delete_from_csr(J.tocsr(), row_indices=jac_row_ind).tocsc()
        row_indexer = SparseRowIndexer(J_c)
        J_f = row_indexer[col_ind]
        J_f = delete_from_csr(J_f.tocsr(), row_indices=jac_row_ind, col_indices=[])
        J_l = delete_from_csr(J_c.tocsr(), col_indices=col_ind)
    
    else:
        J_c = J.tocsc()
        row_indexer = SparseRowIndexer(J_c.T)
        J_f = row_indexer[col_ind].T
        J_l = delete_from_csr(J_c.tocsr(), col_indices=col_ind)
    
    reduced_hessian, Z_mat = _reduced_hessian_matrix(J_f, J_l, H, col_ind)

    return reduced_hessian

def _reduced_hessian_matrix(F, L, H, col_ind):
    """This calculates the reduced hessian by calculating the null-space based
    on the constraints

    :param csr_matrix F: Rows of the Jacobian related to fixed parameters
    :param csr_matrix L: The remainder of the Jacobian without parameters
    :param csr_matrix H: The sparse Hessian
    :param list col_ind: indices of columns with fixed parameters
    
    :return: sparse version of the reduced Hessian
    :rtype: csr_matrix

    """
    from scipy.sparse.linalg import spsolve
    
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

    return reduced_hessian.todense(), Z_mat


def calculate_inverse_hr(size, col_ind):
    """Calculates the inverse of the reduced Hessian using KKT info (k_aug)
    
    :return H_use: The inverse of the reduced Hessian (parameter rows) 
    :rtype: numpy.ndarray
    
    """
    
    reduced_hessian = _build_reduced_hessian(size, col_ind)
    inv_H_r = np.linalg.inv(reduced_hessian)
    
    return inv_H_r
    
    
def add_warm_start_suffixes(model, use_k_aug=False):
    """Adds suffixed variables to problem

    :param ConcreteModel model: A Pyomo model
    :param bool use_k_aug: Indicates if k_aug solver is being used

    :return: None

    """
    # Ipopt bound multipliers (obtained from solution)
    model.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
    model.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
    # Ipopt bound multipliers (sent to solver)
    model.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
    model.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)
    # Obtain dual solutions from first solve and send to warm start
    model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
    
    if use_k_aug:
        model.dof_v = Suffix(direction=Suffix.EXPORT)
        model.rh_name = Suffix(direction=Suffix.IMPORT)
        
    return None
        
def update_warm_start(model):
    """Updates the suffixed variables for a warmstart

    :param ConcreteModel model: A Pyomo model

    :return: None

    """
    model.ipopt_zL_in.update(model.ipopt_zL_out)
    model.ipopt_zU_in.update(model.ipopt_zU_out)
    
    return None


def calculate_reduced_hessian(model, d=None, optimize=False, parameter_set=[],
                              fix_method=None, rho=10, scaled=True):
    
    if optimize:
        stub = optimize_model(model, 
                              parameter_set=parameter_set, 
                              d=d,
                              fix_method=fix_method,
                              rho=rho,
                              scaled=scaled)
    
    parameter_set_full = define_free_parameters(model, global_params=None, kind='full')
    var_index_names, con_index_names = var_con_data(stub)
    size = (len(con_index_names), len(var_index_names))
    col_ind, col_ind_param_hr = free_variables(model, None, parameter_set_full, var_index_names)
    
    delete_fixed_constraints = fix_method == 'global'
    reduced_hessian = _build_reduced_hessian(size, col_ind, con_index_names, parameter_set, delete_fixed_constraints)
    
    return reduced_hessian

def optimize_model(model, parameter_set=[], d=None, verbose=False, fix_method='fixed',
                    use_bounds=False, variable_name='P', rho=10, scaled=True):
    """Takes the model object and performs the optimization for a reduced Hessian problem.
    
    :param dict d: The current state of the parameters

    :return: None

    """
    from kipet.estimability_tools.parameter_handling import set_scaled_parameter_bounds
    
    if verbose:
        print(f'd: {d}')

    ipopt = SolverFactory('ipopt')
    _tmpfile = 'reduced_hessian'

    if fix_method == 'global':

        if d is not None:
            param_val_dict = {p: d[i] for i, p in enumerate(parameter_set)}
            for k, v in getattr(model, variable_name).items():
                v.set_value(param_val_dict[k])

        add_global_constraints(model, parameter_set, variable_name)

        use_bounds = True
        if use_bounds:
            set_scaled_parameter_bounds(model,
                                        parameter_set=parameter_set,
                                        rho=rho,
                                        scaled=scaled)

    elif fix_method == 'fixed':

        # if hasattr(model, self.global_constraint_name):
        #     self.model_object.del_component(self.global_constraint_name)

        delta = 1e-20
        for k, v in getattr(model, variable_name).items():
            if k in parameter_set:
                ub = v.value
                lb = v.value - delta
                v.setlb(lb)
                v.setub(ub)
                v.unfix()
            else:
                v.fix()
    
    ipopt.solve(model,
                symbolic_solver_labels=True,
                keepfiles=True,
                tee=True,
                logfile=_tmpfile,
                )

    stub = ipopt._problem_files[0][:-3]

    kaug = SolverFactory('k_aug')
    kaug.options["print_kkt"] = ""
    kaug.solve(model, tee=True)

    return stub

def add_global_constraints(model, parameter_set, variable_name):
    """This adds the dummy constraints to the model forcing the local
    parameters to equal the current global parameter values

    :return: None
    
    """
    from pyomo.environ import Constraint, Param, Set
    
    current_set = 'current_p_set'
    global_param_name='d'
    global_constraint_name='fix_params_to_global'
    param_set_name='parameter_names'
    
    if parameter_set is None:
        parameter_set = [p for p in getattr(model, variable_name)]

    # if self.scaled:
    #     global_param_init = {p: 1 for p in parameter_set}
    # else:
    global_param_init = {p: getattr(model, variable_name)[p].value for p in parameter_set}

    for comp in [global_constraint_name, global_param_name, current_set]:
        if hasattr(model, comp):
            model.del_component(comp)

    setattr(model, current_set, Set(initialize=parameter_set))

    setattr(model, global_param_name,
            Param(getattr(model, param_set_name),
                  initialize=global_param_init,
                  mutable=True,
                  ))

    def rule_fix_global_parameters(m, k):
        return getattr(m, variable_name)[k] - getattr(m, global_param_name)[k] == 0

    setattr(model, global_constraint_name,
            Constraint(getattr(model, current_set),
                       rule=rule_fix_global_parameters)
            )

    return None

def delete_from_csr(mat, row_indices=[], col_indices=[]):
    """
    Remove the rows (denoted by ``row_indices``) and columns (denoted by 
    ``col_indices``) from the CSR sparse matrix ``mat``.
    WARNING: Indices of altered axes are reset in the returned matrix

    :param csr_matrix mat: Sparse matrix to delete rows and cols from
    :param list row_indicies: rows to delete
    :param list col_indicies: cols to delete

    :return csr_matrix mat: The sparse matrix with the rows and cols removed

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
        return mat[row_mask][:, col_mask]
    elif len(rows) > 0:
        mask = np.ones(mat.shape[0], dtype=bool)
        mask[rows] = False
        return mat[mask]
    elif len(cols) > 0:
        mask = np.ones(mat.shape[1], dtype=bool)
        mask[cols] = False
        return mat[:, mask]
    else:
        return mat


class SparseRowIndexer:
    """Class used to splice sparse matrices"""

    def __init__(self, matrix):
        data = []
        indices = []
        indptr = []

        _class = 'csr'
        if isinstance(matrix, csc_matrix):
            _class = 'csc'

        self._class = _class
        # Iterating over the rows this way is significantly more efficient
        # than matrix[row_index,:] and matrix.getrow(row_index)
        for row_start, row_end in zip(matrix.indptr[:-1], matrix.indptr[1:]):
            data.append(matrix.data[row_start:row_end])
            indices.append(matrix.indices[row_start:row_end])
            indptr.append(row_end - row_start)  # nnz of the row

        self.data = np.array(data, dtype=object)
        self.indices = np.array(indices, dtype=object)
        self.indptr = np.array(indptr)
        self.n_columns = matrix.shape[1]

    def __getitem__(self, row_selector):
        data = np.concatenate(self.data[row_selector])
        indices = np.concatenate(self.indices[row_selector])
        indptr = np.append(0, np.cumsum(self.indptr[row_selector]))

        shape = [indptr.shape[0] - 1, self.n_columns]

        if self._class == 'csr':
            return csr_matrix((data, indices, indptr), shape=shape)
        else:
            return csr_matrix((data, indices, indptr), shape=shape).T.tocsc()