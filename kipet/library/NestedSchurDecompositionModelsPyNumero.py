#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains the nested Schur method for decomposing multiple
experiments into a bi-level optimization problem.

Author: Kevin McBride 2020
"""
import copy
from pathlib import Path
import pickle
import shutil
from string import Template

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.core.base.PyomoModel import ConcreteModel

from pyomo.environ import (
    Constraint, 
    Objective,
    Param, 
    Set,
    SolverFactory,
    Suffix,
    )

from scipy.optimize import (
    Bounds,
    minimize,
    )

from kipet.library.ParameterEstimator import ParameterEstimator
from kipet.library.PyomoSimulator import PyomoSimulator
from kipet.library.KaugReader import KaugReader as KR

global opt_dict
opt_dict = dict()
global opt_count
opt_count = -1

global global_param_name
global global_constraint_name
global parameter_var_name
global global_set_name

# If for some reason your model has these attributes, you will have a problem
global_set_name = 'global_parameter_set'
global_param_name = 'd_params_nsd_globals'
global_constraint_name = 'fix_params_to_global_nsd_constraint'

# Header template
iteration_spacer = Template('\n' + '#'*30 + ' $iter ' + '#'*30 + '\n')

class NestedSchurDecomposition():
    
    """Nested Schur Decomposition approach to parameter fitting using multiple
    experiments
    
    This version takes pyomo Concrrete models instead of templates
    1. This version is working with my examples!
    2. Erase all variables in Spyder before each start if using it!
    
    Handles:
        1. Different experimental measurement times
        2. Different species
        3. Different parameters not included in each experiment (subreactions)
        4. Different numbers of experimental measurements

    """
    def __init__(self, models, d_info, kwargs=None):
        
        """Args:
            models (dict): A dict of pyomo Concrete models
            
            d_info: A dict of the global parameters including the iniital
                value and a tuple of the bounds, i.e. {'p1' : 2.0, (0.0, 4.0)}
        
            kwargs (dict): Optional arguments for the algorithm (incomplete)
        
        """
        # The models should be entered in as a dict (for now)
        self.models_dict = copy.copy(models)
        
        # The global parameter information is needed, especially the bounds
        self.d_info = d_info
        self.d_init = {k: v[0] for k, v in d_info.items()}
        
        # Arrange the kwargs
        self._kwargs = {} if kwargs is None else copy.copy(kwargs)
        
        # Options - inner problem optimization
        self.ncp = self._kwargs.pop('ncp', 3)
        self.nfe = self._kwargs.pop('nfe', 50)
        self.verbose = self._kwargs.pop('verbose', False)
        self.sens = self._kwargs.pop('use_k_aug', True)
        self.parameter_var_name = self._kwargs.pop('parameter_var_name', None)
        self.objective_name = self._kwargs.pop('objective_name', None)
        
        global parameter_var_name
        parameter_var_name = self.parameter_var_name
        
        if self.parameter_var_name is None:
            raise ValueError('NSD requires that the parameter attribute be provided')
        
        # Options - nested Schur decomposition
        self.remove_files = self._kwargs.pop('remove_files', True)
        self.d_init_user = self._kwargs.pop('d_init', True)
        self.method = self._kwargs.pop('method', 'trust-constr')
        
        # Run various assertions that the model is correctly structured
        self._test_models()
        
        # Add the global constraints to the model
        self._add_global_constraints()
        self._prep_models()
        
        # Initialize the opt_dict
        # _inner_problem(self.d_init, self.models_dict, initialize=True)
        
    def _test_models(self):
        """Sanity check on the input models"""
        
        for model in self.models_dict.values():
            
            # Check if the models are even models
            assert(isinstance(model, ConcreteModel) == True)
            
            # Check if the global constraints are correctly implemented
            #assert(hasattr(model, 'd') == True)
            #assert(hasattr(model, 'fix_params_to_global') == True)
            #assert(model.P.extract_values() == model.d.extract_values())
        
            # Check if the model has experimental data (concentration)
            #assert(hasattr(model, 'C') == True)
        
    def _add_global_constraints(self):
        """This adds the dummy constraints to the model forcing the local
        parameters to equal the current global parameter values
        
        """
        
        global global_param_name
        global global_constraint_name
        global global_set_name
        
        for model in self.models_dict.values():
            param_dict = {}
            for param in self.d_info.keys():
                if param in getattr(model, self.parameter_var_name):
                    param_dict.update({param: self.d_info[param][0]})

            setattr(model, global_set_name, Set(initialize=param_dict.keys()))

            setattr(model, global_param_name, Param(getattr(model, global_set_name),
                                  initialize=param_dict,
                                  mutable=True,
                                  ))
            
            def rule_fix_global_parameters(m, k):
                
                return getattr(m, parameter_var_name)[k] - getattr(m, global_param_name)[k] == 0
                
            setattr(model, global_constraint_name, 
            Constraint(getattr(model, global_set_name), rule=rule_fix_global_parameters))
        
            #getattr(model, global_param_name).display()
            #print(getattr(model, global_constraint_name)['k1'].expr.to_string())
            #print(getattr(model, global_constraint_name)['k2'].expr.to_string())
            #getattr(model, global_set_name).display()
        
    def _prep_models(self):
        """Prepare the model for NSD algorithm. Checks discretization,
        checks for an objective named "objective", and adds appropriate
        suffixes for sensitivity analysis.
        
        """
        
        for model in self.models_dict.values():        
            self._check_discretization(model)
            
            if self.objective_name is None:
                if not hasattr(model, 'objective'):
                    model.objective = self._rule_objective(model)
                
            self._prep_model_for_optimization(model)
            
        return None
            
    def _check_discretization(self, model):
        """Checks is the model is discretized and discretizes it in the case
        that it is not
        
        Args:
            model (ConcreteModel): A pyomo ConcreteModel
            
        Returns:
            None
            
        """
        if not model.alltime.get_discretization_info():
        
            model_pe = ParameterEstimator(model)
            model_pe.apply_discretization('dae.collocation',
                                            ncp = self.ncp,
                                            nfe = self.nfe,
                                            scheme = 'LAGRANGE-RADAU')
        
        return None
        
    def _generate_bounds_object(self):
        """Creates the Bounds object needed by SciPy for minimization
        
        """
        lower_bounds = []
        upper_bounds = []
        
        for k, v in self.d_info.items():
            lower_bounds.append(v[1][0])
            upper_bounds.append(v[1][1])
        
        return Bounds(lower_bounds, upper_bounds, True)
        
    def nested_schur_decomposition(self):
        """Here is where the magic happens. This is the outer problem controlled
        by a trust region solver running on scipy. This is the only function that
        the user needs to call to run this thing.
        
        Returns:
            res (scipy.optimize.optimize.OptimizeResult): The results from the 
                trust region optimation (outer problem)
        """    
        #options = copy.copy(self.options)
        #show_plots = options.pop('show_plots', True)
        #tol_0 = options.pop('zero_bound_tol', 1e-8)
        
        print(iteration_spacer.substitute(iter='NSD Start'))
        #_inner_problem(self.d_init, self.models_dict, initialize=True)
    
        d_init = self.d_init #self._generate_initial_d()
        d_bounds = self._generate_bounds_object()
    
        self.d_iter = list()
        def callback(x, *args):
            self.d_iter.append(x)
    
    
        if self.method in ['trust-exact', 'trust-constr']:
        # The args for scipy.optimize.minimize
            fun = _inner_problem
            x0 = list(d_init.values()) #list(d_init.values()) if isinstance(d_init, dict) else d_init
            args = (self.models_dict,)
            jac = _calculate_m
            hess = _calculate_M
            
            callback(x0)
            results = minimize(fun, x0, args=args, method=self.method,
                           jac=jac,
                           hess=hess,
                           callback=callback,
                           bounds=d_bounds,
                           options=dict(gtol=1e-10,
                                      #  initial_tr_radius=0.1,
                                      #  max_tr_radius=0.1
                                        ),
                           )
            self.parameters_opt = {k: results.x[i] for i, k in enumerate(self.d_init.keys())}
            
            
        if self.method in ['newton']:
            x0 = list(d_init.values())
            results = self._run_newton_step(x0, self.models_dict)
            self.parameters_opt = {k: results[i] for i, k in enumerate(self.d_init.keys())}
        
        d_vals = pd.DataFrame(self.d_iter)
        plot_convergence_results(d_vals.values, self.models_dict, d_bounds)
        
        # Clean up the k_aug and pyomo files
        if self.remove_files:
            file_dir = Path.cwd().joinpath('k_aug_output')
            for file in file_dir.iterdir():
                if file.is_dir():
                    shutil.rmtree(file)
        
        
        
        return results, opt_dict
    
    def _run_newton_step(self, d_init, models):
        """This runs a basic Newton step algorithm - use a decent alpha!"""
        
        tol = 1e-6
        alpha = 0.4
        max_iter = 40
        counter = 0
        self.d_iter.append(d_init)
        
        while True:   
        
            _inner_problem(d_init, models, generate_gradients=False)
            M = opt_dict[opt_count]['M']
            m = opt_dict[opt_count]['m']
            d_step = np.linalg.inv(M).dot(-m)
            d_init = [d_init[i] + 0.4*d_step[i] for i, v in enumerate(d_init)]
            self.d_iter.append(d_init)
            
            if max(d_step) <= tol:
                
                print('Terminating sequence: minimum tolerance in step size reached ({tol}).')
                break
            
            if counter == max_iter:
                print('Terminating sequence: maximum number of iterations reached ({max_iter})')
                break
            
            counter += 1
            
        return d_init
            

    def _rule_objective(self, model):
        """This function defines the objective function for the estimability
        
        This is equation 5 from Chen and Biegler 2020. It has the following
        form:
            
        .. math::
            \min J = \frac{1}{2}(\mathbf{w}_m - \mathbf{w})^T V_{\mathbf{w}}^{-1}(\mathbf{w}_m - \mathbf{w})
            
        Originally KIPET was designed to only consider concentration data in
        the estimability, but this version now includes complementary states
        such as reactor and cooling temperatures. If complementary state data
        is included in the model, it is detected and included in the objective
        function.
        
        Args:
            model (pyomo.core.base.PyomoModel.ConcreteModel): This is the pyomo
            model instance for the estimability problem.
                
        Returns:
            obj (pyomo.environ.Objective): This returns the objective function
            for the estimability optimization.
        
        """
        obj = 0

        for k in model.mixture_components & model.measured_data:
            for t, v in model.C.items():
                obj += 0.5*(model.C[t] - model.Z[t]) ** 2 / model.sigma[k]**2
        
        for k in model.complementary_states & model.measured_data:
            for t, v in model.U.items():
                obj += 0.5*(model.X[t] - model.U[t]) ** 2 / model.sigma[k]**2      
    
        return Objective(expr=obj)
    
    def _prep_model_for_optimization(self, model):
        """This function prepares the optimization models with required
        suffixes. This is here because I don't know if this is already in 
        KIPET somewhere else.
        
        Args:
            model (pyomo model): The model of the system
            
        Retuns:
            None
            
        """
        global parameter_var_name
        global global_constraint_name
        
        model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
        model.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
        model.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
        model.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
        model.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)
        model.var_order = Suffix(direction=Suffix.EXPORT)
        model.dcdp = Suffix(direction=Suffix.EXPORT)
        
        count_vars = 1
        for k, v in getattr(model, global_constraint_name).items():
            v.set_suffix_value(model.dcdp, count_vars)    
            count_vars += 1
            
        count_vars = 1
        for k, v in getattr(model, parameter_var_name).items():
            v.set_suffix_value(model.var_order, count_vars)
            count_vars += 1
            
        return None
        
def _optimize(model, d_vals, verbose=False):
    """Solves the optimization problem with optional k_aug sensitivity
    calculations (needed for Nested Schur Decomposition)
    
    Args:
        model (pyomo ConcreteModel): The current model used in parameter
            fitting
            
        d_vals (dict): The dict of global parameter values
        
        verbose (bool): Defaults to false, option to see solver output
        
    Returns:
        model (pyomo ConcreteModel): The model after optimization
    """
    global global_param_name
    global parameter_var_name
    
    delta = 1e-12
    ipopt = SolverFactory('ipopt')
    kaug = SolverFactory('k_aug')
    tmpfile_i = "ipopt_output"

    for k, v in getattr(model, parameter_var_name).items():
        getattr(model, parameter_var_name)[k].unfix()

    for k, v in getattr(model, global_param_name).items():
        getattr(model, global_param_name)[k] = d_vals[k]
    
    results = ipopt.solve(model,
                          symbolic_solver_labels=True,
                          keepfiles=True, 
                          tee=verbose, 
                          logfile=tmpfile_i)

    model.ipopt_zL_in.update(model.ipopt_zL_out)
    model.ipopt_zU_in.update(model.ipopt_zU_out)
    
    return  model

def _inner_problem(d_init_list, models, generate_gradients=False, initialize=False):
    """Calculates the inner problem using the scenario info and the global
    parameters d
    
    Args:
        d_init_last (list): list of the parameter values
        
        models (dict): the dict of pyomo models used as supplemental args
        
        generate_gradients (bool): If the d values do not line up with the 
            current iteration (if M or m is calculated before the inner prob),
            then the inner problem is solved to generate the corrent info
            
        initialize (bool): Option only used in the initial optimization before 
            starting the NSD routine
    
    Returns:
        Either returns None (generating gradients) or the scalar value of the 
        sum of objective function values from the inner problems
        
    """    
    global opt_count
    global opt_dict
    global global_constraint_name
    global parameter_var_name
    global global_param_name
     
    opt_count += 1   
        
    options = {'verbose' : False}
    _models = copy.copy(models) 
  
    m = 0
    M = 0
    Si = []
    Ki = []
    Ei = []
    
    objective_value = 0
    
    print(iteration_spacer.substitute(iter=f'Inner Problem {opt_count}'))
    print(f'Current parameter set: {d_init_list}')
    
    for k, model in _models.items():
        
        valid_parameters = dict(getattr(model, parameter_var_name).items()).keys()
        
        if isinstance(d_init_list, dict):
            d_init = {k: d_init_list[k] for k in valid_parameters}
        else:
            d_init = {param: d_init_list[i] for i, param in enumerate(valid_parameters)}
        
        # Optimize the inner problem
        model_opt = _optimize(model, d_init)
        
        kkt_df, var_ind, con_ind_new = _JH(model_opt)
        duals = [model_opt.dual[getattr(model_opt, global_constraint_name)[key]] for key, val in getattr(model_opt, global_param_name).items()]
        col_ind  = [var_ind.loc[var_ind[0] == f'{parameter_var_name}[{v}]'].index[0] for v in valid_parameters]
        dummy_constraints = _get_dummy_constraints(model_opt)
        dc = [d for d in dummy_constraints]
        
        # Perform the calculations to get M and m
        K = kkt_df.drop(index=dc, columns=dc)
        E = np.zeros((len(dummy_constraints), K.shape[1]))
        K_i_inv = np.linalg.inv(K.values)
         
        for i, indx in enumerate(col_ind):
            E[i, indx] = 1
            
        S = E.dot(K_i_inv).dot(E.T)
        M += np.linalg.inv(S)
        m += np.array(duals)
        objective_value += model_opt.objective.expr()

        Si.append(S)
        Ki.append(K_i_inv)
        Ei.append(E)

    # Save the results in opt_dict - needed for further iterations
    opt_dict[opt_count] = { 'd': d_init_list,
                            'obj': objective_value,
                            'M': M,
                            'm': m,
                            'S': Si,
                            'K_inv': Ki,
                            'E': Ei,
                            } 
    
    if not generate_gradients:
        return objective_value
    else:
        return None

def _KKT_mat(H, A):
    
    KKT_up = pd.merge(H, A.transpose(), left_index=True, right_index=True)
    KKT = pd.concat((KKT_up, A))
    KKT = KKT.fillna(0)
    return KKT

def _JH(model):
    
    nlp = PyomoNLP(model)
    varList = nlp.get_pyomo_variables()
    conList = nlp.get_pyomo_constraints()
    duals = nlp.get_duals()
    
    J = nlp.extract_submatrix_jacobian(pyomo_variables=varList, pyomo_constraints=conList)
    H = nlp.extract_submatrix_hessian_lag(pyomo_variables_rows=varList, pyomo_variables_cols=varList)
    
    var_index_names = [v.name for v in varList]
    con_index_names = [v.name for v in conList]

    J_df = pd.DataFrame(J.todense(), columns=var_index_names, index=con_index_names)
    H_df = pd.DataFrame(H.todense(), columns=var_index_names, index=var_index_names)
    
    var_index_names = pd.DataFrame(var_index_names)
    KKT = _KKT_mat(H_df, J_df)
    
    return KKT, var_index_names, con_index_names

def _get_dummy_constraints(model):
    """Get the locations of the contraints for the local and global parameters
    
    Args:
        models (pyomo ConcreteModel): The current model in the inner problem
        
    Returns:
        dummy_constraints (str): the names of the dummy contraints for the 
            parameters
    
    """
    global global_constraint_name
    global parameter_var_name
    global global_param_name
    
    dummy_constraint_name = global_constraint_name
    dummy_constraint_template = Template(f'{dummy_constraint_name}[$param]')
    parameters = getattr(model, global_param_name).keys()
    dummy_constraints = [dummy_constraint_template.substitute(param=k) for k in parameters]
    
    return dummy_constraints

def _calculate_M(x, scenarios):
    """Calculates the Hessian, M
    This is scipy.optimize.minimize conform
    Checks that the correct data is retrieved
    Needs the global dict to get information from the inner optimization
    
    Args:
        x (list): current parameter values
        
        scenarios (dict): The dict of scenario models
        
    Returns:
        M (np.ndarray): The M matrix from the NSD method
    
    """
    global opt_dict
    global opt_count
    
    if opt_count == 0 or any(opt_dict[opt_count]['d'] != x):
        _inner_problem(x, scenarios, generate_gradients=True)

    M = opt_dict[opt_count]['M']
    return M
    
def _calculate_m(x, scenarios):
    """Calculates the jacobian, m
    This is scipy.optimize.minimize conform
    Checks that the correct data is retrieved
    Needs the global dict to get information from the inner optimization
    
    Args:
        x (list): current parameter values
        
        scenarios (dict): The dict of scenario models
        
    Returns:
        m (np.ndarray): The m matrix from the NSD method
    
    """
    global opt_dict
    global opt_count
    
    if opt_count == 0 or any(opt_dict[opt_count]['d'] != x):
        _inner_problem(x, scenarios, generate_gradients=True)
    
    m = opt_dict[opt_count]['m']
    return m
        
# Not being used at the moement
# def _update_delta_y_and_delta_gamma(del_d, K_i_inv, S_i, E):
#     """Update the y and gamma duals for a warmstart - This is not currently
#     being used, but will be implemented in the future.
#     """
#     global opt_dict
#     global opt_count
    
#     delta_gamma = {k : S_i[k].dot(np.array(list(del_d.values()))) for k in S_i.keys()}
#     delta_y = {k : -1*K_i_inv[k].dot(E.T).dot(delta_gamma[k]) for k in K_i_inv.keys()}
     
#    return delta_gamma, delta_y
    
#%% # # # Plotting for testing purposes # # # #

def _make_plot_data(scenarios, xx, yy, generate_data=False):
    """If a 2 dimensional problem is being used (unlikely), you can use this
    function to generate data for a contour plot of the problem - cool!
    """
    if generate_data:
    
        z = {}
        results = {}
        
        for i, x in enumerate(xx.ravel()):
            for j, y in enumerate(yy.ravel()):
                
                models = {}
                options = {'verbose' : False}
                
                d_initial_guess = {'k1' : x,
                                    'k2' : y,
                                    }
                
                for k, v in scenarios.items():
                    v[0].add_global_parameters(d_initial_guess)
            
                models[k] = ms(v[0], v[1], kwargs=options)
                d_vals = [list(d_initial_guess.values())]
                z[x, y] = [sum([s.model.objective.value() for k, s in models.items()])]
        
        with open('contour_data', 'wb') as f:
            pickle.dump(z, f)

    else:
        with open('contour_data', 'rb') as f:
            z = pickle.load(f)

    return z

def plot_convergence_results(d_vals, models_dict, bounds, show_scenario_plots=False):
    """For 2-D plots of the data, if available - used primarily in testing"""
    
    models = list(models_dict.values())
    dims = max(len(m.P) for m in models)
    
    if dims != 2:
        return None
    
    else:
        spacing = 20
        # bounds = {}
        # for i, v in enumerate(builder_data._parameters_bounds.values()):
        #     bounds[i] = v
        
        bounds = {0: [0, 5],
                  1: [0, 1]}
        
        x = np.arange(bounds[0][0], bounds[0][1], bounds[0][1]/spacing)
        y = np.arange(bounds[1][0], bounds[1][1], bounds[1][1]/spacing)
        xx, yy = np.meshgrid(x, y, sparse=True)
        z = _make_plot_data(models, xx, yy, generate_data=False)
        
        zz = np.array(list(z.values())).reshape((spacing, spacing))
        fig, ax = plt.subplots(1,1)
        
        # To make somehow automatic
        levels = np.arange(0, 3.0e-6, 2.5e-8)
        
        cp = ax.contour(xx.ravel(),yy.ravel(),zz.T, levels=levels)
        fig.colorbar(cp)
        ax.set_title('Multiple Scenario Example')
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
       
        for i, point in enumerate(d_vals):
            if i == 0:
                plt.plot(point[0], point[1], 'x', markersize=20)         
            else:
                plt.plot(point[0], point[1], 'o', color='b')         
            
        points = np.array(d_vals)
        plt.plot(points[:,0], points[:,1], color='b')
        
        plt.plot(2.5, 0.8, 'rx')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()
        
    return None