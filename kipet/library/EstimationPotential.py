"""
A new estimation class for KIPET that gives me the flexibility to do what I
need in order for it to work properly.

@author: Kevin McBride
"""
import copy
from pathlib import Path
from string import Template
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from pyomo.environ import (
    Objective,
    SolverFactory,
    Suffix,
    Constraint,
    Param,
    Set,
    )

from kipet.library.common.read_write_tools import df_from_pyomo_data
from kipet.library.post_model_build.scaling import remove_scaling
from kipet.library.ParameterEstimator import ParameterEstimator
from kipet.library.PyomoSimulator import PyomoSimulator
from kipet.library.TemplateBuilder import TemplateBuilder
from kipet.library.ResultsObject import ResultsObject
from kipet.library.common.VisitorClasses import ReplacementVisitor
from kipet.library.post_model_build.scaling import scale_parameters
from kipet.library.common.objectives import (
    conc_objective,
    comp_objective,
    )

__author__ = 'Kevin McBride'  #: April 2020
    
class EstimationPotential():
    """This class is for estimability analysis. The algorithm here is the one
    presented by Chen and Biegler (accepted AIChE 2020) using the reduced 
    hessian to select the estimable parameters. 

    Attributes:
    
        model_builder (pyomo ConcreteModel): The pyomo model 
    
        simulation_data (pandas.DataFrame): Optional simulation data to use for
            warm starting (Needs testing!)
        
        options (dict): Various options for the esimability algorithm:
        
            nfe (int): The number of finite elements to use in the collocation.
            
            ncp (int): The number of collocation points per finite element.
            
            bound_approach (float): The accepted relative difference for
                determining whether or not a bound is considered active (Step 6).
                
            rho (float): Factor used to determine the lower and upper bounds for
                each parameter in fitting (Step 5).
                
            epsilon (float): The minimum value for parameter values
            
            eta (float): Predetermined cut-off value for accepted std/parameter
                ratios.
                
            max_iter_limit (int): Iteration limits for the estimability algorithm.
            
            verbose (bool): Defaults to False, option to display the progress of
                the algorihm.
                
            debug (bool): Defaults to False, option to ask for user input to 
                proceed during the algorithm.
                
            simulate_start (bool): Option to simulate using the model to 
                warm start the optimization
        
    """

    def __init__(self, model, simulation_data=None, options=None):
        
        # Options handling
        self.options = {} if options is None else options.copy()
        self._options = options.copy()
        
        self.debug = self._options.pop('debug', False)
        self.verbose = self._options.pop('verbose', True)
        self.nfe = self._options.pop('nfe', 50)
        self.ncp = self._options.pop('ncp', 3)
        self.bound_approach = self._options.pop('bound_approach', 1e-2)
        self.rho = self._options.pop('rho', 10)
        self.epsilon = self._options.pop('epsilon', 1e-16)
        self.eta = self._options.pop('eta', 0.1)
        self.max_iter_limit = self._options.pop('max_iter_limit', 20)
        self.simulate_start = self._options.pop('sim_start', False)
        
        # Copy the model
        self.model = copy.deepcopy(model)
        self.simulation_data = simulation_data
        
    def __repr__(self):
        
        repr_str = (f'EstimationPotential({self.model}, simulation_data={"Provided" if self.simulation_data is not None else "None"}, options={self.options})')
        
        return repr_str
        
    def estimate(self):
        """This performs the estimability analysis based on the method
        developed in Chen and Biegler 2020 AIChE...
        
        TODO:
            1. Check method in the last step for clarity in the methodology
            (does not affect the implementation!)
            
            2. Look at the optimization code already in KIPET and see if you
            can use it for anything coded here.
            
            3. Make sure nothing is cicular!
        
        Args:
            None
            
        Returns:
            None
            
        """
        flag = False
        step = Template('\n' + '*' * 20 + ' Step $number ' + '*' * 20)
        
        self._model_preparation()
        
        # Step 1
        if self.verbose:
            print(step.substitute(number=1))
            print('Initializing N_pre and N_curr\n')
        
        N_pre = len(self.parameter_order)
        N_curr = len(self.parameter_order)
        
        # Step 2
        if self.verbose:
            print(step.substitute(number=2))
            print('Initialize Se and Sf\n')
        
        Se = list(self.parameter_order.values())
        Sf = []
        
        if self.verbose:
            print(f'\nSe: {Se}')
            print(f'Sf: {Sf}\n')
        
        # Step 3 - Calculate the reduced hessian for the initial stage
        if self.verbose:
            print(step.substitute(number=3))
            print('Calculating the Reduced Hessian for the initial parameter set\n')
        
        reduced_hessian = self._calculate_reduced_hessian(Se, Sf, verbose=self.verbose)

        if self.debug:
            input("Press Enter to continue...")
    
        # Step 4 - Rank the parameters using Gauss-Jordan Elimination
        if self.verbose:
            print(step.substitute(number=4))
            print('Ranking parameters for estimability and moving to Step 5')
        
        Se, Sf = self._rank_parameters(reduced_hessian, Se)
        
        if len(Se) >= N_curr:
            number_of_parameters_to_move = len(Se) - N_curr + 1
            for i in range(number_of_parameters_to_move):
                Sf.insert(0, Se.pop()) 
            
            N_pre = len(Se)
            N_curr = len(Se)
       
        if self.verbose:
            print(f'\nThe updated parameter sets are:\nSe: {Se}\nSf: {Sf}')
        
        if self.debug:
            input("Press Enter to continue...")

        # Step 5 - Optimize the estimable parameters
        outer_iteration_counter = 0
        
        while True:
        
            if outer_iteration_counter > self.max_iter_limit:
                print('Maximum iteration limit reached - check the model!')
                break
            
            inner_iteration_counter = 0
            
            while True:
            
                if inner_iteration_counter > self.max_iter_limit:
                    print('Maximum iteration limit reached - check the model!')
                    break
                
                if self.verbose:
                    print(step.substitute(number=5))
                    print('Optimizing the estimable parameters\n')
                    
                for free_param in Se:
                    self.model.P[free_param].unfix()
                    
                for fixed_param in Sf:
                    self.model.P[fixed_param].fix(1)
                
                ipopt = SolverFactory('ipopt')
                ipopt.solve(self.model, tee=self.verbose)
                if self.verbose:
                    self.model.P.display()
                    
                # Step 6 - Check for active bounds
                number_of_active_bounds = 0
                
                if self.verbose:
                    print(step.substitute(number=6))
                    print('Checking for active bounds\n')
                
                for key, param in self.model.P.items():
                    if (param.value-param.lb)/param.lb <= self.bound_approach or (param.ub - param.value)/param.value <= self.bound_approach:
                        number_of_active_bounds += 1
                        if self.verbose:
                            print('There is at least one active bound - updating parameters and optimizing again\n')
                        break
                        
                for k, v in self.model.K.items():
                    self.model.K[k] = self.model.K[k] * self.model.P[k].value
                    self.model.P[k].set_value(1)
                    print(self.model.K.display())
                    print(self.model.P.display())
                    
                
                if number_of_active_bounds == 0:
                    if self.verbose:
                        print('There are no active bounds, moving to Step 7')
                    break
                    if self.debug:    
                        input("Press Enter to continue...")
                    
                inner_iteration_counter += 1
                if self.debug:
                    input("Press Enter to continue...")
                
            if self.debug:
                self.model.P.display()
                self.model.K.display()
  
            reduced_hessian = self._calculate_reduced_hessian(Se, Sf, verbose=False)
            
            # Step 7 - Check the ratios of the parameter std to value
            if self.verbose:
                print(step.substitute(number=7))
                print('Checking the ratios of each parameter in Se')
            
            ratios, eigvals = self._parameter_ratios(reduced_hessian, Se)
            ratios_satisfied = max(ratios) < self.eta
        
            if ratios_satisfied:
                if self.verbose:
                    print(f'Step 7 passed, all paramater ratios are less than provided tolerance {self.eta}, moving to Step 10')
                    if self.debug:
                        input("Press Enter to continue...")

                    # Step 10 - Check the current number of free parameters
                    print(step.substitute(number=10))
                    print(f'N_curr = {N_curr}, N_pre = {N_pre}, N_param = {len(self.model.P)}')
        
                
                if (N_curr == (N_pre - 1)) or (N_curr == len(self.model.P)):
                    if self.verbose:
                        print('Step 10 passed, moving to Step 11, the procedure is finished')
                        print(f'Se: {Se}')
                        print(f'Sf: {Sf}')
                    break
                else:
                    if self.verbose:
                        print('Step 10 failed, moving first parameter from Sf to Se and moving to Step 5')
                    Se.append(Sf.pop(0))
                    N_pre = N_curr
                    N_curr = N_curr + 1
                    if self.debug:
                        input("Press Enter to continue...")
                
            else:
                # Step 8 - Compare number of current estimable parameter with previous iteration
                if self.verbose:
                    print('Step 7 failed, moving on to Step 8')
                    print(step.substitute(number=8))
                    print('Comparing the current number of parameters in Se with the previous number')
                    print(f'N_curr = {N_curr}, N_pre = {N_pre}, N_param = {len(self.model.P)}')
                if N_curr == (N_pre + 1):
                    Sf.insert(0, Se.pop())
                    if self.verbose:
                        print('Step 8 passed, moving to Step 11, reloading last Se the procedure is finished')
                        print(f'Se: {Se}')
                        print(f'Sf: {Sf}')
                    break
                else:
                    # Step 9 - Check the inequality condition given by Eq. 27
                    if self.verbose:
                        print('Step 8 failed, moving to Step 9\n')
                        print(step.substitute(number=9))
                        print('Calculating the inequality from Eq. 27 in Chen and Biegler 2020')
                    if sum(1.0/eigvals)<sum((np.array([self.model.P[v].value for v in Se]))**2)*(self.eta**2) and flag == True:
                        Sf.insert(0, Se.pop())
                        if self.verbose:
                            print('Step 9 passed, moving last parameter from Se into Sf and moving to Step 5')
                            print(f'Se: {Se}')
                            print(f'Sf: {Sf}')
                        N_pre = N_curr
                        N_curr = N_curr - 1

                        if self.debug:
                            input("Press Enter to continue...")
                    else:
                        # Step 2a - Reset the parameter vectors (all in Se)
                        if self.verbose:
                            print('Step 9 failed, moving to Step 2\n')
                            print(step.substitute(number=2))
                            print('Reseting the parameter vectors')
                        flag = True
                        Se = list(self.parameter_order.values())
                        Sf = []
            
                        if self.debug:
                            input("Press Enter to continue...")
                        
                        if self.verbose:
                            # Step 3a - Recalculate the reduced hessian
                            print(step.substitute(number=3))
                            print('Recalculating the reduced hessian')
                            if self.debug:
                                print(f'Input model:\n')
                                self.model.P.display()
                  
                        reduced_hessian = self._calculate_reduced_hessian(Se, Sf, verbose=False)
                        
                        if self.debug:
                            input("Press Enter to continue...")
                        if self.verbose:
                            # Step 4 - Rank the updated parameters using Gauss-Jordan elimination
                            print(step.substitute(number=4))
                            print('Ranking the parameters (limited by N_curr)')
                        
                        Se, Sf = self._rank_parameters(reduced_hessian, Se)
                        
                        if len(Se) >= N_curr:
                            number_of_parameters_to_move = len(Se) - N_curr + 1
                            for i in range(number_of_parameters_to_move):
                                Sf.insert(0, Se.pop())   
                            
                        N_pre = len(Se)
                        N_curr = len(Se)
                           
                        if self.verbose:
                            print(f'The parameter sets are:\nSe: {Se}\nSf: {Sf}\nN_pre: {N_pre}\nN_curr: {N_curr}')
                        if self.debug:
                            input("Press Enter to continue...")

            outer_iteration_counter += 1                
        
        print(step.substitute(number='Finished'))
        print('\nThe final parameter values are:\n')
        if self.model.K is not None:
            self.model.K.pprint()
        else:
            self.model.P.pprint()
            
        results = self._get_results(Se)
            
        return results, self.model
    
    def _get_results(self, Se):
        
        scaled_parameter_var = 'K'
    
        results = ResultsObject()
        
        results.estimable_parameters = Se
        
        #results.objective = self.objective_value
        #results.parameter_covariance = self.cov_mat

        # if self._spectra_given:
        #     results.load_from_pyomo_model(self.model,
        #                                   to_load=['Z', 'dZdt', 'X', 'dXdt', 'C', 'S', 'Y'])
        #     if hasattr(self, '_abs_components'):
        #         results.load_from_pyomo_model(self.model,
        #                                       to_load=['Cs'])
        #     if hasattr(self, 'huplc_absorbing'):
        #         results.load_from_pyomo_model(self.model,
        #                                       to_load=['Dhat_bar'])
     
        # elif self._concentration_given:
        results.load_from_pyomo_model(self.model,
                                          to_load=['Z', 'dZdt', 'X', 'U', 'dXdt', 'Cm', 'Y'])
        # else:
        #     raise RuntimeError(
        #         'Must either provide concentration data or spectra in order to solve the parameter estimation problem')

        # if self._spectra_given:
        #     self.compute_D_given_SC(results)

        if hasattr(self.model, scaled_parameter_var): 
            results.P = {name: self.model.P[name].value*getattr(self.model, scaled_parameter_var)[name].value for name in self.model.parameter_names}
        else:
            results.P = {name: self.model.P[name].value for name in self.model.parameter_names}

        # if hasattr(self.model, 'Pinit'):
        #     param_valsinit = dict()
        #     for name in self.model.initparameter_names:
        #         param_valsinit[name] = self.model.init_conditions[name].value
        #     results.Pinit = param_valsinit

        # if self.termination_condition!=None and self.termination_condition!=TerminationCondition.optimal:
        #     raise Exception("The current iteration was unsuccessful.")
        # else:
        #     if self._estimability == True:
        #         return self.hessian, results
        #     else:
        #         return results

        return results
    
    def _model_preparation(self):
        """Helper function that should prepare the models when called from the
        main function. Includes the experimental data, sets the objectives,
        simulates to warm start the models if no data is provided, sets up the
        reduced hessian model with "fake data", and discretizes all models

        """
        sim_model = copy.deepcopy(self.model)
        
        if not hasattr(self.model, 'objective'):
            self.model.objective = self._rule_objective(self.model)
        
        self.parameter_order = {i : name for i, name in enumerate(self.model.P)}
        
        # simulation_data = self.simulation_data
        # if simulation_data is None and self.simulate_start:
        
        #     simulator = PyomoSimulator(sim_model)
        #     simulator.apply_discretization('dae.collocation',
        #                                 ncp = self.ncp,
        #                                 nfe = self.nfe,
        #                                 scheme = 'LAGRANGE-RADAU')
        
        #     for k, v in simulator.model.P.items():
        #         simulator.model.P[k].fix(1)
        
        #     simulator.model.objective = self._rule_objective(self.model, self.model_builder)
        #     options = {'solver_opts' : dict(linear_solver='ma57')}
            
        #     simulation_data = simulator.run_sim('ipopt',
        #                                       tee=True,
        #                                       solver_options=options,
        #                                       )
        # print(type(self.model))
        
        
        
        # The model needs to be discretized
        model_pe = ParameterEstimator(self.model)
        model_pe.apply_discretization('dae.collocation',
                                      ncp = self.ncp,
                                      nfe = self.nfe,
                                      scheme = 'LAGRANGE-RADAU')
        
        scale_parameters(self.model)
            
        for k, v in self.model.P.items():
            ub = self.rho
            lb = 1/self.rho
            self.model.P[k].setlb(lb)
            self.model.P[k].setub(ub)
            self.model.P[k].set_value(1)
            self.model.P[k].unfix()
     
        return None
    
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
        #if model.mixture_components & model.measured_data:
        obj += conc_objective(model) 
        #if model.complementary_states & model.measured_data:
        obj += comp_objective(model)
        # obj += conc_objective(model)
        #obj += comp_objective(model)  
    
        return Objective(expr=obj)
   
    def _get_kkt_info(self, model):
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
        nlp = PyomoNLP(model)
        varList = nlp.get_pyomo_variables()
        conList = nlp.get_pyomo_constraints()
        
        J = nlp.extract_submatrix_jacobian(pyomo_variables=varList, pyomo_constraints=conList)
        H = nlp.extract_submatrix_hessian_lag(pyomo_variables_rows=varList, pyomo_variables_cols=varList)
        
        var_index_names = [v.name for v in varList]
        con_index_names = [v.name for v in conList]
    
        J_df = pd.DataFrame(J.todense(), columns=var_index_names, index=con_index_names)
        H_df = pd.DataFrame(H.todense(), columns=var_index_names, index=var_index_names)
        
        var_index_names = pd.DataFrame(var_index_names)
        
        KKT_up = pd.merge(H_df, J_df.transpose(), left_index=True, right_index=True)
        KKT = pd.concat((KKT_up, J_df))
        KKT = KKT.fillna(0)
        
        return KKT, H_df, J_df, var_index_names, con_index_names
    
    def _add_global_constraints(self, model, Se):
        """This adds the dummy constraints to the model forcing the local
        parameters to equal the current global parameter values
        
        """
        global_param_name = 'd'
        global_constraint_name = 'fix_params_to_global'
        param_set_name = 'parameter_names'
        
        setattr(model, 'current_p_set', Set(initialize=Se))


        setattr(model, global_param_name, Param(getattr(model, param_set_name),
                              initialize=1,
                              mutable=True,
                              ))
        
        def rule_fix_global_parameters(m, k):
            
            return getattr(m, 'P')[k] - getattr(m, global_param_name)[k] == 0
            
        setattr(model, global_constraint_name, 
        Constraint(getattr(model, 'current_p_set'), rule=rule_fix_global_parameters))
    
    def _calculate_reduced_hessian(self, Se, Sf, verbose=False):
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
        delta = 1e-12
        n_free = len(Se)
        ipopt = SolverFactory('ipopt')
        tmpfile_i = "ipopt_output"
        
        rh_model = copy.deepcopy(self.model)
        
        if hasattr(rh_model, 'fix_params_to_global'):
            rh_model.del_component('fix_params_to_global')
        
        self._add_global_constraints(rh_model, Se)
        
        for k, v in rh_model.P.items():
            ub = self.rho
            lb = 1/self.rho
            rh_model.P[k].setlb(lb)
            rh_model.P[k].setub(ub)
            rh_model.P[k].unfix()
        
        for fixed_param in Sf:
            rh_model.P[fixed_param].fix(1)
        
        ipopt.solve(rh_model, symbolic_solver_labels=True, keepfiles=True, tee=True, logfile=tmpfile_i)

        with open(tmpfile_i, 'r') as f:
            output_string = f.read()
        
        stub = output_string.split('\n')[0].split(',')[1][2:-4]
        col_file = Path(stub + '.col')
        
        kkt_df, hess, jac, var_ind, con_ind_new = self._get_kkt_info(rh_model)
       
        dummy_constraints = [f'fix_params_to_global[{k}]' for k in Se]
        jac = jac.drop(index=dummy_constraints)
        col_ind  = [var_ind.loc[var_ind[0] == f'P[{v}]'].index[0] for v in Se]
        Jac_coo = coo_matrix(jac.values)
        Hess_coo = coo_matrix(hess.values)
        Jac = Jac_coo.todense()
        Jac_f = Jac[:, col_ind]
        Jac_l = np.delete(Jac, col_ind, axis=1)
        
        m, n = jac.shape
        
        X = spsolve(coo_matrix(np.mat(Jac_l)).tocsc(), coo_matrix(np.mat(-Jac_f)).tocsc())
        col_ind_left = list(set(range(n)).difference(set(col_ind)))
        col_ind_left.sort()
        
        Z = np.zeros([n, n_free])
        Z[col_ind, :] = np.eye(n_free)
        
        if not isinstance(X, np.ndarray):
            X = X.todense()
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            
        Z[col_ind_left, :] = X
    
        Z_mat = coo_matrix(np.mat(Z)).tocsr()
        Z_mat_T = coo_matrix(np.mat(Z).T).tocsr()
        Hess = Hess_coo.tocsr()
        red_hessian = Z_mat_T * Hess * Z_mat
    
        print(red_hessian.todense())
        
        return red_hessian.todense()
    
    def _parameter_ratios(self, reduced_hessian, Se):
        """This is Eq. 26 from Chen and Biegler 2020 where the ratio of the 
        standard deviation for each parameter is calculated.
        
        Args:
            reduced_hessian (np.ndarray): The current reduced hessian
            
            Se (list): The list of free parameters
            
        Returns:
            rp {np.ndarray}: The ratio of the predicted standard deviation to the parameter
            value
            
            eigenvalues (np.ndarray): The eigenvalues of the parameters
            
        """
        eigenvalues, eigenvectors = np.linalg.eigh(reduced_hessian)
        red_hess_inv = np.dot(np.dot(eigenvectors, np.diag(1.0/abs(eigenvalues))), eigenvectors.T)
        d = red_hess_inv.diagonal()
        d_sqrt = np.asarray(np.sqrt(d)).ravel()
        rp = [d_sqrt[i]/max(self.epsilon, self.model.P[k].value) for i, k in enumerate(Se)]
        
        return rp, eigenvalues
     
    def _rank_parameters(self, reduced_hessian, param_list):
        """Performs the parameter ranking based using the Gauss-Jordan
        elimination procedure.
        
        Args:
            reduced_hessian (numpy.ndarray): Array of the reduced hessian.
            
            param_list (list): The list of parameters currently in Se.
        
        Returns:
            Se_update (list): The updated list of parameters in Se.
            
            Sf_update (list): The updated list of parameters if Sf.
        
        """
        eigenvector_tolerance = 1e-15
        parameter_tolerance = 1e-12
        squared_term_1 = 0
        squared_term_2 = 0
        Sf_update = []
        Se_update = []
        M = {}
        
        param_set = set(param_list)
        param_elim = set()
        eigenvalues, U = np.linalg.eigh(reduced_hessian)
        
        df_eigs = pd.DataFrame(np.diag(eigenvalues),
                    index=param_list,
                    columns=[i for i, x in enumerate(param_list)])
        df_U_gj = pd.DataFrame(U,
                    index=param_list,
                    columns=[i for i, x in enumerate(param_list)])
        
        # Gauss-Jordan elimination
        for i, p in enumerate(param_list):
            
            # Since they are already ranked, just use the index
            piv_col = i         #df_eigs[df_eigs.abs() > 1e-20].min().idxmin()
            piv_row = df_U_gj.loc[:, piv_col].abs().idxmax()
            piv = (piv_row, piv_col)
            rows = list(param_set.difference(param_elim))
            M[i] =  piv_row
            df_U_gj = self._gauss_jordan_step(df_U_gj, piv, rows)
            param_elim.add(piv_row)
            df_eigs.drop(index=[param_list[piv_col]], inplace=True)
            df_eigs.drop(columns=[piv_col], inplace=True)

        # Parameter ranking
        eigenvalues, eigenvectors = np.linalg.eigh(reduced_hessian)
        ranked_parameters = {k: M[abs(len(M)-1-k)] for k in M.keys()}
        
        for k, v in ranked_parameters.items():
        
            name = v.split('[')[-1].split(']')[0]
            squared_term_1 += abs(1/max(eigenvalues[-(k+1)], parameter_tolerance))
            squared_term_2 += (self.eta**2*max(abs(self.model.P[name].value), self.epsilon)**2)
                        
            if squared_term_1 >= squared_term_2:
                Sf_update.append(name)
            else:
                Se_update.append(name)
        
        return Se_update, Sf_update

    @staticmethod
    def _gauss_jordan_step(df_U_update, pivot, rows):
        """Perfoms the Gauss-Jordan Elminiation step in W. Chen's method
        
        Args:
            df_U_update: A pandas DataFrame instance of the U matrix from the
                eigenvalue decomposition of the reduced hessian (can be obtained
                through the HessianObject).
            
            pivot: The element where to perform the elimination.
            
            rows: A set containing the rows that have already been eliminated.
            
        Returns:
            The df_U_update DataFrame is returned after one row is eliminated.
       
        """
        if isinstance(pivot, tuple):
            pivot=dict(r=pivot[0],
                     c=pivot[1]
                     )

        uij = df_U_update.loc[pivot['r'], pivot['c']]
    
        for col in df_U_update.columns:
            if col == pivot['c']:
                continue
            
            factor = df_U_update.loc[pivot['r'], col]/uij
            
            for row in rows:
                if row not in rows:
                    continue
                else:    
                    df_U_update.loc[row, col] -= factor*df_U_update.loc[row, pivot['c']]
                
        df_U_update[abs(df_U_update) < 1e-15] = 0
        df_U_update.loc[pivot['r'], pivot['c']] = 1
        
        return df_U_update

# def reduce_models(models_dict_provided,
#                   parameter_dict=None,
#                   method='reduced_hessian',
#                   simulation_data=None,
#                   ):
#     """Uses the EstimationPotential module to find out which parameters
#     can be estimated using each experiment and reduces the model
#     accordingly
    
#     This can take a single model or a dict of models. It then proceeds to 
#     find the global set of estimable parameters as well. These are returned
#     in the parameter dict that contains the names of the global set, the
#     model specific parameter sets, and a dict of parameter initial values and
#     bounds that can be used in further methods.
    
#     Args:
#         models_dict_provided (dict): model or dict of models
        
#         parameter_dict (dict): parameters and their initial values
        
#         method (str): model reduction method
        
#         simulation_data (pd.DataFrame): simulation data used for a warmstart
        
#     Returns:
#         models_dict_reduced (dict): dict of reduced models
        
#         parameter_data (dict): parameter data that may be useful
        
#         Example:
            
#         {'names': ['Cfa', 'rho', 'ER', 'k', 'Tfc'],
#          'esp_params_model': {'model_1': ['Cfa', 'Tfc', 'ER', 'rho', 'k']},
#          'initial_values': {'Cfa': (2490.7798699208106, (0, 5000)),
#           'rho': (1335.058139853457, (800, 2000)),
#           'ER': (253.78019656674874, (0.0, 500)),
#           'k': (2.4789686423018376, (0.0, 10)),
#           'Tfc': (262.9381531048641, (250, 400))}}
    
#     """
#     if not isinstance(models_dict_provided, dict):
#         try:
#             models_dict_provided = [models_dict_provided]
#             models_dict_provided = {f'model_{i + 1}': model for i, model in enumerate(models_dict_provided)}
#         except:
#             raise ValueError('You passed something other than a model or a dict of models')
    
#     list_of_methods = ['reduced_hessian']
    
#     if method not in list_of_methods:
#         raise ValueError(f'The model reduction method must be one of the following: {", ".join(list_of_methods)}')
    
#     if method == 'reduced_hessian':
#         if parameter_dict is None:
#             raise ValueError('The reduced Hessian parameter selection method requires initial parameter values')
        
#     models_dict = copy.deepcopy(models_dict_provided)
#     parameters = parameter_dict
    
#     all_param = set()
#     all_param.update(p for p in parameters.keys())
    
#     options = {
#             'verbose' : True,
#                     }
    
#     # Loop through to perform EP on each model
#     params_est = {}
#     results = {}
#     reduced_model = {}
#     set_of_est_params = set()
#     for name, model in models_dict.items():
        
#         if method == 'reduced_hessian':
#             print(f'Starting EP analysis of {name}')
#             est_param = EstimationPotential(model, simulation_data=simulation_data, options=options)
#             params_est[name], results[name], reduced_model[name] = est_param.estimate()
#             #print(est_param.model.objective.expr.to_string())
#     # Add model's estimable parameters to global set
#     for param_set in params_est.values():
#         set_of_est_params.update(param_set)
    
    
#     mod = replace_non_estimable_parameters(reduced_model['model_1'], set_of_est_params)
#     #mod = reduced_model['model_1']
    
#     return params_est, results, mod, set_of_est_params
    
    # models_dict_reduced = {}
    
    # # How does this look with MP?

    # # Remove the non-estimable parameters from the odes
    
    # for key, model in reduced_model.items():
    #     print(model)
    #     update_set = set_of_est_params
    
def reduce_model(model, options=None, **kwargs):
    """Reduces a single model using the reduced hessian parameter selection
    method. It takes a pyomo ConcreteModel using P as the parameters to be fit
    and K as the scaled parameter values.
    
    Args:
        model (ConcreteModel): The full model to be reduced
        
        simulation_data (ResultsObject): simulation data for initialization
        
        options (dict): defaults to None, for future option implementations
        
    Returns:
        results (ResultsObject): returns the results from the parameter
            selection and optimization
        reduced_model (ConcreteModel): returns the reduced model with full
            parameter set
    
    """
    simulation_data = kwargs.get('simulation_data', None)
    replace = kwargs.get('replace', True)
    no_scaling = kwargs.get('no_scaling', True)
    
    if options is None:
        options = {}
        
    orig_bounds = {k: v.bounds for k, v in model.P.items()}
        
    est_param = EstimationPotential(model, simulation_data=None, options=options)
    results, reduced_model = est_param.estimate()
    
    if replace:
        reduced_model = replace_non_estimable_parameters(reduced_model, results.estimable_parameters)
    
    if no_scaling:
        remove_scaling(reduced_model, bounds=orig_bounds)
        
    return results, reduced_model

def replace_non_estimable_parameters(model, set_of_est_params):
    """Takes a model and a set of estimable parameters and removes the 
    unestimable parameters from the model by fixing them to their current 
    values in the model
    
    Args:
        model (ConcreteModel): The full model to be reduced
        
        set_of_est_params (set): Parameters found to be estimable
        
    Returns:
        model (ConcreteModel): The model with parameters replaced
        
    """
    all_model_params = set([k for k in model.P.keys()])
    params_to_change = all_model_params.difference(set_of_est_params)
    
    for param in params_to_change:   
        if param in model.P.keys():
            if hasattr(model, 'K'):
                change_value = model.K[param].value
            else:
                change_value = model.P[param].value
        
            for k, v in model.odes.items():
                ep_updated_expr = _update_expression(v.body, model.P[param], change_value)
                if hasattr(model, 'K'):
                    ep_updated_expr = _update_expression(ep_updated_expr, model.K[param], 1)
                model.odes[k] = ep_updated_expr == 0
    
            model.parameter_names.remove(param)
            del model.P[param]
            if hasattr(model, 'K'):
                del model.K[param]

    return model


#%%
    # # Calculate initial values based on averages of EP output
    # initial_values = pd.DataFrame(np.zeros((len(models_dict), len(set_of_est_params))), index=models_dict.keys(), columns=list(set_of_est_params))

    # for exp, param_data in params_est.items(): 
    #     for param in param_data:
    #         initial_values.loc[exp, param] = param_data[param]
    
    # dividers = dict(zip(initial_values.columns, np.count_nonzero(initial_values, axis=0)))
    
    # init_val_sum = initial_values.sum()
    
    # for param in dividers.keys():
    #     init_val_sum.loc[param] = init_val_sum.loc[param]/dividers[param]
    
    # init_vals = init_val_sum.to_dict()
    # init_bounds = {p: parameters[p][1] for p in parameters.keys() if p in set_of_est_params}
    
    # # Redeclare the d_init_guess values using the new values provided by EP
    # d_init_guess = {p: (init_vals[p], init_bounds[p]) for p in init_bounds.keys()}
    
    # new_parameter_data = d_init_guess
    # new_initial_values = {k: v[0] for k, v in new_parameter_data.items()}
    
    # # The parameter names need to be updated as well
    # parameter_names = list(new_initial_values.keys())
    # pe_dict = {k: list(v.keys()) for k, v in params_est.items()}
    
    # # The actual model is not being output -- change this
    # for model in models_dict_reduced.values():
    #     for param, value in model.K.items():
    #         model.K[param].set_value(new_parameter_data[param][0])
    
    # results_data = {
    #     'reduced_model': models_dict_reduced,
    #     'initial_values': new_parameter_data, 
    #     'results': results,
    #     }

    # return results_data


def _update_expression(expr, replacement_param, change_value):
    """Takes the noparam_infon-estiambale parameter and replaces it with its intitial
    value
    
    Args:
        expr (pyomo constraint expr): the target ode constraint
        
        replacement_param (str): the non-estimable parameter to replace
        
        change_value (float): initial value for the above parameter
        
    Returns:
        new_expr (pyomo constraint expr): updated constraints with the
            desired parameter replaced with a float
    
    """
    visitor = ReplacementVisitor()
    visitor.change_replacement(change_value)
    visitor.change_suspect(id(replacement_param))
    new_expr = visitor.dfs_postorder_stack(expr)
    
    return new_expr