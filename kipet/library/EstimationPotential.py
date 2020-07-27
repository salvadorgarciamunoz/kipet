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

from kipet.library.data_tools import df_from_pyomo_data
from kipet.library.ParameterEstimator import ParameterEstimator
from kipet.library.PyomoSimulator import PyomoSimulator
from kipet.library.TemplateBuilder import TemplateBuilder
from kipet.library.common.objectives import (
    conc_objective,
    comp_objective,
    )

__author__ = 'Kevin McBride'  #: April 2020
    
class EstimationPotential(ParameterEstimator):
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
        self.verbose = self._options.pop('verbose', False)
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
            
        return None
    
    def plot_results(self):
        """This function plots the profiles from the final model after
        parameter fitting.
        
        Args:
            None
            
        Returns:
            None
            
        """
        line_options = {
            'linewidth' : 3,
            }
        
        marker_options = {
            'linewidth' : 1,
            'markersize' : 10,
            'alpha' : 0.5,
            }
        
        title_options = {
            'fontsize' : 18,
            }
        
        axis_options = {
            'fontsize' : 16,
            }
        
        exp_data = list(self.model.measured_data.keys())
        
        if len(self.model.mixture_components.value) > 0:
            dfz = df_from_pyomo_data(self.model.Z)       
            dfc = None
        
            if len(self.model.mixture_components.value & self.model.measured_data.value) > 0:        
                dfc = df_from_pyomo_data(self.model.C)
            
                for col in dfc.columns:
                    if col not in exp_data:
                        dfc.drop(columns=[col], inplace=True)
            
            plt.figure(figsize=(4,3))
            
            for col in dfz.columns:
                plt.plot(dfz[col], label=col + ' (pred)', **line_options)
                if dfc is not None:
                    if col in dfc.columns:
                        plt.plot(dfc[col], 'o', label=col + ' (exp)', **marker_options)
                    
            plt.xlabel("Time [h]", **axis_options)
            plt.ylabel("Concentration (mol/L)", **axis_options)
            plt.title("Concentration Profile", **title_options)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            
            plt.legend()
            plt.show()
            
        if len(self.model.complementary_states.value) > 0:
            dfx = df_from_pyomo_data(self.model.X)  
            dfu = None
            
            if len(self.model.complementary_states.value & self.model.measured_data.value) > 0:
                dfu = df_from_pyomo_data(self.model.U)  
            
                for col in dfu.columns:
                    if col not in exp_data:
                        dfu.drop(columns=[col], inplace=True)
            
            plt.figure()
            
            for col in dfx.columns:
                plt.plot(dfx[col], label=col + ' (pred)', **line_options)
                if dfu is not None:
                    if col in dfu.columns:
                        plt.plot(dfu[col], 'o', label=col + ' (exp)', **marker_options)
            
            plt.xlabel("Time [h]", **axis_options)
            plt.ylabel("Temperature [K]", **axis_options)       
            plt.title("Complementary State Profiles", **title_options)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
        
            plt.legend()
            plt.show()
    
        return None
    
    def _model_preparation(self):
        """Helper function that should prepare the models when called from the
        main function. Includes the experimental data, sets the objectives,
        simulates to warm start the models if no data is provided, sets up the
        reduced hessian model with "fake data", and discretizes all models

        """
        sim_model = copy.deepcopy(self.model)
        
        if not hasattr(self.model, 'objective'):
            self.model.objective = self._rule_objective(self.model)
        
        #self.model.objective = self._rule_objective(self.model, self.model_builder)
        self.parameter_order = {i : name for i, name in enumerate(self.model.P)}
        
        simulation_data = self.simulation_data
        if simulation_data is None and self.simulate_start:
        
            simulator = PyomoSimulator(sim_model)
            simulator.apply_discretization('dae.collocation',
                                        ncp = self.ncp,
                                        nfe = self.nfe,
                                        scheme = 'LAGRANGE-RADAU')
        
            for k, v in simulator.model.P.items():
                simulator.model.P[k].fix(1)
        
            simulator.model.objective = self._rule_objective(self.model, self.model_builder)
            options = {'solver_opts' : dict(linear_solver='ma57')}
            
            simulation_data = simulator.run_sim('ipopt_sens',
                                              tee=True,
                                              solver_options=options,
                                              )
        
        # The model needs to be discretized
        model_pe = ParameterEstimator(self.model)
        model_pe.apply_discretization('dae.collocation',
                                            ncp = self.ncp,
                                            nfe = self.nfe,
                                            scheme = 'LAGRANGE-RADAU')
        
        
        if hasattr(simulation_data, 'X'):
            #print(f'Here is the X data:\n{simulation_data.X}')
            model_pe.initialize_from_trajectory('X', simulation_data.X)
            model_pe.initialize_from_trajectory('dXdt', simulation_data.dXdt)
        
        if hasattr(simulation_data, 'Z'):
            #print(f'Here is the Z data:\n{simulation_data.Z}')
            model_pe.initialize_from_trajectory('Z', simulation_data.Z)
            model_pe.initialize_from_trajectory('dZdt', simulation_data.dZdt)
        
        for k, v in self.model.P.items():
            ub = self.rho
            lb = 1/self.rho
            self.model.P[k].setlb(lb)
            self.model.P[k].setub(ub)
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
        obj += conc_objective(model)
        obj += comp_objective(model)  
    
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
        Z[col_ind_left, :] = X.todense()
    
        Z_mat = coo_matrix(np.mat(Z)).tocsr()
        Z_mat_T = coo_matrix(np.mat(Z).T).tocsr()
        Hess = Hess_coo.tocsr()
        red_hessian = Z_mat_T * Hess * Z_mat
        
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
