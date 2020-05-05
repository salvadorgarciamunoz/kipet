#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A new estimation class for KIPET that gives me the flexibility to do what I
need in order for it to work properly.

This should allow one to add a builder template and use it for further purposes
so that it does not screw up when trying to use simulation for the reduced
hessian calculation and then performing the parameter fitting. The issue arose 
because what I need is an abstract model, not a concrete one. It is too much
effort to try and replace the current TemplateBuilder.create_pyomo_model for
this purpose (or is it?).

TODO: Make different sized exp data for each component fit into the
framework somehow. Perhaps a mixed finite element arrangement with specific
properties showing up at the right places in the objective function?

@author: Kevin McBride
"""

import copy
from pathlib import Path
from string import Template
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from pyomo.environ import (
    Objective,
    SolverFactory,
    Suffix,
    )

from kipet.library.ParameterEstimator import ParameterEstimator
from kipet.library.PyomoSimulator import PyomoSimulator
from kipet.library.TemplateBuilder import TemplateBuilder

__author__ = 'Kevin McBride'  #: April 2020
    
class EstimationPotential(ParameterEstimator):
    """This class is for estimability analysis. The algorithm here is the one
    presented by Chen and Biegler (accepted AIChE 2020) using the reduced 
    hessian to select the estimable parameters. 

    Attributes:
    
        model_builder (TemplateBuilder): The template for the model (do not use
        created model!).
        
        exp_data (pandas.DataFrame): The experimental data as a dataframe with
            the column headers as the variable name.
            
        times (Array-like): The start and the end times for the system.
        
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
            
        simulation_data (pandas.DataFrame): Optional simulation data to use for
            warm starting (Needs testing!)
            
        kwargs (dict): Optional arguments, at the moment there is no use for
            any additional arguments
        
    """
    
    def __init__(self, model_builder, exp_data, times=None, nfe=None,
                 ncp=None, bound_approach=1e-2, rho=10, epsilon=1e-16, eta=0.1,
                 max_iter_limit=20, verbose=False, debug=False, 
                 simulation_data=None, simulate_start=False, kwargs=None):
        
        if not isinstance(model_builder, TemplateBuilder):
            raise TypeError('A TemplateBuilder instance is required as the first argument.')
            
        if not isinstance(exp_data, pd.DataFrame):
            raise TypeError('Experimental data needs to be a pandas DataFrame object')
            
        model_builder.set_parameter_scaling(True)
        self.model_builder = copy.deepcopy(model_builder)
        self.rh_model_builder = copy.deepcopy(model_builder)
        
        self.debug = debug
        self.verbose = verbose if not self.debug else True
        self.kwargs = {} if kwargs is None else kwargs.copy()
        
        if nfe is None:
            self.nfe = 50
            msg = (f'The number of finite elements was not defined, setting nfe = {self.nfe}')
            warnings.warn(msg, UserWarning)
        else:
            self.nfe = nfe
            
        if ncp is None:
            self.ncp = 3
            msg = (f'The number of collocation points was not defined, setting ncp = {self.ncp}')
            warnings.warn(msg, UserWarning)
        else:
            self.ncp = ncp
            
        if self.model_builder._times is not None:
            self.times = self.model_builder._times
        else:
            if times is not None:
                try:
                    _ = (t for t in times)
                except TypeError:
                    print(f'The "times" attribute is not iterable')
                self.times = times
            else:
                raise ValueError('A start and end time must be provided')
        
        self.start_time = self.times[0]
        self.end_time = self.times[1]
        self.bound_approach = bound_approach
        self.rho = rho
        self.epsilon = epsilon
        self.eta = eta
        self.max_iter_limit = max_iter_limit
        self.exp_data = exp_data
        self.simulation_data = simulation_data
        self.simulate_start = simulate_start
        
    def __repr__(self):
        
        repr_str = (f'EstimationPotential({self.model_builder}, {self.exp_data}, {self.times}, nfe={self.nfe}, ncp={self.ncp}, bound_approach={self.bound_approach}, rho={self.rho}, epsilon={self.epsilon}, eta={self.eta}, max_iter_limit={self.max_iter_limit}, verbose={self.verbose}, debug={self.debug}, kwargs={self.kwargs})')
        
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
  
            for k, v in self.model.K.items():
                self.rh_model.K[k] = self.model.K[k] * self.model.P[k].value
                self.rh_model.P[k].set_value(1)
          
            reduced_hessian = self._calculate_reduced_hessian(Se, Sf, verbose=False)
            
            # Step 7 - Check the ratios of the parameter std to value
            if self.verbose:
                print(step.substitute(number=7))
                print('Checking the ratios of each parameter in Se')
            
            ratios, eigvals = self._parameter_ratios(reduced_hessian, Se)
            ratios_satisfied = max(ratios) < self.eta
        
            if ratios_satisfied:
                if self.verbose:
                    print('Step 7 passed, all paramater ratios are less than provided tolerance {self.eta}, moving to Step 10')
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
                                # model.P.display()

                        for k, v in self.model.K.items():
                            self.rh_model.K[k] = self.model.K[k] * self.model.P[k].value
                            self.rh_model.P[k].set_value(1)
                
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
        exp_data = list(self.model.measured_data.keys())
        
        if len(self.model.mixture_components.value) > 0:
            dfz = self._get_simulated_data(state='Z')       
            dfc = None
        
            if len(self.model.mixture_components.value & self.model.measured_data.value) > 0:        
                dfc = self._get_simulated_data(state='C')
            
                for col in dfc.columns:
                    if col not in exp_data:
                        dfc.drop(columns=[col], inplace=True)
            
            plt.figure()
            
            for col in dfz.columns:
                plt.plot(dfz[col], label=col + ' (pred)')
                if dfc is not None:
                    if col in dfc.columns:
                        plt.plot(dfc[col], 'o', label=col + ' (exp)')
                    
                plt.xlabel("time")
                plt.ylabel("Concentration (mol/L)")
                plt.title("Concentration Profile")
                
                plt.legend()
                plt.show()
            
        if len(self.model.complementary_states.value) > 0:
            dfx = self._get_simulated_data(state='X')
            dfu = None
            
            if len(self.model.complementary_states.value & self.model.measured_data.value) > 0:
                dfu = self._get_simulated_data(state='U')  
            
                for col in dfu.columns:
                    if col not in exp_data:
                        dfu.drop(columns=[col], inplace=True)
            
            plt.figure()
            
            for col in dfx.columns:
                plt.plot(dfx[col], label=col + ' (pred)')
                if dfu is not None:
                    if col in dfu.columns:
                        plt.plot(dfu[col], 'o', label=col + ' (exp)')
            
            plt.xlabel("time")
            plt.ylabel("State Values")       
            plt.title("Complementary State Profiles")
        
            plt.legend()
            plt.show()
    
        return None
    
    def _model_preparation(self):
        """Helper function that should prepare the models when called from the
        main function. Includes the experimental data, sets the objectives,
        simulates to warm start the models if no data is provided, sets up the
        reduced hessian model with "fake data", and discretizes all models

        """
        # Setup the model for parameter fitting
        self.exp_data = pd.DataFrame(self.exp_data)
        
        model_builder = copy.copy(self.model_builder)
        rh_model_builder = copy.copy(self.rh_model_builder)
     
        conc_state_headers = self.model_builder._component_names & set(self.exp_data.columns)
        if len(conc_state_headers) > 0:
            self.model_builder.add_concentration_data(pd.DataFrame(self.exp_data[conc_state_headers].dropna()))
        
        comp_state_headers = self.model_builder._complementary_states & set(self.exp_data.columns)
        if len(comp_state_headers) > 0:
            self.model_builder.add_complementary_states_data(pd.DataFrame(self.exp_data[comp_state_headers].dropna()))
     
        self.model = self.model_builder.create_pyomo_model(self.start_time, self.end_time)
        sim_model = copy.deepcopy(self.model)
        self._prep_model_for_optimization(self.model)
        self.model.objective = self._rule_objective(self.model, self.model_builder)
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
        #print(f'self.simulation_data:\n{simulation_data}')
    
        model_pe = ParameterEstimator(self.model)
        model_pe.apply_discretization('dae.collocation',
                                            ncp = self.ncp,
                                            nfe = self.nfe,
                                            scheme = 'LAGRANGE-RADAU')
        
        # Take the sim data and acutally make it work here
        
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
        
        # Setup the model for calculating the reduced hessian
        t = np.linspace(self.start_time, self.end_time, self.nfe + 1)
        t = [round(i, 6) for i in t]
        # At the moment, this uses fake data for the indexing - something cleaner?
        fake_data = pd.DataFrame(t, columns=['indx'])
        fake_data.drop(index=[0], inplace=True)
        fake_data.index=fake_data.indx
        fake_data.drop(columns=['indx'], inplace=True)
        
        for state in self.exp_data.columns:
            fake_data[state] = 1
        
        if len(conc_state_headers) > 0:
            self.rh_model_builder.add_concentration_data(pd.DataFrame(fake_data[conc_state_headers]))
            
        if len(comp_state_headers) > 0:
            self.rh_model_builder.add_complementary_states_data(pd.DataFrame(fake_data[comp_state_headers]))
    
        self.rh_model = self.rh_model_builder.create_pyomo_model(self.start_time, self.end_time) 
        self._run_simulation()
     
        return None
    
    def _run_simulation(self):
        """You need to run the simulation of the model to get data equal to 
        the current parameter set. The opt will be zero, but the reduced
        hessian should be correct.
        
        TODO:
            I think this needs to be changed - there must be an easier way
            (Abstract model?) than by generated a new Concrete model each time
            the reduced hessian needs to be calculated.
        
        Args:
            None
                
        Returns:
            None
            
        """
        print('*'*20 + ' Running Simulation ' + '*'*20)
        
        rh_model_builder = copy.copy(self.rh_model_builder)
        rh_model = rh_model_builder.create_pyomo_model(self.start_time, self.end_time)
        
        rh_sim = PyomoSimulator(rh_model)
            
        for k, v in rh_sim.model.P.items():
            rh_sim.model.P[k].fix(1)
            rh_sim.model.K[k] = self.rh_model.K[k]

        rh_sim.apply_discretization('dae.collocation',
                                           ncp = self.ncp,
                                           nfe = self.nfe,
                                           scheme = 'LAGRANGE-RADAU')
            
        results_pyomo = rh_sim.run_sim('ipopt_sens',
                                          tee=False,
                                          solver_options={},
                                          )
                
        self.rh_model = rh_model
        t = np.linspace(self.start_time, self.end_time, self.nfe + 1)
        t = [round(i, 6) for i in t]
        
        all_data = pd.DataFrame(index=t[1:])
        
        if len(self.model_builder._component_names) > 0:
            Z_data_full = pd.DataFrame(results_pyomo.Z)
            Z_data_fe = Z_data_full.loc[t, :]
            Z_data_fe.drop(index=0, inplace=True)
            all_data = all_data.merge(Z_data_fe, right_index=True, left_index=True)
        
        if len(self.model_builder._complementary_states) > 0:
            X_data_full = pd.DataFrame(results_pyomo.X)
            X_data_fe = X_data_full.loc[t, :]
            X_data_fe.drop(index=0, inplace=True)
            all_data = all_data.merge(X_data_fe, right_index=True, left_index=True)
        
        for data_set in self.model.measured_data.value:
            if data_set in self.model_builder._complementary_states:
                for fe in all_data.index:
                    
                    self.rh_model.U[(fe, data_set)].set_value(float(all_data.loc[fe, data_set]))
                    self.rh_model.U[(fe, data_set)].fix()
            
            if data_set in self.model_builder._component_names:
                for fe in all_data.index:
                    self.rh_model.C[(fe, data_set)].set_value(float(all_data.loc[fe, data_set]))
                    self.rh_model.C[(fe, data_set)].fix()
    
        self.rh_model.objective = self._rule_objective(self.rh_model, self.rh_model_builder)
        self._prep_model_for_optimization(self.rh_model)
    
        return None
        
    def _rule_objective(self, model, builder):
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

        # This can be cleaned up
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
        model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
        model.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
        model.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
        model.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
        model.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)
        model.red_hessian = Suffix(direction=Suffix.EXPORT)
        model.dof_v = Suffix(direction=Suffix.EXPORT)
        model.rh_name = Suffix(direction=Suffix.IMPORT)
        
        count_vars = 1
        for k, v in model.P.items():
            model.dof_v[k] = 1
            count_vars += 1
        
        model.npdp = Suffix(direction=Suffix.EXPORT)
        
        return None
        
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
        kaug = SolverFactory('k_aug')
        tmpfile_i = "ipopt_output"
        
        self._run_simulation()
        
        for k, v in self.rh_model.P.items():
            ub = self.rh_model.P[k].value
            lb = self.rh_model.P[k].value - delta
            self.rh_model.P[k].setlb(lb)
            self.rh_model.P[k].setub(ub)
            self.rh_model.P[k].unfix()
        
        for fixed_param in Sf:
            self.rh_model.P[fixed_param].fix(1)
        
        self.rh_model.ipopt_zL_in.update(self.rh_model.ipopt_zL_out)
        self.rh_model.ipopt_zU_in.update(self.rh_model.ipopt_zU_out)
        
        ipopt.solve(self.rh_model, symbolic_solver_labels=True, keepfiles=True, tee=verbose, logfile=tmpfile_i)

        with open(tmpfile_i, 'r') as f:
            output_string = f.read()
        
        stub = output_string.split('\n')[0].split(',')[1][2:-4]
        col_file = Path(stub + '.col')
        
        kaug.options["deb_kkt"] = ""  
        kaug.solve(self.rh_model, tee=verbose)
        
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
        col_ind = [var_ind.loc[var_ind[0] == f'P[{v}]'].index[0] for v in Se]
        
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
    
    def _get_simulated_data(self, state='C'):
    
        val = []
        ix = []
        varobject = getattr(self.model, state)
        for index in varobject:
            ix.append(index)
            val.append(varobject[index].value)
        
        a = pd.Series(index=ix, data=val)
        dfs = pd.DataFrame(a)
        index = pd.MultiIndex.from_tuples(dfs.index)
       
        dfs = dfs.reindex(index)
        dfs = dfs.unstack()
        dfs.columns = [v[1] for v in dfs.columns]
    
        return dfs
        
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