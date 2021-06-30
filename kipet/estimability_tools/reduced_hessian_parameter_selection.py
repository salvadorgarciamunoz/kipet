"""
This module implements the reduced Hessian parameter selection method outlined
in Chen and Biegler (AIChE 2020).

"""
# Standard library imports
import copy
from string import Template

# Third party imports
import numpy as np
from pyomo.environ import (Objective, SolverFactory,
                           Suffix)

# KIPET library imports
from kipet.model_components.objectives import comp_objective, conc_objective
from kipet.estimability_tools.parameter_handling import (check_initial_parameter_values,
                                                       set_scaled_parameter_bounds)
from kipet.estimability_tools.parameter_ranking import parameter_ratios, rank_parameters
from kipet.estimator_tools.parameter_estimator import ParameterEstimator
from kipet.estimator_tools.results_object import ResultsObject
from kipet.model_tools.scaling import (scale_parameters,
                                            update_expression)

__author__ = 'Kevin McBride'  #: April 2020
    
class EstimationPotential():
    """This class is for estimability analysis. The algorithm here is the one
    presented by Chen and Biegler (accepted AIChE 2020) using the reduced 
    hessian to select the estimable parameters. 

    :Attributes:
    
        - model_builder (pyomo ConcreteModel): The pyomo model
    
        - simulation_data (pandas.DataFrame): Optional simulation data to use for
            warm starting (Needs testing!)
        
        - options (dict): Various options for the esimability algorithm:
        
            - nfe (int): The number of finite elements to use in the collocation.
            
            - ncp (int): The number of collocation points per finite element.
            
            - bound_approach (float): The accepted relative difference for
                determining whether or not a bound is considered active (Step 6).
                
            - rho (float): Factor used to determine the lower and upper bounds for
                each parameter in fitting (Step 5).
                
            - epsilon (float): The minimum value for parameter values
            
            - eta (float): Predetermined cut-off value for accepted std/parameter
                ratios.
                
            - max_iter_limit (int): Iteration limits for the estimability algorithm.
            
            - verbose (bool): Defaults to False, option to display the progress of
                the algorihm.
                
            - debug (bool): Defaults to False, option to ask for user input to
                proceed during the algorithm.
                
            - simulate_start (bool): Option to simulate using the model to
                warm start the optimization
        
    """

    def __init__(self, model, simulation_data=None, options=None,
                 method='k_aug', solver_opts={}, scaled=True,
                 use_bounds=False, use_duals=False, calc_method='fixed'):

        """Initialization of the EstimationPotential class

        :param ConcreteModel model: The Pyomo model to perform analysis on
        :param pandas.DataFrame simulation_data: Initialization data for the model
        :param dict options: Dictionary of options to use in the RHPS method
        :param str method: The method to use in calculating sensitivities
        :param dict solver_opts: Options for the solver, if any
        :param bool scaled: Optional scaling
        :param bool use_bounds: Optional use of parameter bounds (dev opt)
        :param bool use_duals: Use the dual option in sensitiivty calculations (dev opt)
        :param str calc_method: Method to calculate the reduced Hessian (dev opt)
        """
        # Options handling
        self.options = {} if options is None else options.copy()
        self._options = options.copy()
        
        print(self._options)
        
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
        self.method = method
        self.solver_opts = solver_opts
        self.scaled = scaled
        self.use_bounds = use_bounds
        self.use_duals = use_duals
        self.rh_method = calc_method
        
        # Copy the model
        self.model = copy.deepcopy(model)
        self.simulation_data = simulation_data
        
        self.orig_bounds = None
        if not self.scaled and self.use_bounds:
            self.orig_bounds = {k: (v.lb, v.ub) for k, v in self.model.P.items()}
        
        self.debug = False
        self.verbose = True
        
    def __repr__(self):
        
        repr_str = (f'EstimationPotential({self.model},'
                    f' simulation_data={"Provided" if self.simulation_data is not None else "None"},'
                    f' options={self.options})')
        
        return repr_str
        
    def estimate(self):
        """This performs the estimability analysis based on the method
        developed in Chen and Biegler 2020 AIChE...

        :return: None

        """
        bound_check = True     

        flag = False
        step = Template('\n' + '*' * 20 + ' Step $number ' + '*' * 20)
        
        self._model_preparation()
        
        # Step 1
        if self.verbose:
            print(step.substitute(number=1))
            print('Initializing N_pre and N_curr\n')
        
        N_pre = len(self.model.P)
        N_curr = len(self.model.P)
        
        # Step 2
        if self.verbose:
            print(step.substitute(number=2))
            print('Initialize Se and Sf\n')
        
        Se = [parameter for parameter in self.model.P.keys()]
        Sf = []
        
        if self.verbose:
            print(f'\nSe: {Se}')
            print(f'Sf: {Sf}\n')
        
        # Step 3 - Calculate the reduced hessian for the initial stage
        if self.verbose:
            print(step.substitute(number=3))
            print('Calculating the Reduced Hessian for the initial parameter set\n')
        
        reduced_hessian = self._calculate_reduced_hessian(Se)
        
        print(reduced_hessian)
        if self.debug:
            
            input("Press Enter to continue...")
    
        # Step 4 - Rank the parameters using Gauss-Jordan Elimination
        if self.verbose:
            print(step.substitute(number=4))
            print('Ranking parameters for estimability and moving to Step 5')
        
        Se, Sf = rank_parameters(self.model, reduced_hessian, Se, epsilon=self.epsilon, eta=self.eta)
        
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
        params_counter = 0
        saved_parameters_K = {}
        
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
                    if self.scaled:
                        self.model.P[fixed_param].fix(1) # changed from (1) to ()
                    else:
                        self.model.P[fixed_param].fix() # changed from (1) to ()
                        
                ipopt = SolverFactory('ipopt')
                ipopt.solve(self.model, tee=False)#self.verbose)
                
                if self.verbose:
                    self.model.P.display()
                    
                # Step 6 - Check for active bounds
                number_of_active_bounds = 0
                
                if self.verbose:
                    print(step.substitute(number=6))
                    print('Checking for active bounds\n')
                else:
                    None
                
                if bound_check:
                    for key, param in self.model.P.items():
                        if (param.value-param.lb)/param.lb <= self.bound_approach or (param.ub - param.value)/param.value <= self.bound_approach:
                            number_of_active_bounds += 1
                            if self.verbose:
                                print('There is at least one active bound - updating parameters and optimizing again\n')
                            break
                
                else:
                    None
                
                if self.scaled and hasattr(self.model, 'K'):
                    for k, v in self.model.K.items():
                        self.model.K[k] = self.model.K[k] * self.model.P[k].value
                        self.model.P[k].set_value(1)
                        
                    # print(self.model.K.display())
                    # print(self.model.P.display())
                        
                else:
                    set_scaled_parameter_bounds(self.model,
                                                parameter_set=Se,
                                                rho=self.rho,
                                                scaled=self.scaled,
                                                original_bounds=self.orig_bounds)
                    
                if hasattr(self.model, 'K'):
                    param_val_save = 'K'
                else:
                    param_val_save = 'P'
                    
                saved_parameters_K[params_counter] = {k: v.value for k, v in getattr(self.model, param_val_save).items()}
                params_counter += 1
                
                if bound_check:
                    if number_of_active_bounds == 0:
                        if self.verbose:
                            print('There are no active bounds, moving to Step 7')
                        break
                        if self.debug:    
                            input("Press Enter to continue...")
                else:
                    None
                        
                inner_iteration_counter += 1
                if self.debug:
                    input("Press Enter to continue...")
                
            if self.debug:
                self.model.P.display()
                self.model.K.display()
                
            reduced_hessian = self._calculate_reduced_hessian(Se)
            
            # Step 7 - Check the ratios of the parameter std to value
            if self.verbose:
                print(reduced_hessian)
                print(step.substitute(number=7))
                print('Checking the ratios of each parameter in Se')
            
            ratios, eigvals = parameter_ratios(self.model, reduced_hessian, Se, epsilon=self.epsilon)
            
            if self.verbose:
                 print('Ratios:')
                 print(ratios)
            
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
                    
                    if self.scaled and hasattr(self.model, 'K'):
                        for k, v in self.model.K.items():
                            self.model.K[k] = saved_parameters_K[params_counter-2][k]
                            self.model.P[k].set_value(1)
                    else:
                        for k, v in self.model.P.items():
                            self.model.P[k].set_value(saved_parameters_K[params_counter-1][k])
                        
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
                    if sum(1.0/eigvals) < sum((np.array([self.model.P[v].value for v in Se]))**2)*(self.eta**2) and flag == True:
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
                        Se = [parameter for parameter in self.model.P.keys()]
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
                  
                        reduced_hessian = self._calculate_reduced_hessian(Se)
                        
                        if self.debug:
                            print(reduced_hessian)
                            input("Press Enter to continue...")
                        if self.verbose:
                            # Step 4 - Rank the updated parameters using Gauss-Jordan elimination
                            print(step.substitute(number=4))
                            print('Ranking the parameters (limited by N_curr)')
                        
                        Se, Sf = rank_parameters(self.model, reduced_hessian, Se, epsilon=self.epsilon, eta=self.eta)
                        
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
        
        self.model.K_vals = saved_parameters_K
        
        print(f'The estimable parameters are: {", ".join(Se)}')
        print('\nThe final parameter values are:\n')
        if hasattr(self.model, 'K') and self.model.K is not None:
            self.model.K.pprint()
        else:
            self.model.P.pprint()
            
        results = self._get_results(Se)
            
        return results, self.model
    
    def _get_results(self, Se):
        """Arranges the results into a ResultsObject

        :param list Se: The list of estimable parameters

        :return: The results from the parameter estimation process
        :rtype: ResultsObject

        """
        scaled_parameter_var = 'K'
        results = ResultsObject()
        results.estimable_parameters = Se
        results.load_from_pyomo_model(self.model)

        if hasattr(self.model, scaled_parameter_var): 
            results.P = {name: self.model.P[name].value*getattr(self.model, scaled_parameter_var)[name].value for name in self.model.parameter_names}
        else:
            results.P = {name: self.model.P[name].value for name in self.model.parameter_names}

        return results
    
    def _model_preparation(self):
        """Helper function that should prepare the models when called from the
        main function. Includes the experimental data, sets the objectives,
        simulates to warm start the models if no data is provided, sets up the
        reduced hessian model with "fake data", and discretizes all models

        :return: None

        """
        if not hasattr(self.model, 'objective'):
            self.model.objective = self._rule_objective(self.model)
        
        # The model needs to be discretized
        model_pe = ParameterEstimator(self.model)
        model_pe.apply_discretization('dae.collocation',
                                      ncp=self.ncp,
                                      nfe=self.nfe,
                                      scheme='LAGRANGE-RADAU')
        
        # Here is where the parameters are scaled
        if self.scaled:
            scale_parameters(self.model)
            set_scaled_parameter_bounds(self.model, rho=self.rho)
        
        else:
            check_initial_parameter_values(self.model)
        
        if self.use_duals:
            self.model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
        
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
        
        :param ConcreteModel model: This is the pyomo model instance for the estimability problem.
                
        :return: This returns the objective function
        :rtype: expression

        """
        obj = 0
        obj += 0.5*conc_objective(model) 
        obj += 0.5*comp_objective(model)
    
        return Objective(expr=obj)

    def _calculate_reduced_hessian(self, Se):
        """This function solves an optimization with very restrictive bounds
        on the paramters in order to get the reduced hessian at fixed 
        conditions.

        :param list Se: The current list of estimable parameters.

        :return numpy.ndarray reduced_hessian: The reduced Hessian

        """
        rh_model = copy.deepcopy(self.model)
        
        from kipet.estimator_tools.reduced_hessian_methods import calculate_reduced_hessian
        
        reduced_hessian = calculate_reduced_hessian(
            rh_model, 
            optimize=True, 
            parameter_set=Se, 
            fix_method=self.rh_method,
        )
        return reduced_hessian


def rhps_method(model, **kwargs):
    """Reduces a single model using the reduced hessian parameter selection
    method. It takes a pyomo ConcreteModel using P as the parameters to be fit
    and K as the scaled parameter values.

    :param ConcreteModel model: The full model to be reduced
    :param dict kwargs: The keyword args passed to EstimaitonPotential

    :return results: The results from the parameter selection and optimization
    :return reduced_model: returns the reduced model with full parameter set
    :rtype: tuple(ResultsObject, ConcreteModel)

    """
    simulation_data = kwargs.get('simulation_data', None)
    replace = kwargs.get('replace', True)
    no_scaling = kwargs.get('no_scaling', True)
    method = kwargs.get('method', 'k_aug')
    solver_opts = kwargs.get('solver_opts', {})
    scaled = kwargs.get('scaled', True)
    use_bounds = kwargs.get('use_bounds', False)
    use_duals = kwargs.get('use_duals', False)
    calc_method = kwargs.get('calc_method', 'fixed')
    
    options = kwargs#options if options is not None else dict()
    orig_bounds = {k: v.bounds for k, v in model.P.items()}
    est_param = EstimationPotential(model,
                                    simulation_data=None,
                                    options=options,
                                    method=method,
                                    solver_opts=solver_opts,
                                    scaled=scaled,
                                    use_bounds=use_bounds,
                                    use_duals=use_duals,
                                    calc_method=calc_method)
    results, reduced_model = est_param.estimate()
    
    if replace:
        reduced_model = replace_non_estimable_parameters(reduced_model, 
                                                         results.estimable_parameters)
    # if no_scaling:
    #     remove_scaling(reduced_model, bounds=orig_bounds)
        
    return results, reduced_model


def replace_non_estimable_parameters(model, set_of_est_params):
    """Takes a model and a set of estimable parameters and removes the 
    unestimable parameters from the model by fixing them to their current 
    values in the model
    
    :param ConcreteModel model: The full model to be reduced
    :param set set_of_est_params: Parameters found to be estimable

    :return model: The model with parameters replaced
    :rtype: ConcreteModel

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
                ep_updated_expr = update_expression(v.body, model.P[param], change_value)
                if hasattr(model, 'K'):
                    ep_updated_expr = update_expression(ep_updated_expr, model.K[param], 1)
                model.odes[k] = ep_updated_expr == 0
    
            model.parameter_names.remove(param)
            del model.P[param]
            if hasattr(model, 'K'):
                del model.K[param]

    return model
