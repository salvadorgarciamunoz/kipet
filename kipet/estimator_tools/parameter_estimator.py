"""
Primary class for performing the parameter fitting in KIPET
"""
# Standard library imports
import copy
import time

# Third party imports
import numpy as np
import pandas as pd
from pyomo.environ import (
    Constraint, 
    ConstraintList, 
    Objective,
    SolverFactory, 
    TerminationCondition, 
    value, 
    Var)

# KIPET library imports
from kipet.model_components.objectives import (
    comp_objective,
    conc_objective)
from kipet.estimator_tools.pyomo_simulator import PyomoSimulator
from kipet.estimator_tools.reduced_hessian_methods import (
    covariance_k_aug,
    covariance_sipopt, 
    define_free_parameters)
from kipet.estimator_tools.results_object import ResultsObject
from kipet.mixins.parameter_estimator_mixins import PEMixins
from kipet.model_tools.pyomo_model_tools import convert
from kipet.general_settings.variable_names import VariableNames


class ParameterEstimator(PEMixins, PyomoSimulator):

    """Optimizer for parameter estimation"""

    def __init__(self, model):
        super(ParameterEstimator, self).__init__(model)

        self.__var = VariableNames()

        self.hessian = None
        self.cov = None
        self._estimability = False
        self._idx_to_variable = dict()
        self.model_variance = True
        self.termination_condition = None
        
        # This should be a subclass or a mixin
        self.G_contribution = None
        self.unwanted_G = False
        self.time_variant_G = False
        self.time_invariant_G = False
        self.time_invariant_G_decompose = False
        self.time_invariant_G_no_decompose = False
        
        # Parameter name list
        self.param_names_full = define_free_parameters(self.model, kind='full')
        self.param_names = define_free_parameters(self.model, kind='simple')

        # for new huplc structure (CS):
        if self._huplc_given:
            self._list_huplcabs = self._huplc_absorbing
            # self._n_huplc = len(self._list_huplcabs)

        self.n_val = self._n_components
        
        self.cov = None
            
    def run_opt(self, solver, **kwds):

        """ Solves parameter estimation problem.

        :param str solver: The solver used to solve the NLP
        :param dict kwds: The dictionary of options passed from the ReactionModel

        :Keyword Args:
            - solver (str): name of the nonlinear solver to used
            - solver_opts (dict, optional): options passed to the nonlinear solver
            - variances (dict or float, optional): map of component name to noise variance. The
              map also contains the device noise variance. If not float then we only use device variance
              and ignore model variance.
            - tee (bool,optional): flag to tell the optimizer whether to stream output
              to the terminal or not.
            - with_d_vars (bool,optional): flag to the optimizer whether to add
            - variables and constraints for D_bar(i,j).
            - report_time (bool, optional): flag as to whether to time the parameter estimation or not.
            - estimability (bool, optional): flag to tell the model whether it is
              being used by the estimability analysis and therefore will need to return the
              hessian for analysis.
            - model_variance (bool, optional): Default is True. Flag to tell whether we are only
              considering the variance in the device, or also model noise as well.
            - model_variance (bool, optional): Default is True. Flag to tell whether we are only
              considering the variance in the device, or also model noise as well.

        :return: The results from the parameter fitting
        :rtype: ResultsObject

        """
        run_opt_kwargs = copy.copy(kwds)
        
        solver_opts = kwds.pop('solver_opts', dict())
        variances = kwds.pop('variances', dict())
        tee = kwds.pop('tee', False)
        with_d_vars = kwds.pop('with_d_vars', False)
        covariance = kwds.pop('covariance', None)
 
        estimability = kwds.pop('estimability', False)
        report_time = kwds.pop('report_time', False)
        model_variance = kwds.pop('model_variance', True)

        inputs_sub = kwds.pop("inputs_sub", None)
        trajectories = kwds.pop("trajectories", None)
        fixedtraj = kwds.pop("fixedtraj", False)
        fixedy = kwds.pop("fixedy", False)
        yfix = kwds.pop("yfix", None)
        yfixtraj = kwds.pop("yfixtraj", None)

        jump = kwds.pop("jump", False)
        
        self.confidence = kwds.pop('confidence', None)
        if self.confidence is None:
            self.confidence = 0.95
        
        G_contribution = kwds.pop('G_contribution', None)
        St = kwds.pop('St', dict())
        Z_in = kwds.pop('Z_in', dict())

        self.solver = solver
        self.covariance_method = covariance
        self.model_variance = model_variance
        self._estimability = estimability

        if not self.model.alltime.get_discretization_info():
            raise RuntimeError('apply discretization first before initializing')

        self.G_contribution = G_contribution
       
        if report_time:
            start = time.time()
        
        if self.G_contribution == 'time_invariant_G':
            self.decompose_G_test(St, Z_in)
        self.g_handling_status_messages()

        # Move?
        if self.covariance_method == 'ipopt_sens':
            if not 'compute_red_hessian' in solver_opts.keys():
                solver_opts['compute_red_hessian'] = 'yes'
                
        self.solver_opts = solver_opts
            
        if inputs_sub is not None:
            from kipet.estimator_tools.additional_inputs import add_inputs
            
            add_kwargs = dict(
                fixedtraj = fixedtraj,
                fixedy = fixedy, 
                inputs_sub = inputs_sub,
                yfix = yfix,
                yfixtraj = yfixtraj,
                trajectories = trajectories,
            )
            add_inputs(self, add_kwargs)
        
        if jump:
            from kipet.estimator_tools.jumps_method import set_up_jumps
            set_up_jumps(self.model, run_opt_kwargs)
            
        # # I am not sure if this is still needed...
        # active_objectives = [o for o in self.model.component_map(Objective, active=True)]        
        # if active_objectives:
        #     print(
        #         "WARNING: The model has an active objective. Running optimization with models objective.\n"
        #         " To solve optimization with default objective (Weifengs) deactivate all objectives in the model.")
        #     solver_results = opt.solve(self.model, tee=tee)

        #if self._spectra_given:
        self.objective_value = self._solve_model(
            variances,
            tee=tee,
            covariance=covariance,
            with_d_vars=with_d_vars,
            **kwds)

        if report_time:
            end = time.time()
            print("Total execution time in seconds for variance estimation:", end - start)

        return self._get_results()

    def _get_results(self):
        """Removed results unit from function

        :return: The formatted results
        :rtype: ResultsObject

        """
        results = ResultsObject()
        results.objective = self.objective_value
        results.parameter_covariance = self.cov
        results.load_from_pyomo_model(self.model)
        results.show_parameters(self.confidence)

        if self._spectra_given:
            from kipet.calculation_tools.beer_lambert import D_from_SC
            D_from_SC(self.model, results)

        if hasattr(self.model, self.__var.model_parameter_scaled): 
            setattr(results, self.__var.model_parameter, {name: getattr(self.model, self.__var.model_parameter)[name].value*getattr(self.model, self.__var.model_parameter_scaled)[name].value for name in self.model.parameter_names})
        else:
            setattr(results, self.__var.model_parameter, {name: getattr(self.model, self.__var.model_parameter)[name].value for name in self.model.parameter_names})

        if self.termination_condition!=None and self.termination_condition!=TerminationCondition.optimal:
            raise Exception("The current iteration was unsuccessful.")
        else:
            if self._estimability == True:
                return self.hessian, results
            else:
                return results

        return results
            
    def _solve_model(self, sigma_sq, **kwds):
        """Main function to getup the objective and perform the parameter estimation 

           This method is not intended to be used by users directly

        :param dict sigma_sq: variances
        :param SolverFactory optimizer: Pyomo Solver factory object

        :Keyword Args:

            - with_d_vars (bool,optional): flag to the optimizer whether to add
              variables and constraints for D_bar(i,j)

        :return: None

        """
        with_d_vars = kwds.pop('with_d_vars', False)
     
        # These are not being used
        penaltyparam = kwds.pop('penaltyparam', False)
        penaltyparamcon = kwds.pop('penaltyparamcon', False) #added for optional penalty term related to constraint CS
        ppenalty_dict = kwds.pop('ppenalty_dict', None)
        ppenalty_weights = kwds.pop('ppenalty_weights', None)
        
        model = self.model
        model.objective = Objective(expr=0)
        
        if self._spectra_given:

            all_sigma_specified = True
    
            if isinstance(sigma_sq, dict): 
                keys = sigma_sq.keys()
                for k in self.comps['unknown_absorbance']:
                    if k not in keys:
                        all_sigma_specified = False
                        sigma_sq[k] = max(sigma_sq.values())
    
                if not 'device' in sigma_sq.keys():
                    all_sigma_specified = False
                    sigma_sq['device'] = 1.0
    
            elif isinstance(sigma_sq, float):
                sigma_dev = sigma_sq
                sigma_sq = dict()
                sigma_sq['device'] = sigma_dev
            
            if self.time_invariant_G_no_decompose:
                for i in model.times_spectral:
                    model.qr[i] = 1.0
                model.qr.fix()
    
            def _qr_end_constraint(model):
                return model.qr[model.times_spectral[-1]] == 1.0
            
            if self.G_contribution == 'time_variant_G':
                model.qr_end_cons = Constraint(rule = _qr_end_constraint)
            
            
            if with_d_vars and self.model_variance:
                model.D_bar = Var(model.times_spectral, model.meas_lambdas)
                
                def rule_D_bar(model, t, l):
                    if hasattr(model, 'huplc_absorbing') and hasattr(model, 'solid_spec_arg1'):
                        return model.D_bar[t, l] == sum(
                            getattr(model, 'C')[t, k] * model.S[l, k] for k in self.comps['unknown_absorbance'] if k not in model.solid_spec_arg1)
                    else:
                        return model.D_bar[t, l] == sum(getattr(model, 'C')[t, k] * model.S[l, k] for k in self.comps['unknown_absorbance'])
    
                model.D_bar_constraint = Constraint(model.times_spectral,
                                                    model.meas_lambdas,
                                                    rule=rule_D_bar)
    
            if self.model_variance:
            
                # D_bar - Beer Lambert's Law
                self._primary_D_term(model, with_d_vars, self.comps['unknown_absorbance'], sigma_sq)
                # Model fit
                self._concentration_spectra_term(model, sigma_sq)     
                # L2 penalty
                self._l2_term(model, penaltyparam, ppenalty_dict, ppenalty_weights)
                # HUPLC
                if hasattr(model, 'huplc_absorbing'):
                    self._huplc_obj_term(model, sigma_sq)
                    self._penalty_term(model, penaltyparamcon, rho=1e-1)
                    
            else: # device only
                if hasattr(model, 'huplc_absorbing'):
                    component_set = self.comps['absorbing']
                    self._huplc_obj_term(model, sigma_sq)
                else:
                    component_set = self.comps['unknown_absorbance']
                
                self._primary_D_term(model, with_d_vars, component_set, sigma_sq, device=True)
         
        elif self._concentration_given: 
            
            self._concentration_term(model, sigma_sq, 'concentration')
            self._state_term(model)
            self._custom_term(model)
            self._penalty_term(model, penaltyparamcon)
         
        # # Break up the objective
        # custom_weight = 1
        # if hasattr(model, 'custom_obj'):
        #     model.objective.expr += custom_weight*(model.custom_obj)

        # HUPLC shows up here too - if needed in a future version, this can be reimplemented
        # for k in self.comps['all']:
        #     if hasattr(self, 'huplc_absorbing') and k not in keys:
        #         for ki in self.solid_spec_args1:
        #             for key in keys:
        #                 if key == ki:
        #                     sigma_sq[ki] = sigma_sq[key]

        #     elif hasattr(self, 'huplc_absorbing') == False and k not in keys:
        #         all_sigma_specified = False
        #         sigma_sq[k] = max(sigma_sq.values())

        # if self._huplc_given:  # added for new huplc structure CS
        #     weights = kwds.pop('weights', [1.0, 1.0, 1.0])
        # else:
        #     weights = kwds.pop('weights', [1.0, 1.0])

        obj_val = self.optimize(model, sigma_sq)
        
        return obj_val

    @staticmethod
    def _concentration_term(model, sigma_sq, source):
        """Sets up the LSE terms for the concentration in the objective
        
        :param ConcreteModel model: The current model
        :param dict sigma_sq: The component variances
        :param str source: concentration or specta
        
        :return: None
        
        """

        obj=0
        obj += conc_objective(model, variance=sigma_sq, source=source)  
        model.objective.expr += obj
    
        return None
    
    @staticmethod
    def _state_term(model):
        """Sets up the LSE terms for the states in the objective
        
        :param ConcreteModel model: The current model
        
        :return: None
        
        """
        obj = comp_objective(model)  
        model.objective.expr += obj
    
    @staticmethod
    def _custom_term(model):
        """Sets up the LSE terms for the custom objective
        
        :param ConcreteModel model: The current model
        
        :return: None
        
        """
        if hasattr(model, 'custom_obj'):
            model.objective.expr += model.custom_obj
    
        return None
    
    def _concentration_spectra_term(self, model, sigma_sq):
        """Sets up the LSE terms for the concentration (from spectra) in the objective
        
        :param ConcreteModel model: The current model
        :param dict sigma_sq: The component variances

        :return: None
        
        """
        obj = 0
        for t in model.times_spectral:
            obj += sum((model.C[t, k] - model.Z[t, k]) ** 2 / sigma_sq[k] for k in self.comps['unknown_absorbance'])
            
        model.objective.expr += obj
        
        return None
    
    @staticmethod
    def _penalty_term(model, penaltyparamcon, rho=100):
        """Sets up the penalty terms in the objective
        
        :param ConcreteModel model: The current model
        :param bool penaltyparacon: option whether penalties are used
        :param float rho: weighting for the penalties
        
        :return: None
        
        """
        if penaltyparamcon == True:
            sum_penalty = 0.0
            for t in model.allmeas_times:
                sum_penalty += model.Y[t, 'npen']

            model.objective.expr += rho * sum_penalty
            
        return None
    
    @staticmethod
    def _l2_term(model, penaltyparam, ppenalty_dict, ppenalty_weights):
        """Sets up the L2 penatly terms in the objective
        
        :param ConcreteModel model: The current model
        :param dict penatlyparam: option if penalty params are used
        :param dict ppenalty_dict: set-up info for the penalty parameters
        :param dict ppenalty_weights: weights associated with penalties
        
        :return: None
        
        """
        terms = 0
        # L2 penalty term to objective penalizing values that deviate from values defined in ppenalty_dict:
        if penaltyparam==True:
            if ppenalty_weights is None:
                terms = 0.0
                for k in model.P.keys():
                    if k in ppenalty_dict.keys():
                        terms += (model.P[k] - ppenalty_dict[k]) ** 2
            else:
                if len(ppenalty_dict)!=len(ppenalty_weights):
                    raise RuntimeError(
                        'For every penalty term a weight must be defined.')
                if ppenalty_dict.keys()!=ppenalty_weights.keys():
                    raise RuntimeError(
                        'Check the parameter names in ppenalty_weights and ppenalty_dict again. They must match.')
                else:
                    for k in model.P.keys():
                        if k in ppenalty_dict.keys():
                                terms += ppenalty_weights[k] * (model.P[k] - ppenalty_dict[k]) ** 2
                                
        model.objective.expr += terms
    
    def _primary_D_term(self, model, with_d_vars, component_set, sigma_sq, device=False):
        """Build the objective expression for D_bar (primary objective term in KIPET)

        :param ConcreteModel model: The model for use in parameter fitting
        :param bool with_d_vars: For spectral problems
        :param list component_set: A list of the components
        :param dict sigma_sq: A dictionary of component variances
        :param bool device: Indicates if the device variance is used

        :return expr: The objective expression
        :rtype: expression

        """
        expr = 0
        for t in model.times_spectral:
            for l in model.meas_lambdas:
                
                if with_d_vars:
                    D_bar = model.D_bar[t, l]
                else:
                    if device:    
                        if hasattr(model, 'huplc_absorbing'):
                            D_bar = sum(model.Z[t, k] * model.S[l, k] for k in component_set if k not in model.solid_spec_arg1)
                        else:
                            D_bar = sum(model.Z[t, k] * model.S[l, k] for k in component_set)    
                    else:
                        if hasattr(model, 'huplc_absorbing') and hasattr(model, 'solid_spec_arg1'):
                            D_bar = sum(model.C[t, k] * model.S[l, k] for k in component_set if k not in model.solid_spec_arg1)
                        else:
                            D_bar = sum(model.C[t, k] * model.S[l, k] for k in component_set)
                
                if self.G_contribution == 'time_variant_G':
                    expr += (model.D[t, l] - D_bar - model.qr[t]*model.g[l]) ** 2 / (sigma_sq['device'])
                elif self.time_invariant_G_no_decompose:
                    expr += (model.D[t, l] - D_bar - model.g[l]) ** 2 / (sigma_sq['device'])
                else:
                    expr += (model.D[t, l] - D_bar) ** 2 / (sigma_sq['device'])
                   
        model.objective.expr += expr
            
        return None
    
    
    def _regular_optimization(self, model):
        """Optimize without the covariance.
        
        This is used for setting up the solver factory with ipopt for models
        where the covariance is not needed. This is also used for the first
        step in solving models with k_aug.
        
        """
        optimizer = SolverFactory(self.solver)
        for key, val in self.solver_opts.items():
            optimizer.options[key] = val
        solver_results = optimizer.solve(model, tee=False, symbolic_solver_labels=True)
        self._termination_problems(solver_results, optimizer)
    
    
    def optimize(self, model, sigma_sq=None):
        """Handler for optimization using covariance or not"""
        
        if self.covariance_method in ['k_aug', 'ipopt_sens']:
            optimizer = SolverFactory(self.covariance_method)
            for key, val in self.solver_opts.items():
                optimizer.options[key] = val
            self.covariance(optimizer, sigma_sq) 
        else:
            self._regular_optimization(model)
            
        obj_val = model.objective.expr()
       
        model.del_component('objective')
        if hasattr(model, 'D_bar'):
            model.del_component('D_bar')
        if hasattr(model,' D_bar_constraint'):
            model.del_component('D_bar_constraint')
            
        return obj_val
    
    
    def covariance(self, optimizer=None, sigma_sq=None):
        """This consolidates the covariances methods into a convenient function
        
        :param SolverFactory optimizer: The solver (ipopt) used in k_aug
        :param dict sigma_sq: The variances used in covariances calculations
        
        :return: None
        
        """
        if self.covariance_method == 'ipopt_sens':
            self.inv_hessian, self.inv_hessian_reduced = covariance_sipopt(self.model, optimizer, self.comps['unknown_absorbance'], self.param_names_full)
            
        elif self.covariance_method == 'k_aug':
            self.inv_hessian, self.inv_hessian_reduced = covariance_k_aug(self.model, None, self.comps['unknown_absorbance'], self.param_names_full)
            
        if hasattr(self.model, 'C'):
            from kipet.estimator_tools.reduced_hessian_methods import compute_covariance
            models_dict = {'reaction_model': self.model}
            free_params = len(self.param_names)
            variances = {'reaction_model': sigma_sq}
            self.covariance_parameters = compute_covariance(models_dict, self.inv_hessian_reduced, free_params, variances)
        else:
            self.covariance_parameters = self.inv_hessian_reduced
            
        self.cov = pd.DataFrame(self.covariance_parameters, index=self.param_names, columns=self.param_names)  
    
        return None
    
    
    def _termination_problems(self, solver_results, optimizer):
        """Checks the termination conditions and will try once again if it fails

        :param ResultsObject solver_results: The results from the parameter fittings
        :param SolverFactory optimizer: The solver factory being used

        :return: None

        """
        self.termination_condition = solver_results.solver.termination_condition
        if self.termination_condition != TerminationCondition.optimal:
            print("WARNING: The solution of the iteration was unsuccessful. The problem is solved with additional solver options.")
            optimizer.options["OF_start_with_resto"] = 'yes'
            solver_results = optimizer.solve(self.model, tee=False, symbolic_solver_labels=True)
            self.termination_condition = solver_results.solver.termination_condition
            if self.termination_condition != TerminationCondition.optimal:
                print(
                    "WARNING: The solution of the iteration was unsuccessful. The problem is solved with additional solver options.")
                optimizer.options["OF_start_with_resto"] = 'no'
                # optimizer.options["OF_bound_push"] = 1E-02
                optimizer.options["OF_bound_relax_factor"] = 1E-05
                solver_results = optimizer.solve(self.model, tee=False, symbolic_solver_labels=True)
                self.termination_condition = solver_results.solver.termination_condition
                # options["OF_bound_relax_factor"] = 1E-08
                if self.termination_condition != TerminationCondition.optimal:
                        print("The current iteration was unsuccessful.")
                        
        return None
    
        
    def _huplc_obj_term(self, model, sigma_sq):
        """Adds the HUPLC term to the objective

        :param ConcreteModel m: The model used in parameter fitting
        :param dict sigma_sq: The dict of variances

        :return expressions third_term: The HUPLC objective expression

        """
        def rule_Dhat_bar(m, t, l):
            list_huplcabs = [k for k in m.huplc_absorbing.value]
            return m.Dhat_bar[t, l] == (m.Z[t, l] + m.solidvol[t, l]) / (
                sum(m.Z[t, j] + m.solidvol[t, j] for j in list_huplcabs))

        third_term = 0.0
        
        # Arrange the HUPLC device variance
        if not 'device-huplc' in sigma_sq.keys():
            sigma_sq['device-huplc'] = 1.0
            
        if hasattr(model, 'solid_spec_arg1') and hasattr(model, 'solid_spec_arg2'):
            solidvol_dict = dict()
            model.add_component('cons_solidvol', ConstraintList())
            new_consolidvol = getattr(model, 'cons_solidvol')
            model.add_component('solidvol', Var(model.huplctime, model.huplc_absorbing.value, initialize=solidvol_dict))

            for k in model.solid_spec_arg1:
                for j in model.algebraics:
                    for time in model.huplctime:
                        if j == k:
                            for l in model.huplc_absorbing.value:
                                strsolidspec = "\'" + str(k) + "\'" + ', ' + "\'" + str(
                                    l) + "\'"  # pair of absorbing solid and liquid
                                if l in model.solid_spec_arg2 and strsolidspec in str(
                                        model.solid_spec.keys()):  #check whether pair of solid and liquid in keys and whether liquid in huplcabs species
                                    valY = value(model.Y[time, k]) / value(model.vol)
                                    if valY <= 0:
                                        new_consolidvol.add(model.solidvol[time, l] == 0.0)
                                    else:
                                        new_consolidvol.add(model.solidvol[time, l] == model.Y[time, k] / model.vol)
                                else:
                                    new_consolidvol.add(model.solidvol[time, l] == 0.0)
                for jk in self.comps['all']:
                    for time in model.huplctime:
                        if jk == k:
                            for l in model.huplc_absorbing.value:
                                strsolidspec = "\'" + str(k) + "\'" + ', ' + "\'" + str(
                                    l) + "\'"  # pair of absorbing solid and liquid
                                if l in model.solid_spec_arg2 and strsolidspec in str(
                                        model.solid_spec.keys()):  #check whether pair of solid and liquid in keys and whether liquid in huplcabs species
                                    valZ = value(model.Z[time, k]) / value(model.vol)
                                    if valZ <= 0:
                                        new_consolidvol.add(model.solidvol[time, l] == 0.0)
                                    else:
                                        new_consolidvol.add(model.solidvol[time, l] == model.Z[time, k] / model.vol)
                                else:
                                    new_consolidvol.add(model.solidvol[time, l] == 0.0)

            model.Dhat_bar = Var(model.huplcmeas_times,
                             model.huplc_absorbing)

            model.Dhat_bar_constraint = Constraint(model.huplcmeas_times,
                                               model.huplc_absorbing,
                                               rule=rule_Dhat_bar)

            for t in model.huplcmeas_times:
                list_huplcabs = [k for k in model.huplc_absorbing.value]
                for k in list_huplcabs:
                    third_term += (model.Dhat[t, k] - model.Dhat_bar[t, k]) ** 2 / sigma_sq['device-huplc']

        model.objective.expr += third_term

        return None

    def run_param_est_with_subset_lambdas(self, builder_clone, end_time, subset, nfe, ncp, sigmas, solver='ipopt'):
        """ Performs the parameter estimation with a specific subset of wavelengths.
            At the moment, this is performed as a totally new Pyomo model, based on the
            original estimation. Initialization strategies for this will be needed.

        :param TemplateBuilder builder_clone: Template builder class of complete model without the data added yet
        :param float end_time: the end time for the data and simulation
        :param list subset: list of selected wavelengths
        :param int nfe: number of finite elements
        :param int ncp: number of collocation points
        :param dict sigmas: dictionary containing the variances, as used in the ParameterEstimator class

        :return ResultsObject results: The solved pyomo model results

        """
        if not isinstance(subset, (list, dict)):
            raise RuntimeError("subset must be of type list or dict!")

        if isinstance(subset, dict):
            lists1 = sorted(subset.items())
            x1, y1 = zip(*lists1)
            subset = list(x1)
        
        # This is the filter for creating the new data subset
        old_D = convert(self.model.D)
        new_D = old_D.loc[:, subset] 
        
        
        print(end_time, new_D)
        # Now that we have a new DataFrame, we need to build the entire problem from this
        # An entire new ParameterEstimation problem should be set up, on the outside of
        # this function and class structure, from the model already developed by the user.
        new_template = construct_model_from_reduced_set(builder_clone, end_time, new_D)
        # need to put in an optional running of the variance estimator for the new
        # parameter estiamtion run, or just use the previous full model run to initialize...

        results, lof = run_param_est(new_template, nfe, ncp, sigmas, solver=solver)

        return results

    def run_lof_analysis(self, builder_before_data, end_time, correlations, lof_full_model, nfe, ncp, sigmas,
                         step_size=0.2, search_range=(0, 1)):
        """ Runs the lack of fit minimization problem used in the Michael's Reaction paper
        from Chen et al. (submitted). To use this function, the full parameter estimation
        problem should be solved first and the correlations for wavelngths from this optimization
        need to be supplied to the function as an option.


        :param TemplateBuilder builder_before_data: Template builder class of complete model without the data added yet
        :param int end_time: the end time for the data and simulation
        :param dict correlations: dictionary containing the wavelengths and their correlations to the concentration profiles
        :param int lof_full_model: the value of the lack of fit of the full model (with all wavelengths)
        :param int nfe: number of finite elements
        :param int ncp: number of collocation points
        :param dict sigmas: dictionary containing the variances, as used in the ParameterEstimator class
        :param float step_size: The spacing used in correlation thresholds
        :param tuple search_range: correlation bounds within to search

        :return: None

        """
        if not isinstance(step_size, float):
            raise RuntimeError("step_size must be a float between 0 and 1")
        elif step_size >= 1 or step_size <= 0:
            return RuntimeError("step_size must be a float between 0 and 1")

        if not isinstance(search_range, tuple):
            raise RuntimeError("search range must be a tuple")
        elif search_range[0] < 0 or search_range[0] > 1 and not (
                isinstance(search_range, float) or isinstance(search_range, int)):
            raise RuntimeError("search range lower value must be between 0 and 1 and must be type float")
        elif search_range[1] < 0 or search_range[1] > 1 and not (
                isinstance(search_range, float) or isinstance(search_range, int)):
            raise RuntimeError("search range upper value must be between 0 and 1 and must be type float")
        elif search_range[1] <= search_range[0]:
            raise RuntimeError("search_range[1] must be bigger than search_range[0]!")
        # firstly we will run the initial search from at increments of 20 % for the correlations
        # we already have lof(0) so we want 10,30,50,70, 90.
        
        count = 0
        filt = 0.0
        initial_solutions = list()
        initial_solutions.append((0, lof_full_model))
        while filt < search_range[1]:
            filt += step_size
            if filt > search_range[1]:
                break
            elif filt == 1:
                break
            new_subs = wavelength_subset_selection(correlations=correlations, n=filt)
            lists1 = sorted(new_subs.items())
            x1, y1 = zip(*lists1)
            x = list(x1)

            old_D = convert(self.model.D)
            new_D = old_D.loc[:, new_subs]

            # opt_model, nfe, ncp = construct_model_from_reduced_set(builder_before_data, end_time, new_D)
            # Now that we have a new DataFrame, we need to build the entire problem from this
            # An entire new ParameterEstimation problem should be set up, on the outside of
            # this function and class structure, from the model already developed by the user.
            new_template = construct_model_from_reduced_set(builder_before_data, end_time, new_D)
            # need to put in an optional running of the variance estimator for the new
            # parameter estimation run, or just use the previous full model run to initialize...
            results, lof = run_param_est(new_template, nfe, ncp, sigmas)
            initial_solutions.append((filt, lof))

            count += 1

        count = 0
        for x in initial_solutions:
            print(f'When wavelengths of less than {x[0]:0.3f} correlation are removed')
            print(f'The lack of fit is {x[1]:0.6f} %')

    # =============================================================================
    # --------------------------- DIAGNOSTIC TOOLS ------------------------
    # =============================================================================

    def lack_of_fit(self):
        """ Runs basic post-processing lack of fit analysis

        :return float lack of fit: percentage lack of fit

        """
        # Unweighted SSE - Difference between the calculated and measured D
        if self._spectra_given:
        
            sum_e = 0
            sum_d = 0
            C = np.zeros((len(self.model.times_spectral), self.n_val))
            S = np.zeros((len(self.model.meas_lambdas), self.n_val))
            
            for c_count, c in enumerate(self.comps['unknown_absorbance']):
                for t_count, t in enumerate(self.model.times_spectral):
                    C[t_count, c_count] = getattr(self.model, 'C')[t, c].value
    
            for c_count, c in enumerate(self.comps['unknown_absorbance']):
                for l_count, l in enumerate(self.model.meas_lambdas):
                    S[l_count, c_count] = self.model.S[l, c].value
                 
            D_model = C.dot(S.T)
            
            for t_count, t in enumerate(self.model.times_spectral):
                for l_count, l in enumerate(self.model.meas_lambdas):
                    sum_e += (D_model[t_count, l_count] - self.model.D[t, l]) ** 2
                    sum_d += (self.model.D[t, l]) ** 2
      
            lof = np.sqrt(sum_e/sum_d)*100
            
        return lof

    def wavelength_correlation(self):
        """ determines the degree of correlation between the individual wavelengths and
        the and the concentrations.


        :return dict: dictionary of correlations with wavelength

        """
        nt = len(self.model.allmeas_times)

        cov_d_l = dict()
        for c in self.comps['unknown_absorbance']:
            for l in self.model.meas_lambdas:
                mean_d = (sum(self.model.D[t, l] for t in self.model.times_spectral) / nt)
                mean_c = (sum(self.model.C[t, c].value for t in self.model.times_spectral) / nt)
                cov_d_l[l, c] = 0
                for t in self.model.times_spectral:
                    cov_d_l[l, c] += (self.model.D[t, l] - mean_d) * (self.model.C[t, c].value - mean_c)

                cov_d_l[l, c] = cov_d_l[l, c] / (nt - 1)

        # calculating the standard devs for dl and ck over time
        s_dl = dict()

        for l in self.model.meas_lambdas:
            s_dl[l] = 0
            mean_d = (sum(self.model.D[t, l] for t in self.model.times_spectral) / nt)
            error = 0
            for t in self.model.times_spectral:
                error += (self.model.D[t, l] - mean_d) ** 2
            s_dl[l] = (error / (nt - 1)) ** 0.5

        s_ck = dict()

        for c in self.comps['unknown_absorbance']:
            s_ck[c] = 0
            mean_c = (sum(self.model.C[t, c].value for t in self.model.times_spectral) / nt)
            error = 0
            for t in self.model.times_spectral:
                error += (self.model.C[t, c].value - mean_c) ** 2
            s_ck[c] = (error / (nt - 1)) ** 0.5

        cor_lc = dict()

        for c in self.comps['unknown_absorbance']:
            for l in self.model.meas_lambdas:
                cor_lc[l, c] = cov_d_l[l, c] / (s_ck[c] * s_dl[l])

        cor_l = dict()
        for l in self.model.meas_lambdas:
            cor_l[l] = max(cor_lc[l, c] for c in self.comps['unknown_absorbance'])

        return cor_l

    def lack_of_fit_huplc(self):
        """ Runs basic post-processing lack of fit analysis

        :return float lack of fit: percentage lack of fit

        """
        nc = len(self.model.huplc_absorbing)
        sum_e = 0
        sum_d = 0
        D_model = np.zeros((len(self.model.huplcmeas_times), nc))
        
        for t_count, t in enumerate(self.model.huplcmeas_times):
            for l_count, l in enumerate(self._list_huplcabs):
                
                if hasattr(self.model, 'solidvol'):
                    D_model[t_count, l_count] = (self.model.Z[t, l].value + self.model.solidvol[t, l].value) / (
                        sum(self.model.Z[t, j].value + self.model.solidvol[t, j].value for j in self._list_huplcabs))
                else:
                    D_model[t_count, l_count] = (self.model.Z[t, l].value) / (
                        sum(self.model.Z[t, j].value for j in self._list_huplcabs))

                sum_e += (D_model[t_count, l_count] - self.model.Dhat[t, l].value) ** 2
                sum_d += (self.model.Dhat[t, l].value) ** 2

        lof = np.sqrt((sum_e/sum_d))*100

        print(f"The lack of fit for the huplc data is {lof:0.2f}%")
        return lof
    
    def g_handling_status_messages(self):
        """Determines the unwanted contribution type and informs the user of this

        :return: None

        """
        if self.G_contribution == 'unwanted_G':
            print("\nType of unwanted contributions not set, so assumed that it is time-variant.\n")
            self.G_contribution = 'time_variant_G'
        elif self.G_contribution == 'time_variant_G':
            print("\nTime-variant unwanted contribution is involved.\n")
        elif self.G_contribution == 'time_invariant_G_decompose':
            print("\nTime-invariant unwanted contribution is involved and G can be decomposed.\n")
        elif self.G_contribution == 'time_invariant_G_no_decompose':
            print("\nTime-invariant unwanted contribution is involved but G cannot be decomposed.\n")
        return None
     
    def decompose_G_test(self, St, Z_in):
        """Check whether or not G can be decomposed

        :param dict St: Reaction coefficient matrix
        :param dict Z_in: Dosing points

        :return: None

        """
        if St == dict() and Z_in == dict():
            raise RuntimeError('Because time-invariant unwanted contribution is chosen, please provide information of St or Z_in to build omega matrix.')
        
        omega_list = [St[i] for i in St.keys()]
        omega_list += [Z_in[i] for i in Z_in.keys()]
        omega_sub = np.array(omega_list)
        rank = np.linalg.matrix_rank(omega_sub)
        cols = omega_sub.shape[1]
        rko = cols - rank
        
        if rko > 0:
            self.time_invariant_G_decompose = True
            self.G_contribution = 'time_invariant_G_decompose'
        else:
            self.time_invariant_G_no_decompose = True
            self.G_contribution = 'time_invariant_G_no_decompose'
        return None

def wavelength_subset_selection(correlations=None, n=None):
    """ identifies the subset of wavelengths that needs to be chosen, based
    on the minimum correlation value set by the user (or from the automated
    lack of fit minimization procedure)

    :param dict correlations: dictionary obtained from the wavelength_correlation  function, containing every
       wavelength from the original set and their correlations to the concentration profile.
    :param int n: a value between 0 - 1 that slects the minimum amount correlation between the wavelength and the
       concentrations.

    :return: A dictionary of correlations with wavelength
    :rtype: dict

    """
    if not isinstance(correlations, dict):
        raise RuntimeError("correlations must be of type dict()! Use wavelength_correlation function first!")

    # should check whether the dictionary contains all wavelengths or not
    if not isinstance(n, float):
        raise RuntimeError("n must be of type int!")
    elif n > 1 or n < 0:
        raise RuntimeError("n must be a number between 0 and 1")

    subset_dict = dict()
    for l in correlations.keys():
        if correlations[l] >= n:
            subset_dict[l] = correlations[l]
    return subset_dict


# =============================================================================
# ----------- PARAMETER ESTIMATION WITH WAVELENGTH SELECTION ------------------
# =============================================================================

def construct_model_from_reduced_set(builder_clone, end_time, D):
    """ constructs the new pyomo model based on the selected wavelengths.

    :param TemplateBuilder builder_clone: Template builder class of complete model without the data added yet
    :param float end_time: The end time for the data and simulation
    :param pandas.DataFrame D: The new, reduced dataset with only the selected wavelengths.

    :return TemplateBuilder opt_mode: new Pyomo model from TemplateBuilder, ready for parameter estimation

    """
    import pandas as pd
    from kipet.model_tools.template_builder import TemplateBuilder
    from kipet.model_components.spectral_handler import SpectralData
    
    
    if not isinstance(builder_clone, TemplateBuilder):
        raise RuntimeError('builder_clone needs to be of type TemplateBuilder')

    if not isinstance(D, pd.DataFrame):
        raise RuntimeError('Spectral data format not supported. Try pandas.DataFrame')

    builder_clone._spectral_data = D
    
    # spectral_data = SpectralData('D_new', data=D)
    # builder_clone.input_data(None, spectral_data)
    
    opt_model = builder_clone.create_pyomo_model(0.0, end_time, estimator='p_estimator')

    return opt_model


def run_param_est(opt_model, nfe, ncp, sigmas, solver='ipopt'):
    """ Runs the parameter estimator for the selected subset

    :param ConcreteModel opt_model: The model that we wish to run the
    :param int nfe: The number of finite elements
    :param int ncp: The number of collocation points
    :param dict sigmas: A dictionary containing the variances, as used in the ParameterEstimator class

    :return results_pyomo: Parameter Estimation results
    :return lof: lack of fit results

    """
    p_estimator = ParameterEstimator(opt_model)
    p_estimator.apply_discretization('dae.collocation', nfe=nfe, ncp=ncp, scheme='LAGRANGE-RADAU')
    options = dict()

    

    # These may not always solve, so we need to come up with a decent initialization strategy here
    results_pyomo = p_estimator.run_opt('ipopt',
                                        tee=False,
                                        solver_opts=options,
                                        variances=sigmas)
    # else:
    #     results_pyomo = p_estimator.run_opt(solver,
    #                                         tee=False,
    #                                         solver_opts=options,
    #                                         variances=sigmas,
    #                                         covariance=True)
    
    lof = p_estimator.lack_of_fit()

    return results_pyomo, lof
