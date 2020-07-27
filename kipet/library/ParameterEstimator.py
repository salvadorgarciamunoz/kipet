# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import copy
import os
import re
import time

import matplotlib.pyplot as plt
import numpy as np
from pyomo import *
from pyomo.dae import *
from pyomo.environ import *
from pyomo.opt import (
    ProblemFormat,
    SolverFactory, 
    TerminationCondition,
)

from kipet.library.Optimizer import *
from kipet.library.TemplateBuilder import *
from kipet.library.common.jumps import *
from kipet.library.common.read_hessian import *
from kipet.library.spectra_methods.spectra_compute_D import compute_D_given_SC
from kipet.library.spectra_methods.G_handling import (
    decompose_G_test,
    g_handling_status_messages,
    g_handling_check_options,
    )
from kipet.library.common.objectives import conc_objective

class ParameterEstimator(Optimizer):
    """Optimizer for parameter estimation.

    Parameters
    ----------
    model : Pyomo model
        Pyomo model to be used in the parameter estimation

    """
    def __init__(self, model):
        super(ParameterEstimator, self).__init__(model)

        self.hessian = None
        self._estimability = False
        self._idx_to_variable = dict()
        self._n_actual = self._n_components
        self.model_variance = True
        self.termination_condition = None
        
        # This should be a subclass or a mixin
        self.unwanted_G = False
        self.time_variant_G = False
        self.time_invariant_G = False
        self.time_invariant_G_decompose = False
        self.time_invariant_G_no_decompose = False

        if hasattr(self.model, 'non_absorbing'):
            warnings.warn("Overriden by non_absorbing")
            list_components = [k for k in self._mixture_components if k not in self._non_absorbing]
            self._sublist_components = list_components
            self._n_actual = len(self._sublist_components)

        # for new huplc structure (CS):
        if self._huplc_given:
            list_huplcabs = [k for k in self._huplc_absorbing]
            self._list_huplcabs = list_huplcabs
            self._n_huplc = len(list_huplcabs)

        if hasattr(self.model, 'known_absorbance'):
            warnings.warn("Overriden by known_absorbance")
            list_components = [k for k in self._mixture_components if k not in self._known_absorbance]
            self._sublist_components = list_components
            self._n_actual = len(self._sublist_components)

        else:
            self._sublist_components = [k for k in self._mixture_components]
            
        if hasattr(self, '_abs_components'):
            self.component_set = self._abs_components
            self.component_var = 'Cs'
            self.n_val = self._nabs_components
            
        else:
            self.component_set = self._sublist_components
            self.component_var = 'C'
            self.n_val = self._n_actual
            
    def run_sim(self, solver, **kdws):
        raise NotImplementedError("ParameterEstimator object does not have run_sim method. Call run_opt")

    def run_opt(self, solver, **kwds):

        """ Solves parameter estimation problem.

        Args:
            solver (str): name of the nonlinear solver to used

            solver_opts (dict, optional): options passed to the nonlinear solver

            variances (dict or float, optional): map of component name to noise variance. The
            map also contains the device noise variance. If not float then we only use device variance
            and ignore model variance.

            tee (bool,optional): flag to tell the optimizer whether to stream output
            to the terminal or not.

            with_d_vars (bool,optional): flag to the optimizer whether to add

            variables and constraints for D_bar(i,j).
            
            report_time (bool, optional): flag as to whether to time the parameter estimation or not.

            estimability (bool, optional): flag to tell the model whether it is
            being used by the estimability analysis and therefore will need to return the
            hessian for analysis.
            
            model_variance (bool, optional): Default is True. Flag to tell whether we are only
            considering the variance in the device, or also model noise as well.

            model_variance (bool, optional): Default is True. Flag to tell whether we are only
            considering the variance in the device, or also model noise as well.

        Returns:
            Results object with loaded results

        """
        run_opt_kwargs = copy.copy(kwds)
        
        solver_opts = kwds.pop('solver_opts', dict())
        variances = kwds.pop('variances', dict())
        tee = kwds.pop('tee', False)
        with_d_vars = kwds.pop('with_d_vars', False)
        covariance = kwds.pop('covariance', False)
        symbolic_solver_labels = kwds.pop('symbolic_solver_labels', False)

        estimability = kwds.pop('estimability', False)
        report_time = kwds.pop('report_time', False)
        model_variance = kwds.pop('model_variance', True)

        # additional arguments for inputs CS
        inputs = kwds.pop("inputs", None)
        inputs_sub = kwds.pop("inputs_sub", None)
        trajectories = kwds.pop("trajectories", None)
        fixedtraj = kwds.pop("fixedtraj", False)
        fixedy = kwds.pop("fixedy", False)
        yfix = kwds.pop("yfix", None)
        yfixtraj = kwds.pop("yfixtraj", None)

        jump = kwds.pop("jump", False)
        var_dic = kwds.pop("jump_states", None)
        jump_times = kwds.pop("jump_times", None)
        feed_times = kwds.pop("feed_times", None)
        
        # user should input if the unwanted contribuiton is involved, and what type it is.
        # If it's time_invariant, St or Z_in should be inputed to check the rank of kernal of Omega matrix. KH.L
        unwanted_G = kwds.pop('unwanted_G', False)
        time_variant_G = kwds.pop('time_variant_G', False)
        time_invariant_G = kwds.pop('time_invariant_G', False)
        St = kwds.pop('St', dict())
        Z_in = kwds.pop('Z_in', dict())

        self.solver = solver
        self.model_variance = model_variance
        self._estimability = estimability

        if not self.model.alltime.get_discretization_info():
            raise RuntimeError('apply discretization first before initializing')

        self.unwanted_G = unwanted_G
        self.time_variant_G = time_variant_G
        self.time_invariant_G = time_invariant_G

        if report_time:
            start = time.time()

        opt = SolverFactory(self.solver)
        
        g_handling_check_options(self)
        if self.time_invariant_G:
            decompose_G_test(self)
        g_handling_status_messages(self)
        
        def rule_init_conditionsnew(mod, k):
            st = mod.start_time.value  # important!
            if k in mod.Pinit.keys():
                bla = mod.X[st, k]
            else:
                bla = mod.Z[st, k]
            return bla == mod.init_conditions[k]
        
        if hasattr(self.model, 'Pinit'):
            
            self.allinitcomps = dict()
            for k in self._mixture_components:
                self.allinitcomps[k] = self.model.init_conditions[k].value
            for i in self._complementary_states:
                self.allinitcomps[i] = self.model.init_conditions[i].value

            self.model.del_component('init_conditions')
            self.model.del_component('init_conditions_c')

            init_dict = dict()
            for k, l in self.allinitcomps.items():
                init_dict[k] = l

            self._init_conditions = Var(self.model.states, initialize=init_dict)
            self.model.add_component('init_conditions', self._init_conditions)
            for k in self.allinitcomps.keys():
                if k in self.model.Pinit.keys():
                    st = self.model.start_time.value
                    self.model.init_conditions[k].fixed = False
                    self.model.init_conditions[k].stale = False
                else:
                    self.model.init_conditions[k].fixed = True
                    self.model.init_conditions[k].stale = True

            for k in self.model.Pinit.keys():
                lb = self.model.Pinit[k].lb #just to take value for bounds and initial values from user input
                ub = self.model.Pinit[k].ub #fixed var

                self.model.init_conditions[k].setlb(lb)
                self.model.init_conditions[k].setub(ub)

            self.model.init_conditions_c = Constraint(self.model.states, rule=rule_init_conditionsnew)

            Pinitset=[k for k in self.model.Pinit.keys()]
            if hasattr(self.model, 'Pinitsetset')==False:
                self.model.add_component('Pinitsetset', Set(initialize=Pinitset))

        if covariance:
            if self.solver != 'ipopt_sens' and self.solver != 'k_aug':
                raise RuntimeError('To get covariance matrix the solver needs to be ipopt_sens or k_aug')
            
            if self.solver == 'ipopt_sens':
                if not 'compute_red_hessian' in solver_opts.keys():
                    solver_opts['compute_red_hessian'] = 'yes'
            
            if self.solver == 'k_aug':
                solver_opts['compute_inv'] = ''

            self._define_reduce_hess_order()

        if inputs_sub is not None:
            from kipet.library.common.additional_inputs import add_inputs
            add_inputs(self)
        
        if jump:
            set_up_jumps(self, run_opt_kwargs)
            
        for key, val in solver_opts.items():
            opt.options[key] = val
            
        active_objectives = [o for o in self.model.component_map(Objective, active=True)]
        if active_objectives:
            print(
                "WARNING: The model has an active objective. Running optimization with models objective.\n"
                " To solve optimization with default objective (Weifengs) deactivate all objectives in the model.")
            solver_results = opt.solve(self.model, tee=tee)

        elif self._spectra_given:
            self._solve_extended_model(variances, opt,
                                       tee=tee,
                                       covariance=covariance,
                                       with_d_vars=with_d_vars,
                                       **kwds)

        elif self._concentration_given:
            self._solve_model_given_c(variances, opt,
                                      tee=tee,
                                      covariance=covariance,
                                      **kwds)
            
        else:
            raise RuntimeError(
                'Must either provide concentration data or spectra in order to solve the parameter estimation problem')

        if report_time:
            end = time.time()
            print("Total execution time in seconds for variance estimation:", end - start)

        return self.get_results()


    def get_results(self):
        """Removed results unit from function"""
    
        results = ResultsObject()

        if self._spectra_given:
            results.load_from_pyomo_model(self.model,
                                          to_load=['Z', 'dZdt', 'X', 'dXdt', 'C', 'S', 'Y'])
            if hasattr(self, '_abs_components'):
                results.load_from_pyomo_model(self.model,
                                              to_load=['Cs'])
            if hasattr(self, 'huplc_absorbing'):
                results.load_from_pyomo_model(self.model,
                                              to_load=['Dhat_bar'])
     
        elif self._concentration_given:
            results.load_from_pyomo_model(self.model,
                                          to_load=['Z', 'dZdt', 'X', 'dXdt', 'C', 'Y'])
        else:
            raise RuntimeError(
                'Must either provide concentration data or spectra in order to solve the parameter estimation problem')

        if self._spectra_given:
            compute_D_given_SC(self, results)

        results.P = {name: self.model.P[name].value for name in self.model.parameter_names}

        if hasattr(self.model, 'Pinit'):
            param_valsinit = dict()
            for name in self.model.initparameter_names:
                param_valsinit[name] = self.model.init_conditions[name].value
            results.Pinit = param_valsinit

        if self.termination_condition!=None and self.termination_condition!=TerminationCondition.optimal:
            raise Exception("The current iteration was unsuccessful.")
        else:
            if self._estimability == True:
                return self.hessian, results
            else:
                return results

        return results

    def _solve_extended_model(self, sigma_sq, optimizer, **kwds):
        """Solves estimation based on spectral data. (known variances)

           This method is not intended to be used by users directly
        Args:
            sigma_sq (dict): variances

            optimizer (SolverFactory): Pyomo Solver factory object

            tee (bool,optional): flag to tell the optimizer whether to stream output
            to the terminal or not

            with_d_vars (bool,optional): flag to the optimizer whether to add
            variables and constraints for D_bar(i,j)

            subset_lambdas (array_like,optional): Set of wavelengths to used in
            the optimization problem (not yet fully implemented). Default all wavelengths.

        Returns:
            None
        """

        tee = kwds.pop('tee', False)
        with_d_vars = kwds.pop('with_d_vars', False)

        if self._huplc_given:  # added for new huplc structure CS
            weights = kwds.pop('weights', [1.0, 1.0, 1.0])
        else:
            weights = kwds.pop('weights', [1.0, 1.0])
        warmstart = kwds.pop('warmstart', False)
        eigredhess2file = kwds.pop('eigredhess2file', False)
        penaltyparam = kwds.pop('penaltyparam', False)
        penaltyparamcon = kwds.pop('penaltyparamcon', False) #added for optional penalty term related to constraint CS
        ppenalty_dict = kwds.pop('ppenalty_dict', None)
        ppenalty_weights = kwds.pop('ppenalty_weights', None)
        covariance = kwds.pop('covariance', False)
        species_list = kwds.pop('subset_components', None)
        set_A = kwds.pop('subset_lambdas', list())
        self._eigredhess2file=eigredhess2file

        if not set_A:
            set_A = self._meas_lambdas

        list_components = []
        if species_list is None:
            list_components = [k for k in self._mixture_components]
        else:
            for k in species_list:
                if k in self._mixture_components:
                    list_components.append(k)
                else:
                    warnings.warn("Ignored {} since is not a mixture component of the model".format(k))

        if not self._spectra_given:
            raise NotImplementedError("Extended model requires spectral data model.D[ti,lj]")

        if hasattr(self.model, 'non_absorbing'):
            warnings.warn("Overriden by non_absorbing!")
            list_components = [k for k in self._mixture_components if k not in self._non_absorbing]

        if hasattr(self.model, 'known_absorbance'):
            warnings.warn("Overriden by known_absorbance!")
            list_components = [k for k in self._mixture_components if k not in self._known_absorbance]

        all_sigma_specified = True
        print(sigma_sq)

        if isinstance(sigma_sq, dict): 
            keys = sigma_sq.keys()
            for k in list_components:
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

        m = self.model
        
        #%%
        # When G is time-invariant and cannot be decomposed, problem (19) should be called.
        # Instead of calling (19), (24) is called but all qr[i] are fixed at 1.0. KH.L
        if self.time_invariant_G_no_decompose:
            for i in m.alltime:
                m.qr[i] = 1.0
            m.qr.fix()

        def _qr_end_constraint(m):
            return m.qr[m.alltime[-1]] == 1.0
        
        if self.time_variant_G or self.unwanted_G:
            m.qr_end_cons = Constraint(rule = _qr_end_constraint)
        
        #%%
        if with_d_vars and self.model_variance:
            m.D_bar = Var(m.meas_times,
                          m.meas_lambdas)
            # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
            #if hasattr(self, '_abs_components'):
            def rule_D_bar(m, t, l):
                if hasattr(m, 'huplc_absorbing') and hasattr(m, 'solid_spec_arg1'):
                    return m.D_bar[t, l] == sum(
                        getattr(m, self.component_var)[t, k] * m.S[l, k] for k in self.component_set if k not in m.solid_spec_arg1)
                else:
                    return m.D_bar[t, l] == sum(getattr(m, self.component_var).Cs[t, k] * m.S[l, k] for k in self.component_set)

            m.D_bar_constraint = Constraint(m.meas_times,
                                            m.meas_lambdas,
                                            rule=rule_D_bar)
        
        #For addition of huplc data and matching liquid and solid species (CS):
        def rule_Dhat_bar(m, t, l):
            list_huplcabs = [k for k in m.huplc_absorbing.value]
            return m.Dhat_bar[t, l] == (m.Z[t, l] + m.solidvol[t, l]) / (
                sum(m.Z[t, j] + m.solidvol[t, j] for j in list_huplcabs))

        # estimation with model variance and device:

        def rule_objective(m):
            expr = 0
            for t in m.meas_times:
                for l in m.meas_lambdas:
                    if with_d_vars:
                        if self.unwanted_G or self.time_variant_G:
                            expr += (m.D[t, l] - m.D_bar[t, l] - m.qr[t]*m.g[l]) ** 2 / (sigma_sq['device'])
                        elif self.time_invariant_G_no_decompose:
                            expr += (m.D[t, l] - m.D_bar[t, l] - m.g[l]) ** 2 / (sigma_sq['device'])
                        else:
                            expr += (m.D[t, l] - m.D_bar[t, l]) ** 2 / (sigma_sq['device'])
                    else:
                        # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
                        if hasattr(m, '_abs_components'):
                            if hasattr(m, 'huplc_absorbing') and hasattr(m, 'solid_spec_arg1'):
                                D_bar = sum(
                                    m.Cs[t, k] * m.S[l, k] for k in m._abs_components if k not in m.solid_spec_arg1)
                            else:
                                D_bar = sum(m.Cs[t, k] * m.S[l, k] for k in m._abs_components)
                            if self.unwanted_G or self.time_variant_G:
                                expr += (m.D[t, l] - D_bar - m.qr[t]*m.g[l]) ** 2 / (sigma_sq['device'])
                            elif self.time_invariant_G_no_decompose:
                                expr += (m.D[t, l] - D_bar - m.g[l]) ** 2 / (sigma_sq['device'])
                            else:
                                expr += (m.D[t, l] - D_bar) ** 2 / (sigma_sq['device'])
                        else:
                            if hasattr(m, 'huplc_absorbing') and hasattr(m, 'solid_spec_arg1'):
                                D_bar = sum(
                                    m.C[t, k] * m.S[l, k] for k in list_components if k not in m.solid_spec_arg1)
                            else:
                                D_bar = sum(m.C[t, k] * m.S[l, k] for k in list_components)
                            if self.unwanted_G or self.time_variant_G:
                                expr += (m.D[t, l] - D_bar - m.qr[t]*m.g[l]) ** 2 / (sigma_sq['device'])
                            elif self.time_invariant_G_no_decompose:
                                expr += (m.D[t, l] - D_bar - m.g[l]) ** 2 / (sigma_sq['device'])
                            else:
                                expr += (m.D[t, l] - D_bar) ** 2 / (sigma_sq['device'])

            expr *= weights[0]
            second_term = 0.0
           
            for t in m.meas_times:
                second_term += sum((m.C[t, k] - m.Z[t, k]) ** 2 / sigma_sq[k] for k in list_components)
                
            expr += weights[1] * second_term

            #for addition of L2 penalty term to objective penalizing values that deviate from values defined in ppenalty_dict (CS):
            if penaltyparam==True:
                if ppenalty_weights is None:
                    fourth_term = 0.0
                    for k in m.P.keys():
                        if k in ppenalty_dict.keys():
                            fourth_term = (m.P[k] - ppenalty_dict[k]) ** 2
                else:
                    if len(ppenalty_dict)!=len(ppenalty_weights):
                        raise RuntimeError(
                            'For every penalty term a weight must be defined.')
                    if ppenalty_dict.keys()!=ppenalty_weights.keys():
                        raise RuntimeError(
                            'Check the parameter names in ppenalty_weights and ppenalty_dict again. They must match.')
                    else:
                        fourth_term = 0.0
                        for k in m.P.keys():
                            if k in ppenalty_dict.keys():
                                fourth_term += ppenalty_weights[k] * (m.P[k] - ppenalty_dict[k]) ** 2
                expr +=fourth_term


            # for new huplc structure (CS):
            if hasattr(m, 'huplc_absorbing'):
                third_term = 0.0
                if not 'device-huplc' in sigma_sq.keys():
                    sigma_sq['device-huplc'] = 1.0
                if hasattr(m, 'solid_spec_arg1') and hasattr(m, 'solid_spec_arg2'):
                    solidvol_dict = dict()
                    m.add_component('cons_solidvol', ConstraintList())
                    new_consolidvol = getattr(m, 'cons_solidvol')
                    m.add_component('solidvol', Var(m.huplctime, m.huplc_absorbing.value, initialize=solidvol_dict))

                    for k in m.solid_spec_arg1:
                        for j in m.algebraics:
                            for time in m.huplctime:
                                if j == k:
                                    for l in m.huplc_absorbing.value:
                                        strsolidspec = "\'" + str(k) + "\'" + ', ' + "\'" + str(
                                            l) + "\'"  # pair of absorbing solid and liquid
                                        if l in m.solid_spec_arg2 and strsolidspec in str(
                                                m.solid_spec.keys()):  #check whether pair of solid and liquid in keys and whether liquid in huplcabs species
                                            valY = value(m.Y[time, k]) / value(m.vol)
                                            if valY <= 0:
                                                new_consolidvol.add(m.solidvol[time, l] == 0.0)
                                            else:
                                                new_consolidvol.add(m.solidvol[time, l] == m.Y[time, k] / m.vol)
                                        else:
                                            new_consolidvol.add(m.solidvol[time, l] == 0.0)
                        for jk in list_components:
                            for time in m.huplctime:
                                if jk == k:
                                    for l in m.huplc_absorbing.value:
                                        strsolidspec = "\'" + str(k) + "\'" + ', ' + "\'" + str(
                                            l) + "\'"  # pair of absorbing solid and liquid
                                        if l in m.solid_spec_arg2 and strsolidspec in str(
                                                m.solid_spec.keys()):  #check whether pair of solid and liquid in keys and whether liquid in huplcabs species
                                            valZ = value(m.Z[time, k]) / value(m.vol)
                                            if valZ <= 0:
                                                new_consolidvol.add(m.solidvol[time, l] == 0.0)
                                            else:
                                                new_consolidvol.add(m.solidvol[time, l] == m.Z[time, k] / m.vol)
                                        else:
                                            new_consolidvol.add(m.solidvol[time, l] == 0.0)
                    else:
                        new_consolidvol.add(m.solidvol[time, l] == 0.0)


                    m.Dhat_bar = Var(m.huplcmeas_times,
                                     m.huplc_absorbing)

                    m.Dhat_bar_constraint = Constraint(m.huplcmeas_times,
                                                       m.huplc_absorbing,
                                                       rule=rule_Dhat_bar)

                    for t in m.huplcmeas_times:
                        list_huplcabs = [k for k in m.huplc_absorbing.value]
                        for k in list_huplcabs:
                            third_term += (m.Dhat[t, k] - m.Dhat_bar[t, k]) ** 2 / sigma_sq['device-huplc']

                expr += weights[2] * third_term

                if penaltyparamcon == True: #added for optional penalty term related to constraint CS
                    #if ppenalty_weights is None:
                    # fifth_term = 0.0
                    rho=1e-1
                    sumpen=0.0
                    for t in m.alltime:
                        sumpen = sumpen + m.Y[t,'npen']
                    fifth_term = rho*sumpen
                    expr += fifth_term
            return expr

        # estimation without model variance and only device variance
        def rule_objective_device_only(m):
            
            if hasattr(m, 'huplc_absorbing'):
                component_set = m._abs_components
            else:
                component_set = list_components
            print(component_set)
            
            expr = 0
            for t in m.meas_times:
                for l in m.meas_lambdas:
                    if with_d_vars:
                        if self.unwanted_G or self.time_variant_G:
                            expr += (m.D[t, l] - m.D_bar[t, l] - m.qr[t]*m.g[l]) ** 2 / (sigma_sq['device'])
                        elif self.time_invariant_G_no_decompose:
                            expr += (m.D[t, l] - m.D_bar[t, l] - m.g[l]) ** 2 / (sigma_sq['device'])
                        else:
                            expr += (m.D[t, l] - m.D_bar[t, l]) ** 2 / (sigma_sq['device'])
                    else:
                        if hasattr(m, 'huplc_absorbing'):
                            D_bar = sum(
                                m.Z[t, k] * m.S[l, k] for k in component_set if k not in m.solid_spec_arg1)
                        else:
                            D_bar = sum(m.Z[t, k] * m.S[l, k] for k in component_set)
                        if self.unwanted_G or self.time_variant_G:
                            expr += (m.D[t, l] - D_bar - m.qr[t]*m.g[l]) ** 2 / (sigma_sq['device'])
                        elif self.time_invariant_G_no_decompose:
                            expr += (m.D[t, l] - D_bar - m.g[l]) ** 2 / (sigma_sq['device'])
                        else:
                            expr += (m.D[t, l] - D_bar) ** 2 / (sigma_sq['device'])
                      
            # for new huplc structure CS:
            if hasattr(m, 'huplc_absorbing'):
                third_term = 0.0
                if not 'device-huplc' in sigma_sq.keys():
                    sigma_sq['device-huplc'] = 1.0
                if m.solid_spec_arg1 is not None and m.solid_spec_arg2 is not None:
                    solidvol_dict = dict()
                    m.add_component('cons_solidvol', ConstraintList())
                    new_consolidvol = getattr(m, 'cons_solidvol')
                    m.add_component('solidvol', Var(m.huplctime, m.huplc_absorbing.value, initialize=solidvol_dict))

                    for k in m.solid_spec_arg1:
                        for j in m.algebraics:
                            for time in m.huplctime:
                                if j == k:
                                    for l in m.huplc_absorbing.value:
                                        strsolidspec = "\'" + str(k) + "\'" + ', ' + "\'" + str(
                                            l) + "\'"  # pair of absorbing solid and liquid
                                        if l in m.solid_spec_arg2 and strsolidspec in str(
                                                m.solid_spec.keys()):  #check whether pair of solid and liquid in keys and whether liquid in huplcabs species
                                            valY = value(m.Y[time, k]) / value(m.vol)
                                            if valY <= 0:
                                                new_consolidvol.add(m.solidvol[time, l] == 0.0)
                                            else:
                                                new_consolidvol.add(m.solidvol[time, l] == m.Y[time, k] / m.vol)
                                        else:
                                            new_consolidvol.add(m.solidvol[time, l] == 0.0)
                        for jk in list_components:
                            for time in m.huplctime:
                                if jk == k:
                                    for l in m.huplc_absorbing.value:
                                        strsolidspec = "\'" + str(k) + "\'" + ', ' + "\'" + str(
                                            l) + "\'"  # pair of absorbing solid and liquid
                                        if l in m.solid_spec_arg2 and strsolidspec in str(
                                                m.solid_spec.keys()): #check whether pair of solid and liquid in keys and whether liquid in huplcabs species
                                            valZ = value(m.Z[time, k]) / value(m.vol)
                                            if valZ <= 0:
                                                new_consolidvol.add(m.solidvol[time, l] == 0.0)
                                            else:
                                                new_consolidvol.add(m.solidvol[time, l] == m.Z[time, k] / m.vol)
                                        else:
                                            new_consolidvol.add(m.solidvol[time, l] == 0.0)

                    m.Dhat_bar = Var(m.huplcmeas_times,
                                     m.huplc_absorbing)

                    m.Dhat_bar_constraint = Constraint(m.huplcmeas_times,
                                                       m.huplc_absorbing,
                                                       rule=rule_Dhat_bar)

                    for t in m.huplcmeas_times:
                        list_huplcabs = [k for k in m.huplc_absorbing.value]
                        for k in list_huplcabs:
                            third_term += (m.Dhat[t, k] - m.Dhat_bar[t, k]) ** 2 / sigma_sq['device-huplc']

                expr += weights[2] * third_term
             
            return expr

        if self.model_variance == True:
            m.objective = Objective(rule=rule_objective)
        else:
            m.objective = Objective(rule=rule_objective_device_only)

        if covariance and self.solver == 'ipopt_sens':
            if self.model_variance == False:
                print("WARNING: FOR PROBLEMS WITH NO MODEL VARIANCE it is advised to use k_aug!!!")
            self._tmpfile = "ipopt_hess"
            solver_results = optimizer.solve(m, tee=tee,
                                             logfile=self._tmpfile,
                                             report_timing=True)

            # self.model.red_hessian.pprint
            print("Done solving building reduce hessian")
            output_string = ''
            with open(self._tmpfile, 'r') as f:
                output_string = f.read()
            if os.path.exists(self._tmpfile):
                os.remove(self._tmpfile)
            # output_string = f.getvalue()
            ipopt_output, hessian_output = split_sipopt_string(output_string)
            #print(hessian_output)
            print("build strings")
            #if tee == True:
                #print(ipopt_output)

            if not all_sigma_specified:
                raise RuntimeError(
                    'All variances must be specified to determine covariance matrix.\n Please pass variance dictionary to run_opt')

            n_vars = len(self._idx_to_variable)
            print('n_vars', n_vars)
            hessian = read_reduce_hessian(hessian_output, n_vars)
            
            print(hessian.size, "hessian size")
            print(hessian.shape,"hessian shape")
            # hessian = read_reduce_hessian2(hessian_output,n_vars)
            # print hessian
            if self.model_variance:
                self._compute_covariance(hessian, sigma_sq)
            else:
                self._compute_covariance_no_model_variance(hessian, sigma_sq)

        if warmstart==True:
            if hasattr(m,'dual') and hasattr(m,'ipopt_zL_out') and hasattr(m,'ipopt_zU_out') and hasattr(m,'ipopt_zL_in') and hasattr(m,'ipopt_zU_in'):
                self.update_warm_start(m)
            else:
                self.add_warm_start_suffixes(m, use_k_aug=False)
                
        if covariance and self.solver == 'k_aug':
            self.add_warm_start_suffixes(m, use_k_aug=True)   
            
            count_vars = 1

            if not self._spectra_given:
                pass
            else:
                if self.model_variance:
                     for t in self._allmeas_times:
                        for c in self.component_set:
                            getattr(m, self.component_var)[t, c].set_suffix_value(m.dof_v, count_vars)
                            count_vars += 1
            
            if not self._spectra_given:
                pass
            else:
                for l in self._meas_lambdas:
                    for c in self.component_set:
                        m.S[l, c].set_suffix_value(m.dof_v, count_vars)
                        count_vars += 1

            for v in m.P.values():
                if v.is_fixed():
                    continue
                m.P.set_suffix_value(m.dof_v, count_vars)
                count_vars += 1

            if hasattr(m,'Pinit'):
                for k, v in self.model.Pinit.items():
                    m.init_conditions[k].set_suffix_value(m.dof_v, count_vars)
                    count_vars += 1

            print("SET FOR k_aug")
            self._tmpfile = "k_aug_hess"
            ip = SolverFactory('ipopt')
            solver_results = ip.solve(m, tee=tee,
                                      logfile=self._tmpfile,
                                      report_timing=True)
            k_aug = SolverFactory('k_aug')
            # k_aug.options["compute_inv"] = ""
            self.update_warm_start(m)
            # m.write(filename="mynl.nl", format=ProblemFormat.nl)
            k_aug.solve(m, tee=tee)
            print("Done solving building reduce hessian")

            if not all_sigma_specified:
                raise RuntimeError(
                    'All variances must be specified to determine covariance matrix.\n Please pass variance dictionary to run_opt')

            n_vars = len(self._idx_to_variable)
            print("n_vars", n_vars)
            #m.rh_name.pprint()
            var_loc = m.rh_name
            for v in self._idx_to_variable.values():
                try:
                    var_loc[v]
                except:
                    # print(v, "is an error")
                    var_loc[v] = 0
                    # print(v, "is thus set to ", var_loc[v])
                    # print(var_loc[v])

            vlocsize = len(var_loc)
            # print("var_loc size, ", vlocsize)
            unordered_hessian = np.loadtxt('result_red_hess.txt')
            if os.path.exists('result_red_hess.txt'):
                os.remove('result_red_hess.txt')
            
            hessian = self._order_k_aug_hessian(unordered_hessian, var_loc)
            if self._estimability == True:
                self.hessian = hessian
            if self.model_variance:
                self._compute_covariance(hessian, sigma_sq)
            else:
                self._compute_covariance_no_model_variance(hessian, sigma_sq)
        else:
            solver_results = optimizer.solve(m, tee=tee)
           
        if with_d_vars:
            m.del_component('D_bar')
            m.del_component('D_bar_constraint')
        m.del_component('objective')

        return None
    
    def _solve_model_given_c(self, sigma_sq, optimizer, **kwds):
        """Solves estimation based on concentration data. (known variances)

           This method is not intended to be used by users directly
        Args:
            sigma_sq (dict): variances

            optimizer (SolverFactory): Pyomo Solver factory object

            tee (bool,optional): flag to tell the optimizer whether to stream output
            to the terminal or not

        Returns:
            None
        """

        tee = kwds.pop('tee', False)
        if self._huplc_given:  # added for new huplc structure CS
            weights = kwds.pop('weights', [0.0, 1.0, 1.0])
        else:
            weights = kwds.pop('weights', [0.0, 1.0])
        covariance = kwds.pop('covariance', False)
        warmstart = kwds.pop('warmstart', False)
        eigredhess2file = kwds.pop('eigredhess2file', False)
        penaltyparam = kwds.pop('penaltyparam', False)
        penaltyparamcon = kwds.pop('penaltyparamcon', False) #added for optional penalty term related to constraint CS
        ppenalty_dict = kwds.pop('ppenalty_dict', None)
        ppenalty_weights = kwds.pop('ppenalty_weights', None)
        species_list = kwds.pop('subset_components', None)
        symbolic_solver_labels = kwds.pop('symbolic_solver_labels', False)

        list_components = []
        if species_list is None:
            list_components = [k for k in self._mixture_components]
        else:
            for k in species_list:
                if k in self._mixture_components:
                    list_components.append(k)
                else:
                    warnings.warn("Ignored {} since is not a mixture component of the model".format(k))

        if not self._concentration_given:
            raise NotImplementedError(
                "Parameter Estimation from concentration data requires concentration data model.C[ti,cj]")

        all_sigma_specified = True

        keys = sigma_sq.keys()
        for k in list_components:
            if hasattr(self, 'huplc_absorbing') and k not in keys:
                for ki in self.solid_spec_args1:
                    for key in keys:
                        if key == ki:
                            sigma_sq[ki] = sigma_sq[key]

            elif hasattr(self, 'huplc_absorbing') == False and k not in keys:
                all_sigma_specified = False
                sigma_sq[k] = max(sigma_sq.values())

        m = self.model

        # estimation
        def rule_objective(m):
            obj=0
            if penaltyparamcon == True: #added for optional penalty term related to constraint CS
                rho = 100
                sumpen = 0.0
                obj = conc_objective(m)
                for t in m.allmeas_times:
                    sumpen += m.Y[t, 'npen']
                fifth_term = rho * sumpen
                obj += fifth_term
            else:
                obj = conc_objective(m)
            return obj

        m.objective = Objective(rule=rule_objective)

        if warmstart==True:
            if hasattr(m,'dual') and hasattr(m,'ipopt_zL_out') and hasattr(m,'ipopt_zU_out') and hasattr(m,'ipopt_zL_in') and hasattr(m,'ipopt_zU_in'):
                self.update_warm_start(m)
            else:
                self.add_warm_start_suffixes(m)

        if covariance and self.solver == 'ipopt_sens':
            self._tmpfile = "ipopt_hess"
            solver_results = optimizer.solve(m, tee=False,
                                             logfile=self._tmpfile,
                                             report_timing=True)
            #self.model.red_hessian.pprint
            # m.P.pprint()
            print("Done solving building reduce hessian")
            output_string = ''
            with open(self._tmpfile, 'r') as f:
                output_string = f.read()

                print("output_string", output_string)
            if os.path.exists(self._tmpfile):
                os.remove(self._tmpfile)
            # output_string = f.getvalue()
            ipopt_output, hessian_output = split_sipopt_string(output_string)
            # print hessian_output
            print("build strings")
            # if tee == True:
            #    print(ipopt_output)

            if not all_sigma_specified:
                raise RuntimeError(
                    'All variances must be specified to determine covariance matrix.\n Please pass variance dictionary to run_opt')

            n_vars = len(self._idx_to_variable)
            hessian = read_reduce_hessian(hessian_output, n_vars)
            print(hessian.size, "hessian size")
            # hessian = read_reduce_hessian2(hessian_output,n_vars)
            if self._concentration_given:
                self._compute_covariance_C(hessian, sigma_sq)

        elif covariance and self.solver == 'k_aug':
            self.add_warm_start_suffixes(m, use_k_aug=True)
            
            m.dof_v.pprint()
            m.rh_name.pprint()
            count_vars = 1

            if self._spectra_given:
                for t in self._allmeas_times:
                    for c in self._sublist_components:
                        m.C[t, c].set_suffix_value(m.dof_v, count_vars)
                        count_vars += 1
                        
                for l in self._meas_lambdas:
                    for c in self._sublist_components:
                        m.S[l, c].set_suffix_value(m.dof_v, count_vars)
                        count_vars += 1

            for v in m.P.values():
                if v.is_fixed():
                    continue
                m.P.set_suffix_value(m.dof_v, count_vars)
                count_vars += 1

            if hasattr(m,'Pinit'):
                for v in self.model.initparameter_names:
                    m.init_conditions[v].set_suffix_value(m.dof_v, count_vars)
                    count_vars += 1

            self._tmpfile = "k_aug_hess"
            ip = SolverFactory('ipopt')
            solver_results = ip.solve(m, 
                                      tee=False,
                                      logfile=self._tmpfile,
                                      report_timing=True, 
                                      symbolic_solver_labels=True)
            ##############################################
            # Try different options in case it fails! (CS)
            ############################################
            self.termination_condition = solver_results.solver.termination_condition
            if self.termination_condition != TerminationCondition.optimal:
                print("WARNING: The solution of the iteration was unsuccessful. The problem is solved with additional solver options.")
                optimizer.options["OF_start_with_resto"] = 'yes'
                solver_results = optimizer.solve(m, tee=tee, symbolic_solver_labels=True)
                self.termination_condition = solver_results.solver.termination_condition
                if self.termination_condition != TerminationCondition.optimal:
                    print(
                        "WARNING: The solution of the iteration was unsuccessful. The problem is solved with additional solver options.")
                    optimizer.options["OF_start_with_resto"] = 'no'
                    # optimizer.options["OF_bound_push"] = 1E-02
                    optimizer.options["OF_bound_relax_factor"] = 1E-05
                    solver_results = optimizer.solve(m, tee=tee, symbolic_solver_labels=True)
                    self.termination_condition = solver_results.solver.termination_condition
                    # options["OF_bound_relax_factor"] = 1E-08
                    if self.termination_condition != TerminationCondition.optimal:
                        print("The current iteration was unsuccessful.")
            #############################################
            # m.P.pprint()
            k_aug = SolverFactory('k_aug')

            # k_aug.options["no_scale"] = ""
            self.update_warm_start(m)
            # m.write(filename="mynl.nl", format=ProblemFormat.nl)
            k_aug.solve(m, tee=False)
            print("Done solving building reduce hessian")

            if not all_sigma_specified:
                raise RuntimeError(
                    'All variances must be specified to determine covariance matrix.\n Please pass variance dictionary to run_opt')

            n_vars = len(self._idx_to_variable)
            print("n_vars", n_vars)
            # m.rh_name.pprint()
            
            var_loc = m.rh_name
            for v in self._idx_to_variable.values():
                try:
                    var_loc[v]
                except:
                    # print(v, "is an error")
                    var_loc[v] = 0
                    # print(v, "is thus set to ", var_loc[v])
                    # print(var_loc[v])

            vlocsize = len(var_loc)
            print("var_loc size, ", vlocsize)
            unordered_hessian = np.loadtxt('result_red_hess.txt')
            if os.path.exists('result_red_hess.txt'):
                os.remove('result_red_hess.txt')
            # hessian = read_reduce_hessian_k_aug(hessian_output, n_vars)
            # hessian =hessian_output
            # print(hessian)
            print(unordered_hessian.size, "unordered hessian size")
            hessian = self._order_k_aug_hessian(unordered_hessian, var_loc)
            if self._estimability == True:
                self.hessian = hessian
            if self._concentration_given:
                self._compute_covariance_C(hessian, sigma_sq)
        elif self.solver == 'gams' and covariance==False: #To use conopt as alternative NLP solver, CS
            ip = SolverFactory('gams')
            solver_results = ip.solve(m, solver='conopt', tee=True)
            ##############################################
        else:
            solver_results = optimizer.solve(m, tee=tee, symbolic_solver_labels=True)
            ##############################################
            #Try different options in case it fails! (CS)
            ############################################
            self.termination_condition = solver_results.solver.termination_condition
            if self.termination_condition!= TerminationCondition.optimal:
                print("WARNING: The solution of the iteration was unsuccessful. The problem is solved with additional solver options.")
                optimizer.options["OF_start_with_resto"] = 'yes'
                solver_results = optimizer.solve(m, tee=tee, symbolic_solver_labels=True)
                self.termination_condition = solver_results.solver.termination_condition
                if self.termination_condition!= TerminationCondition.optimal:
                    print(
                        "WARNING: The solution of the iteration was unsuccessful. The problem is solved with additional solver options.")
                    optimizer.options["OF_start_with_resto"] = 'no'
                    optimizer.options["bound_push"] = 1E-02
                    optimizer.options["OF_bound_relax_factor"] = 1E-05
                    solver_results = optimizer.solve(m, tee=tee, symbolic_solver_labels=True)
                    self.termination_condition = solver_results.solver.termination_condition
                    # options["OF_bound_relax_factor"] = 1E-08
                    if self.termination_condition!= TerminationCondition.optimal:
                        raise Exception("The current iteration was unsuccessful.")
        m.del_component('objective')

    def set_up_reduced_hessian(self, time_set, component_set, var_name, index):
        
        print(f'index in: {index}')
        
        for t in time_set:
            for c in component_set:
                v = getattr(self.model, var_name)[t, c]
                self._idx_to_variable[index] = v
                self.model.red_hessian[v] = index
                index += 1
                print(index)
   
        return index
                    
    def _define_reduce_hess_order(self):
        """
        This sets up the suffixes of the reduced hessian
        """
        self.model.red_hessian = Suffix(direction=Suffix.IMPORT_EXPORT)
        count_vars = 1

        if self._spectra_given:
            if self.model_variance:
                count_vars = self.set_up_reduced_hessian(self._meas_times, self.component_set, self.component_var, count_vars)

        if self._spectra_given:
            if self.model_variance:
                count_vars = self.set_up_reduced_hessian(self._meas_lambdas, self.component_set, 'S', count_vars)
                
        for v in self.model.P.values():
            if v.is_fixed():
                print(v, end='\t')
                print("is fixed")
                continue
            self._idx_to_variable[count_vars] = v
            self.model.red_hessian[v] = count_vars
            count_vars += 1
            
        if hasattr(self.model, 'Pinit'):
            for k, v in self.model.Pinit.items():
                v = self.model.init_conditions[k]
                self._idx_to_variable[count_vars] = v
                self.model.red_hessian[v] = count_vars
                count_vars += 1
               
        return None
        
    def _confidence_interval_display(self, variances):
        """
        Function to display calculated confidence intervals
        """
        print('\nConfidence intervals:')
        i = 0
        for k, p in self.model.P.items():
            if p.is_fixed():
                continue
            print('{} ({},{})'.format(k, p.value - variances[i] ** 0.5, p.value + variances[i] ** 0.5))
            i += 1
        if hasattr(self.model, 'Pinit'): 
            for k in self.model.Pinit.keys():
                self.model.Pinit[k] = self.model.init_conditions[k].value
                print('{} ({},{})'.format(k, self.model.Pinit[k].value - variances[i] ** 0.5, self.model.Pinit[k].value + variances[i] ** 0.5))
                i += 1
        
        return None
    
    def _get_nparams(self, isSkipFixed=True):
        """Returns the number of unfixed parameters"""
        
        nparams = 0
        for v in self.model.P.values():
            if v.is_fixed():
                print(str(v) + '\has been skipped for covariance calculations')
                continue
            nparams += 1
        if hasattr(self.model, 'Pinit'):
            for v in self.model.Pinit.values():
                if isSkipFixed:
                    if v.is_fixed():
                        print(str(v) + '\has been skipped for covariance calculations')
                        continue
                nparams += 1

        return nparams

    def _compute_covariance(self, hessian, variances):

        nt = self._n_allmeas_times
        nw = self._n_meas_lambdas
        
        if hasattr(self, '_abs_components'):
            isSkipFixed = True
        else:
            isSkipFixed = False

        nparams = self._get_nparams()
        nd = nw * nt
        ntheta = self.n_val * (nw + nt) + nparams

        print("Computing H matrix\n shape ({},{})".format(nparams, ntheta))
        all_H = hessian
        H = all_H[-nparams:, :]
 
        print("Computing B matrix\n shape ({},{})".format(ntheta, nd))
        self._compute_B_matrix(variances)
        B = self.B_matrix
        
        print("Computing Vd matrix\n shape ({},{})".format(nd, nd))
        self._compute_Vd_matrix(variances)
        Vd = self.Vd_matrix
        
        R = B.T.dot(H.T)
        A = Vd.dot(R)
        L = H.dot(B)
        Vtheta = A.T.dot(L.T)
        V_theta = Vtheta.T

        if self._eigredhess2file==True:
            save_eig_red_hess(V_theta)

        nt = self._n_allmeas_times
        nw = self._n_meas_lambdas

        nd = nw * nt
        ntheta = self.n_val * (nw + nt)
        V_param = V_theta
        variances_p = np.diag(V_param)
        self._confidence_interval_display(variances_p)
        
        return None

    def _compute_covariance_C_generic(self, hessian, variances, use_model_variance=False):
        """
        Generic covariance function to reduce code
        """
        if use_model_variance:
            self._compute_residuals()
            res = self.residuals
            nc = self._n_actual
            varmat = np.zeros((nc, nc))
            for c, k in enumerate(self._sublist_components):
                varmat[c, c] = variances[k]
        
        nparams = self._get_nparams(isSkipFixed=False)
        
        all_H = hessian
        H = all_H[-nparams:, :]

        if not use_model_variance and self._eigredhess2file:
            save_eig_red_hess(H)

        covariance_C = H
        variances_p = np.diag(covariance_C)
        print("Parameter variances: ", variances_p)
        self._confidence_interval_display(variances_p)
       
        return 1

    def _compute_covariance_C(self, hessian, variances):
        """
        Computes the covariance matrix for the paramaters taking in the Hessian
        matrix and the variances for the problem where only C data is provided.
        Outputs the parameter confidence intervals.

        This function is not intended to be used by the users directly

        """
        self._compute_covariance_C_generic(hessian, variances, use_model_variance=True)
        return None

    def _compute_covariance_no_model_variance(self, hessian, variance):
        """
        Computes the covariance matrix for the paramaters taking in the Hessian
        matrix and the device variance for the problem where model error is ignored.
        Outputs the parameter confidence intervals.

        This function is not intended to be used by the users directly
        """
        self._compute_covariance_C_generic(hessian, variances, use_model_variance=False)
        return None
    
    def _compute_B_matrix(self, variances, **kwds):
        """Builds B matrix for calculation of covariances

           This method is not intended to be used by users directly

        Args:
            variances (dict): variances

        Returns:
            None
        """
        nt = self._n_meas_times
        time_set = self.model.meas_times
        conc_data_var = 'C'
        nw = self._n_meas_lambdas
        nparams = self._get_nparams()
       
        if hasattr(self, '_abs_components'):
            n_val = self._nabs_components
            component_set = self._abs_componets
        else:
            n_val = self._n_actual
            component_set = self._sublist_components
    
        variance = variances['device']
        ntheta = n_val * (nw + nt) + nparams
        self.B_matrix = np.zeros((ntheta, nw * nt))

        for i, t in enumerate(time_set):
            for j, l in enumerate(self.model.meas_lambdas):
                for k, c in enumerate(component_set):
                    r_idx1 = i * n_val + k
                    r_idx2 = j * n_val + k + n_val * nt
                    c_idx = i * nw + j
                    self.B_matrix[r_idx1, c_idx] = -2 * self.model.S[l, c].value / variance
                    self.B_matrix[r_idx2, c_idx] = -2 * getattr(self.model, conc_data_var)[t, c].value / variance
                    
        return None

    def _compute_Vd_matrix(self, variances, **kwds):
        """Builds d covariance matrix

           This method is not intended to be used by users directly

        Args:
            variances (dict): variances

        Returns:
            None
        """
        nt = self._n_meas_times
        nw = self._n_meas_lambdas
        row = []
        col = []
        data = []
        nd = nt * nw
        v_device = variances['device']

        s_array = np.zeros(nw * self.n_val)
        v_array = np.zeros(self.n_val)
        
        for k, c in enumerate(self.component_set):
            v_array[k] = variances[c]

        for j, l in enumerate(self.model.meas_lambdas):
            for k, c in enumerate(self.component_set):
                s_array[j * self.n_val + k] = self.model.S[l, c].value

        for i in range(nt):
            for j in range(nw):
                val = sum(v_array[k] * s_array[j * self.n_val + k] ** 2 for k in range(self.n_val)) + v_device
                row.append(i * nw + j)
                col.append(i * nw + j)
                data.append(val)

                for p in range(nw):
                    if j != p:
                        val = sum(v_array[k] * s_array[j * self.n_val + k] * s_array[p * self.n_val + k] for k in range(self.n_val))
                        row.append(i * nw + j)
                        col.append(i * nw + p)
                        data.append(val)

        self.Vd_matrix = scipy.sparse.coo_matrix((data, (row, col)), shape=(nd, nd)).tocsr()
        return None
        
    def _compute_residuals(self):
        """
        Computes the square of residuals between the optimal solution (Z) and the concentration data (C)
        Note that this returns a matrix of time points X components and it has not been divided by sigma^2

        This method is not intended to be used by users directly
        """
        nt = self._n_allmeas_times
        nc = self._n_actual
        self.residuals = dict()
        for c in self._sublist_components:
            for t in self._allmeas_times:
                a = self.model.C[t, c].value
                b = self.model.Z[t, c].value
                r = ((a - b) ** 2)
                self.residuals[t, c] = r
             
        return None

    def _order_k_aug_hessian(self, unordered_hessian, var_loc):
        """
        not meant to be used directly by users. Takes in the inverse of the reduced hessian
        outputted by k_aug and uses the rh_name to find the locations of the variables and then
        re-orders the hessian to be in a format where the other functions are able to compute the
        confidence intervals in a way similar to that utilized by sIpopt.
        """
        vlocsize = len(var_loc)
        n_vars = len(self._idx_to_variable)
        hessian = np.zeros((n_vars, n_vars))
        
        # why not enumerate? C++ guy?
        
        for i, vi in enumerate(self._idx_to_variable.values()):
            for j, vj in enumerate(self._idx_to_variable.values()):
                if n_vars == 1:
                    h = unordered_hessian
                    hessian[i, j] = h
                else:
                    h = unordered_hessian[(var_loc[vi]), (var_loc[vj])]
                    hessian[i, j] = h
        print(hessian.size, "hessian size")
        return hessian

    def calc_new_D(self, subset):
        
        new_D = pd.DataFrame(np.nan, index=self._meas_times, columns=subset)
        for t in self._meas_times:
            for l in self._meas_lambdas:
                if l in subset:
                    new_D.at[t, l] = self.model.D[t, l]
                    
        return new_D
        

    def run_param_est_with_subset_lambdas(self, builder_clone, end_time, subset, nfe, ncp, sigmas, solver='ipopt', ):
        """ Performs the parameter estimation with a specific subset of wavelengths.
            At the moment, this is performed as a totally new Pyomo model, based on the
            original estimation. Initialization strategies for this will be needed.

                Args:
                    builder_clone (TemplateBuidler): Template builder class of complete model
                                without the data added yet
                    end_time (float): the end time for the data and simulation
                    subset(list): list of selected wavelengths
                    nfe (int): number of finite elements
                    ncp (int): number of collocation points
                    sigmas(dict): dictionary containing the variances, as used in the ParameterEstimator class

                Returns:
                    results (Pyomo model solved): The solved pyomo model

        """
        if not isinstance(subset, (list, dict)):
            raise RuntimeError("subset must be of type list or dict!")

        if isinstance(subset, dict):
            lists1 = sorted(subset.items())
            x1, y1 = zip(*lists1)
            subset = list(x1)
        
        # This is the filter for creating the new data subset
        new_D = self.calc_new_D(subset)
        
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

                Args:
                    builder_before_data (TemplateBuilder): Template builder class of complete model
                                without the data added yet
                    end_time (int): the end time for the data and simulation

                    correlations (dict): dictionary containing the wavelengths and their correlations
                                to the concentration profiles
                    lof_full_model(int): the value of the lack of fit of the full model (with all wavelengths)

                Returns:
                    *****final model results.

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

            new_D = self.calc_new_D(new_subs)
                        

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
            print("When wavelengths of less than ", x[0], "correlation are removed")
            print("The lack of fit is: ", x[1])


    # =============================================================================
    # --------------------------- DIAGNOSTIC TOOLS ------------------------
    # =============================================================================

    def lack_of_fit(self):
        """ Runs basic post-processing lack of fit analysis

            Args:
                None

            Returns:
                lack of fit (int): percentage lack of fit

        """
        nt = self._n_meas_times
        nw = self._n_meas_lambdas
        sum_e = 0
        sum_d = 0
        C = np.zeros((nt, self.n_val))
        S = np.zeros((nw, self.n_val))
        
        for c_count, c in enumerate(self.component_set):
            for t_count, t in enumerate(self._meas_times):
                C[t_count, c_count] = getattr(self.model, self.component_var)[t, c].value

        for c_count, c in enumerate(self.component_set):
            for l_count, l in enumerate(self._meas_lambdas):
                S[l_count, c_count] = self.model.S[l, c].value
             
        D_model = C.dot(S.T)
        
        for t_count, t in enumerate(self._meas_times):
            for l_count, l in enumerate(self._meas_lambdas):
                sum_e += (D_model[t_count, l_count] - self.model.D[t, l]) ** 2
                sum_d += (self.model.D[t, l]) ** 2
  
        lof = np.sqrt(sum_e/sum_d)*100

        print("The lack of fit is ", lof, " %")
        return lof

    def wavelength_correlation(self):
        """ determines the degree of correlation between the individual wavelengths and
        the and the concentrations.

            Args:
                None

            Returns:
                dictionary of correlations with wavelength

        """
        nt = self._n_meas_times

        cov_d_l = dict()
        for c in self._sublist_components:
            for l in self._meas_lambdas:
                mean_d = (sum(self.model.D[t, l] for t in self._meas_times) / nt)
                mean_c = (sum(self.model.C[t, c].value for t in self._meas_times) / nt)
                cov_d_l[l, c] = 0
                for t in self._meas_times:
                    cov_d_l[l, c] += (self.model.D[t, l] - mean_d) * (self.model.C[t, c].value - mean_c)

                cov_d_l[l, c] = cov_d_l[l, c] / (nt - 1)

        # calculating the standard devs for dl and ck over time
        s_dl = dict()

        for l in self._meas_lambdas:
            s_dl[l] = 0
            mean_d = (sum(self.model.D[t, l] for t in self._meas_times) / nt)
            error = 0
            for t in self._meas_times:
                error += (self.model.D[t, l] - mean_d) ** 2
            s_dl[l] = (error / (nt - 1)) ** 0.5

        s_ck = dict()

        for c in self._sublist_components:
            s_ck[c] = 0
            mean_c = (sum(self.model.C[t, c].value for t in self._meas_times) / nt)
            error = 0
            for t in self._meas_times:
                error += (self.model.C[t, c].value - mean_c) ** 2
            s_ck[c] = (error / (nt - 1)) ** 0.5

        cor_lc = dict()

        for c in self._sublist_components:
            for l in self._meas_lambdas:
                cor_lc[l, c] = cov_d_l[l, c] / (s_ck[c] * s_dl[l])

        cor_l = dict()
        for l in self._meas_lambdas:
            cor_l[l] = max(cor_lc[l, c] for c in self._sublist_components)

        return cor_l

    #To estimate huplc fit in a similar fashion as for IR or concentration data (CS):
       
    # def lack_of_fit_huplc(self):
    #     """ Runs basic post-processing lack of fit analysis

    #         Args:
    #             None

    #         Returns:
    #             lack of fit (int): percentage lack of fit

    #     """
    #     nt = self._n_huplcmeas_times
    #     nc = self._n_huplc
    #     sum_e = 0
    #     sum_d = 0
    #     D_model = np.zeros((nt, nc))
    #     C = np.zeros((nt, nc))
        
    #     for t_count, t in enumerate(self._huplcmeas_times):
    #         for l_count, l in enumerate(self._list_huplcabs):
                
    #             if hasattr(self.model, 'solidvol'):
    #                 D_model[t_count, l_count] = (self.model.Z[t, l].value + self.model.solidvol[t, l].value) / (
    #                     sum(self.model.Z[t, j].value + self.model.solidvol[t, j].value for j in self._list_huplcabs))
    #             else:
    #                 D_model[t_count, l_count] = (self.model.Z[t, l].value) / (
    #                     sum(self.model.Z[t, j].value for j in self._list_huplcabs))

    #             sum_e += (D_model[t_count, l_count] - self.model.Dhat[t, l].value) ** 2
    #             sum_d += (self.model.Dhat[t, l].value) ** 2

    #     lof = np.sqrt((sum_e/sum_d))*100

    #     print("The lack of fit for the huplc data is ", lof, " %")
    #     return lof

    def lack_of_fit_huplc(self):
        """ Runs basic post-processing lack of fit analysis
            Args:
                None
            Returns:
                lack of fit (int): percentage lack of fit
        """
        nt = self._n_huplcmeas_times
        nc = self._n_huplc

        D_model = np.zeros((nt, nc))

        C = np.zeros((nt, nc))
        t_count = -1
        for t in self._huplcmeas_times:
            t_count += 1
            l_count = 0
            if hasattr(self.model, 'solidvol'):
                for l in self._list_huplcabs:
                    D_model[t_count, l_count] = (self.model.Z[t, l].value + self.model.solidvol[t, l].value) / (
                        sum(self.model.Z[t, j].value + self.model.solidvol[t, j].value for j in self._list_huplcabs))
                    l_count += 1
            else:
                for l in self._list_huplcabs:
                    D_model[t_count, l_count] = (self.model.Z[t, l].value) / (
                        sum(self.model.Z[t, j].value for j in self._list_huplcabs))
                    l_count += 1

        sum_e = 0
        sum_d = 0
        t_count = -1
        l_count = 0

        for t in self._huplcmeas_times:
            t_count += 1
            l_count = 0
            for l in self._list_huplcabs:
                sum_e += (D_model[t_count, l_count] - self.model.Dhat[t, l].value) ** 2
                l_count += 1

        t_count = -1
        l_count = 0
        for t in self._huplcmeas_times:
            for l in self._list_huplcabs:
                sum_d += (self.model.Dhat[t, l].value) ** 2

        lof = ((sum_e / sum_d) ** 0.5) * 100

        print("The lack of fit for the huplc data is ", lof, " %")
        return lof

def wavelength_subset_selection(correlations=None, n=None):
    """ identifies the subset of wavelengths that needs to be chosen, based
    on the minimum correlation value set by the user (or from the automated
    lack of fit minimization procedure)

        Args:
            correlations (dict): dictionary obtained from the wavelength_correlation
                    function, containing every wavelength from the original set and
                    their correlations to the concentration profile.

            n (int): a value between 0 - 1 that slects the minimum amount
                    correlation between the wavelength and the concentrations.

        Returns:
            dictionary of correlations with wavelength

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

        Args:
            builder_clone (TemplateBuilder): Template builder class of complete model
                            without the data added yet
            end_time (int): the end time for the data and simulation
            D (dataframe): the new, reduced dataset with only the selected wavelengths.

        Returns:
            opt_mode(TemplateBuilder): new Pyomo model from TemplateBuilder, ready for
                    parameter estimation

    """

    if not isinstance(builder_clone, TemplateBuilder):
        raise RuntimeError('builder_clone needs to be of type TemplateBuilder')

    if not isinstance(D, pd.DataFrame):
        raise RuntimeError('Spectral data format not supported. Try pandas.DataFrame')

    if not isinstance(end_time, int):
        raise RuntimeError('nfe needs to be type int. Number of finite elements must be defined')

    builder_clone.add_spectral_data(D)
    opt_model = builder_clone.create_pyomo_model(0.0, end_time)

    return opt_model


def run_param_est(opt_model, nfe, ncp, sigmas, solver='ipopt'):
    """ Runs the parameter estimator for the selected subset

        Args:
            opt_model (pyomo model): The model that we wish to run the
            nfe (int): number of finite elements
            ncp (int): number of collocation points
            sigmas(dict): dictionary containing the variances, as used in the ParameterEstimator class

        Returns:
            results_pyomo (results of optimization): Parameter Estimation results
            lof (float): lack of fit results

    """

    p_estimator = ParameterEstimator(opt_model)
    p_estimator.apply_discretization('dae.collocation', nfe=nfe, ncp=ncp, scheme='LAGRANGE-RADAU')
    options = dict()

    # These may not always solve, so we need to come up with a decent initialization strategy here
    if solver != 'ipopt':
        convariance = True
        
    results_pyomo = p_estimator.run_opt('ipopt',
                                            tee=False,
                                            solver_opts=options,
                                            covariance=covariance,
                                            )
    
    lof = p_estimator.lack_of_fit()

    return results_pyomo, lof
