# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from pyomo.environ import *
from pyomo.dae import *
from kipet.library.ParameterEstimator import *
from kipet.library.VarianceEstimator import *
from kipet.library.Optimizer import *
from pyomo import *
import matplotlib.pyplot as plt
import numpy as np
import scipy
import six
import copy
import re
import os

__author__ = 'Michael Short'  #: February 2019

class MultipleExperimentsEstimator():
    """This class is for Estimation of Variances and parameters when we have multiple experimental datasets.
    This class relies heavily on the Pyomo block class as we put each experimental class into its own block.
    This blocks are first run individually in order to find good initializations and then they are linked and
    run together in a large optimization problem.

    Parameters
    ----------
    model : TemplateBuilder
        The full model TemplateBuilder problem needs to be fed into the MultipleExperimentsEstimator. This
        pyomo model will form the basis for all the optimization tasks
    """

    def __init__(self,datasets):
        super(MultipleExperimentsEstimator, self).__init__()
        self.block_models = dict()
        
        if datasets != None:
            if isinstance(datasets, dict):
                for key, val in datasets.items():
                    if not isinstance(key, str):
                        raise RuntimeError('The key for the dictionary must be a str')
                    if not isinstance(val, pd.DataFrame):
                        raise RuntimeError('The value in the dictionary must be the experimental dataset as a pandas DataFrame')

        else:
            raise RuntimeError("datasets not given, add datasets as a dict to use this class")
            
        self.datasets = datasets
        self.experiments = list()
        for key,val in self.datasets.items():
            self.experiments.append(key)
        
        self._variance_solved = False
        
        self.variances= dict()
        self.variance_results = dict()
        self.start_time =dict()
        self.end_time = dict()
        self.builder = dict()
        self.opt_model = dict()
        
        self.initialization_model = dict()


    def build_individual_blocks(self, m, exp):
        """This function forms the rule for the construction of the individual blocks 
        for multiple experiments, referenced in run_parameter_estimation. This function 
        is not meant to be used by users directly.
        
        Args:
            m (pyomo Concrete model): the concrete model that we are adding the block to
            exp (list): a list containing the experiments
            
        Returns:
            Pyomo model: Pyomo model inside the block
            
        """
        #WITH_D_VARS
        with_d_vars= True
        
        m = self.initialization_model[exp]
        if with_d_vars:
            m.D_bar = Var(m.meas_times,
                          m.meas_lambdas)

            def rule_D_bar(m, t, l):
                return m.D_bar[t, l] == sum(m.C[t, k] * m.S[l, k] for k in m._sublist_components)

            m.D_bar_constraint = Constraint(m.meas_times,
                                            m.meas_lambdas,
                                            rule=rule_D_bar)
        
        m.error = Var(bounds = (0, None))

        def rule_objective(m):
            expr = 0
            for t in m.meas_times:
                for l in m.meas_lambdas:
                    if with_d_vars:
                        expr += (m.D[t, l] - m.D_bar[t, l]) ** 2 / (self.variances[exp]['device'])
                    else:
                        D_bar = sum(m.C[t, k] * m.S[l, k] for k in list_components)
                        expr += (m.D[t, l] - D_bar) ** 2 / (self.variances[exp]['device'])

            expr *= weights[0]
            second_term = 0.0
            for t in m.meas_times:
                second_term += sum((m.C[t, k] - m.Z[t, k]) ** 2 / self.variances[exp][k] for k in list_components)

            expr += weights[1] * second_term
            return m.error == expr

        m.obj_const = Constraint(rule=rule_objective)
        
        return m
            
                    
    def run_variance_estimation(self, builder, **kwds):
        """Solves the Variance Estimation procedure described in Chen et al 2016. Here, we call the VarianceEstimator
            seperately on each dataset in order to not only get the model noise and the 
           This method solved a sequence of optimization problems
           to determine variances and initial guesses for parameter estimation.

        Args:
            solver (str,optional): solver to be used, default is ipopt
            
            solver_opts (dict, optional): options passed to the nonlinear solver
            
            start_time (dict): dictionary with key of experiment name and value float
            
            end_time (dict): dictionary with key of experiment name and value float
            
            nfe (int): number of finite elements
            
            ncp (int): number of collocation points
            
            builder (TemplateBuilder): need to add the TemplateBuilder before create_pyomo_model

            tee (bool,optional): flag to tell the optimizer whether to stream output
            to the terminal or not

            norm (optional): norm for checking convergence. The default value is the infinity norm,
            it uses same options as scipy.linalg.norm

            max_iter (int,optional): maximum number of iterations for Weifengs procedure. Default 400.

            tolerance (float,optional): Tolerance for termination by the change Z. Default 5.0e-5

            subset_lambdas (array_like,optional): Set of wavelengths to used in initialization problem 
            (Weifeng paper). Default all wavelengths.

            lsq_ipopt (bool,optional): Determines whether to use ipopt for solving the least squares 
            problems in Weifengs procedure. Default False. The default used scipy.least_squares.

            init_C (DataFrame,optional): Dataframe with concentration data used to start Weifengs procedure.

        Returns:

            None

        """
        #Require the same arguments as the VarianceEstimator as these will be applied in the same
        #function, just for each individual dataset
        solver = kwds.pop('solver',str)
        solver_opts = kwds.pop('solver_opts', dict())
        tee = kwds.pop('tee', False)
        norm_order = kwds.pop('norm',np.inf)
        max_iter = kwds.pop('max_iter', 400)
        tol = kwds.pop('tolerance', 5.0e-5)
        A = kwds.pop('subset_lambdas', None)
        lsq_ipopt = kwds.pop('lsq_ipopt', False)
        init_C = kwds.pop('init_C', None)
        
        start_time = kwds.pop('start_time', dict())
        end_time = kwds.pop('end_time', dict())
        nfe = kwds.pop('nfe', 50)
        ncp = kwds.pop('ncp', 3)
        
        # additional arguments for inputs
        # These will need to be made into dicts if there are different 
        # conditions in each experiment
        inputs = kwds.pop("inputs", None)
        inputs_sub = kwds.pop("inputs_sub", None)
        trajectories = kwds.pop("trajectories", None)
        fixedtraj = kwds.pop('fixedtraj', False)
        fixedy = kwds.pop('fixedy', False)
        yfix = kwds.pop("yfix", None)
        yfixtraj = kwds.pop("yfixtraj", None)

        jump = kwds.pop("jump", False)
        var_dic = kwds.pop("jump_states", None)
        jump_times = kwds.pop("jump_times", None)
        feed_times = kwds.pop("feed_times", None)

        species_list = kwds.pop('subset_components', None)
        
        if not isinstance(start_time, dict):
            raise RuntimeError("Must provide start_times as dict with each experiment")
        else:
            for key, val in start_time.items():
                if key not in self.experiments:
                    raise RuntimeError("The keys for start_time need to be the same as the experimental datasets")
        self.start_time = start_time    
        if not isinstance(end_time, dict):
            raise RuntimeError("Must provide end_times as dict with each experiment")    
        else:
            for key, val in end_time.items():
                if key not in self.experiments:
                    raise RuntimeError("The keys for end_time need to be the same as the experimental datasets")
        self.end_time = end_time 
        #This will need to change if we have different models for each experiment
        if not isinstance(builder, TemplateBuilder):
            raise RuntimeError('builder needs to be type TemplateBuilder')
        
        if solver == '':
            solver = 'ipopt'
            
        v_est_dict = dict()
        results_variances = dict()
        sigmas = dict()
        print("SOLVING VARIANCE ESTIMATION FOR INDIVIDUAL DATASETS")
        for l in self.experiments:
            print("solving for dataset ", l)
            self.builder[l] = builder
            self.builder[l].add_spectral_data(self.datasets[l])
            self.opt_model[l] = self.builder[l].create_pyomo_model(start_time[l],end_time[l])
            
            v_est_dict[l] = VarianceEstimator(self.opt_model[l])
            v_est_dict[l].apply_discretization('dae.collocation',nfe=nfe,ncp=ncp,scheme='LAGRANGE-RADAU')
            results_variances[l] = v_est_dict[l].run_opt(solver,
                                            tee=tee,
                                            solver_options=solver_opts,
                                            max_iter=max_iter,
                                            tol=tol)
            print("\nThe estimated variances are:\n")
            for k,v in six.iteritems(results_variances[l].sigma_sq):
                print(k, v)
            self.variance_results[l] = results_variances[l]
            # and the sigmas for the parameter estimation step are now known and fixed
            sigmas[l] = results_variances[l].sigma_sq
            self.variances[l] = sigmas[l] 
        self._variance_solved = True
        
        return results_variances
            
    def run_parameter_estimation(self, builder, **kwds):
        """Solves the Parameter Estimation procedure described in Chen et al 2016. Here, 
            we call the ParameterEstimator separately on each dataset before adding them
            to blocks and solving simultaneously.

        Args:
            solver (str): name of the nonlinear solver to used

            solver_opts (dict, optional): options passed to the nonlinear solver

            variances (dict, optional): map of component name to noise variance. The
            map also contains the device noise variance

            tee (bool,optional): flag to tell the optimizer whether to stream output
            to the terminal or not

            with_d_vars (bool,optional): flag to the optimizer whether to add
            variables and constraints for D_bar(i,j)
            
            start_time (dict): dictionary with key of experiment name and value float
            
            end_time (dict): dictionary with key of experiment name and value float
            
            nfe (int): number of finite elements
            
            ncp (int): number of collocation points
            
            builder (TemplateBuilder): need to add the TemplateBuilder before create_pyomo_model

            subset_lambdas (array_like,optional): Set of wavelengths to used in initialization problem 
            (Weifeng paper). Default all wavelengths.
            
            sigma_sq (dict): variances

        Returns:

            None

        """
        #Require the same arguments as the VarianceEstimator as these will be applied in the same
        #function, just for each individual dataset
        solver = kwds.pop('solver', str)
        solver_opts = kwds.pop('solver_opts', dict())
        tee = kwds.pop('tee', False)
        A = kwds.pop('subset_lambdas', None)
        
        start_time = kwds.pop('start_time', dict())
        end_time = kwds.pop('end_time', dict())
        nfe = kwds.pop('nfe', 50)
        ncp = kwds.pop('ncp', 3)
        
        sigma_sq = kwds.pop('sigma_sq', dict())
        # additional arguments for inputs
        # These will need to be made into dicts if there are different 
        # conditions in each experiment
        inputs = kwds.pop("inputs", None)
        inputs_sub = kwds.pop("inputs_sub", None)
        trajectories = kwds.pop("trajectories", None)
        fixedtraj = kwds.pop('fixedtraj', False)
        fixedy = kwds.pop('fixedy', False)
        yfix = kwds.pop("yfix", None)
        yfixtraj = kwds.pop("yfixtraj", None)

        jump = kwds.pop("jump", False)
        var_dic = kwds.pop("jump_states", None)
        jump_times = kwds.pop("jump_times", None)
        feed_times = kwds.pop("feed_times", None)

        species_list = kwds.pop('subset_components', None)
        
        if not isinstance(start_time, dict):
            raise RuntimeError("Must provide start_times as dict with each experiment")
        else:
            for key, val in start_time.items():
                if key not in self.experiments:
                    raise RuntimeError("The keys for start_time need to be the same as the experimental datasets")
            
        if not isinstance(end_time, dict):
            raise RuntimeError("Must provide end_times as dict with each experiment")    
        else:
            for key, val in end_time.items():
                if key not in self.experiments:
                    raise RuntimeError("The keys for end_time need to be the same as the experimental datasets")
         
        #This will need to change if we have different models for each experiment
        if not isinstance(builder, TemplateBuilder):
            raise RuntimeError('builder needs to be type TemplateBuilder')
        
        p_est_dict = dict()
        results_pest = dict()
        
        if bool(sigma_sq) == False:
            sigma_sq = self.variances
        else:
            all_sigma_specified = True
            keys = sigma_sq.keys()
            for k in list_components:
                if k not in keys:
                    all_sigma_specified = False
                    sigma_sq[k] = max(sigma_sq.values())
    
            if not 'device' in sigma_sq.keys():
                all_sigma_specified = False
                sigma_sq['device'] = 1.0

        print("SOLVING PARAMETER ESTIMATION FOR INDIVIDUAL DATASETS - For initialization")
        
        ind_p_est = dict()
        list_params_across_blocks = list()
        for l in self.experiments:
            print("solving for dataset ", l)
            if self._variance_solved == True:
                #then we already have inits
                ind_p_est[l] = ParameterEstimator(self.opt_model[l])
                #ind_p_est[l].apply_discretization('dae.collocation',nfe=nfe,ncp=ncp,scheme='LAGRANGE-RADAU')
                               
                ind_p_est[l].initialize_from_trajectory('Z',self.variance_results[l].Z)
                ind_p_est[l].initialize_from_trajectory('S',self.variance_results[l].S)
                ind_p_est[l].initialize_from_trajectory('C',self.variance_results[l].C)
            
                ind_p_est[l].scale_variables_from_trajectory('Z',self.variance_results[l].Z)
                ind_p_est[l].scale_variables_from_trajectory('S',self.variance_results[l].S)
                ind_p_est[l].scale_variables_from_trajectory('C',self.variance_results[l].C)
                
                results_pest[l] = ind_p_est[l].run_opt('ipopt',
                                                     tee=tee,
                                                      solver_opts = solver_opts,
                                                      variances = self.variances[l])
                
                self.initialization_model[l] = ind_p_est[l]
                print("The estimated parameters are:")
                for k,v in six.iteritems(results_pest[l].P):
                    print(k, v)
                    if k not in list_params_across_blocks:
                        list_params_across_blocks.append(k)

            else:
                #we do not have inits
                self.builder[l]=builder
                self.builder[l].add_spectral_data(self.datasets[l])
                self.opt_model[l] = self.builder[l].create_pyomo_model(start_time[l],end_time[l])
                ind_p_est[l] = ParameterEstimator(self.opt_model[l])
                #ind_p_est[l].apply_discretization('dae.collocation',nfe=nfe,ncp=ncp,scheme='LAGRANGE-RADAU')
                
                results_pest[l] = ind_p_est[l].run_opt('ipopt',
                                                     tee=tee,
                                                      solver_opts = solver_opts,
                                                      variances = self.variances)

                self.initialization_model[l] = ind_p_est[l]
                print("The estimated parameters are:")
                for k,v in six.iteritems(results_pest[l].P):
                    print(k, v)
                # I want to build a list of the parameters here that we will use to link the models later on
                    
        #Now that we have all our datasets solved individually we can build our blocks and use
        #these solutions to initialize
        m = ConcreteModel()
        m.solver_opts = solver_opts
        m.tee = tee
        
        def build_individual_blocks(m, exp):
            """This function forms the rule for the construction of the individual blocks 
            for multiple experiments, referenced in run_parameter_estimation. This function 
            is not meant to be used by users directly.
            
            Args:
                m (pyomo Concrete model): the concrete model that we are adding the block to
                exp (list): a list containing the experiments
                
            Returns:
                Pyomo model: Pyomo model inside the block
                
            """
            species_list=None
            #if not set_A:
            #    set_A = before_solve_extend._meas_lambdas
            
            list_components = []
            if species_list is None:
                list_components = [k for k in self.initialization_model[exp]._mixture_components]
            else:
                for k in species_list:
                    if k in self.initialization_model[exp]._mixture_components:
                        list_components.append(k)
                    else:
                        warnings.warn("Ignored {} since is not a mixture component of the model".format(k))
                        
            #WITH_D_VARS as far as I can tell should always be True, unless we ignore the device noise
            with_d_vars= True
            
            m = self.initialization_model[exp].model
            if with_d_vars:
                m.D_bar = Var(m.meas_times,
                              m.meas_lambdas)
    
                def rule_D_bar(m, t, l):
                    return m.D_bar[t, l] == sum(m.C[t, k] * m.S[l, k] for k in self.initialization_model[exp]._sublist_components)
    
                m.D_bar_constraint = Constraint(m.meas_times,
                                                m.meas_lambdas,
                                                rule=rule_D_bar)
                
            m.error = Var(bounds = (0, None))

            def rule_objective(m):
                expr = 0
                for t in m.meas_times:
                    for l in m.meas_lambdas:
                        if with_d_vars:
                            expr += (m.D[t, l] - m.D_bar[t, l]) ** 2 / (self.variances[exp]['device'])
                        else:
                            D_bar = sum(m.C[t, k] * m.S[l, k] for k in list_components)
                            expr += (m.D[t, l] - D_bar) ** 2 / (self.variances[exp]['device'])
                
                #If we require weights then we would add them back in here
                #expr *= weights[0]
                second_term = 0.0
                for t in m.meas_times:
                    second_term += sum((m.C[t, k] - m.Z[t, k]) ** 2 / self.variances[exp][k] for k in list_components)
    
                #expr += weights[1] * second_term
                expr += second_term
                return m.error == expr
    
            m.obj_const = Constraint(rule=rule_objective)
            
            return m  
        
        m.experiment = Block(self.experiments, rule = build_individual_blocks)
        count = 0
        m.dataset_count = list()
        m.map_exp_to_count = dict()
        m.first_exp = None
        for i in self.experiments:
            m.dataset_count.append(count)
            m.map_exp_to_count[count] = i
            if count == 0:
                m.first_exp = i
            count+=1
  
        def param_linking_rule(m, exp, param):
            if exp == m.first_exp:
                return Constraint.Skip
            else:
                for key, val in m.map_exp_to_count.items():
                    if val == exp:
                        prev_exp = m.map_exp_to_count[key-1]
                        
                return m.experiment[exp].P[param] == m.experiment[prev_exp].P[param]
            
        # For the number of parameters we need a constraint for each one that links
        # not sure if this index is the best way to go... Should be self.experiments
        m.parameter_linking = Constraint(self.experiments, list_params_across_blocks, rule = param_linking_rule)
        
        m.obj = Objective(sense = minimize, expr=sum(b.error for b in m.experiment[:]))
        
        optimizer = SolverFactory(solver)  
        solver_results = optimizer.solve(m, options = solver_opts,tee=tee) 
        
        #for i in m.experiment:
        #    m.experiment[i].P.pprint()
            
        solver_results = dict()   
        
        # loading the results, notice that we return a dictionary
        for i in m.experiment:
            solver_results[i] = ResultsObject()
            solver_results[i].load_from_pyomo_model(m.experiment[i],to_load=['Z', 'dZdt', 'X', 'dXdt', 'C', 'S', 'Y', 'P'])
        
        return solver_results