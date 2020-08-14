import copy
import math
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyomo import *
from pyomo.dae import *
from pyomo.environ import *
from pyomo.opt import ProblemFormat

from scipy.sparse import coo_matrix

from kipet.library.common.objectives import conc_objective
from kipet.library.common.pe_methods import PEMixins
from kipet.library.common.read_hessian import split_sipopt_string

from kipet.library.fe_factory import *
from kipet.library.FESimulator import *
from kipet.library.Optimizer import *
from kipet.library.ParameterEstimator import *
from kipet.library.PyomoSimulator import *
from kipet.library.VarianceEstimator import *

__author__ = 'Michael Short, Kevin McBride'  #: February 2019 - August 2020

class MultipleExperimentsEstimator(PEMixins, object):
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
        self._idx_to_variable = dict()
        
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
        #added for new initialization options (CS):
        self._sim_solved = False
        self.sim_results = dict()
        self.cloneopt_model = dict()
        # self.clonecloneopt_model = dict()
        
        self.variances= dict()
        self.variance_results = dict()
        self.start_time =dict()
        self.end_time = dict()
        self.builder = dict()
        self.opt_model = dict()
        
        self.initialization_model = dict()
        self._sublist_components = dict()

        self._n_meas_times = 0
        self._n_meas_lambdas = 0
        self._n_actual = 0
        self._n_params = 0
        
        self._spectra_given = True
        self._concentration_given = False
        
        self.global_params = None
        
        # set of flags to mark the how many times and wavelengths are in each dataset
        # TODO: make an object for the datasets to handle these things
        
        self.l_mark = dict()
        self.t_mark = dict()
        self.n_mark = dict()
        self.p_mark = dict()
 
          
    def _define_reduce_hess_order_mult(self):
        """This function is used to link the variables to the columns in the reduced
           hessian for multiple experiments.   
           
           Not meant to be used directly by users
        """
        self.model.red_hessian = Suffix(direction=Suffix.IMPORT_EXPORT)
        count_vars = 1
        model_obj = self.model.experiment

        for i in self.experiments:
            print(f'exp: {i}')
            if self._spectra_given:
                if hasattr(self, 'model_variance') and self.model_variance or not hasattr(self, 'model_variance'):
                    count_vars = self._set_up_reduced_hessian(model_obj[i], self.model.experiment[i].meas_times, self._sublist_components[i], 'C', count_vars)
        
        for i in self.experiments:
            if self._spectra_given:
                if hasattr(self, 'model_variance') and self.model_variance or not hasattr(self, 'model_variance'):       
                    count_vars = self._set_up_reduced_hessian(model_obj[i], self.model.experiment[i].meas_lambdas, self._sublist_components[i], 'S', count_vars)
                
        for i in self.experiments:
            for v in model_obj[i].P.values():
                if v.is_fixed():
                    print(v, end='\t')
                    print("is fixed")
                    continue
                self._idx_to_variable[count_vars] = v
                self.model.red_hessian[v] = count_vars
                count_vars += 1
            
            if hasattr(model_obj[i], 'Pinit'):
                for k, v in model_obj[i].Pinit.items():
                    v = model_obj[i].init_conditions[k]
                    self._idx_to_variable[count_vars] = v
                    self.model.red_hessian[v] = count_vars
                    count_vars += 1
                
        return None
           
    def _set_up_marks(self, conc_only=True):
        """Set up the data based on the number of species and measurements
        
        """    
        nt = np.cumsum(np.array([len(self.model.experiment[exp].meas_times) for i, exp in enumerate(self.experiments)]))
        self.t_mark = {i: n for i, n in enumerate(nt)}
        nt = nt[-1]
        self._n_meas_times = nt
        
        nc = np.cumsum(np.array([len(self._sublist_components[exp]) for i, exp in enumerate(self.experiments)]))
        self.n_mark = {i: n for i, n in enumerate(nc)}
        nc = nc[-1]
        self._n_actual = nc
        
        nparams = np.cumsum(np.array([self._get_nparams(self.model.experiment[exp]) for i, exp in enumerate(self.experiments)]))
        self.p_mark = {i: n for i, n in enumerate(nparams)}
        nparams = nparams[-1]
        self._n_params = nparams
        
        if not conc_only:
            
            nw = np.cumsum(np.array([len(self.model.experiment[exp].meas_lambdas) for i, exp in enumerate(self.experiments)]))
            self.l_mark = {i: n for i, n in enumerate(nw)}
            nw = nw[-1]
            self._n_meas_lambdas = nw
        
        return None
    
    def _display_covariance(self, variances_p):
        """Displays the covariance results to the console
        """
        
        print('\nParameters:')
        for exp in self.experiments:
            for k, p in self.model.experiment[exp].P.items():
                if p.is_fixed():
                    continue
                print('{}, {}'.format(k, p.value))
        print('\nConfidence intervals:')
        for exp in self.experiments:
            for i, (k, p) in enumerate(self.model.experiment[exp].P.items()):
                if p.is_fixed():
                    continue
                print('{} ({},{})'.format(k, p.value - variances_p[i] ** 0.5, p.value + variances_p[i] ** 0.5))
        
        # Does this work? Where does the i come from?
        if hasattr(self.model.experiment[exp], 'Pinit'):
            print('\nLocal Parameters:')
            for exp in self.experiments:
                for k in self.model.experiment[exp].Pinit.keys():
                    self.model.experiment[exp].Pinit[k] = self.model.experiment[exp].init_conditions[k].value
                    print('{}, {}'.format(k, self.model.experiment[exp].Pinit[k].value))
            print('\nConfidence intervals:')
            for exp in self.experiments:
                for i, k in enumerate(self.model.experiment[exp].Pinit.keys()):
                    self.model.experiment[exp].Pinit[k] = self.model.experiment[exp].init_conditions[k].value
                    print('{} ({},{})'.format(k, self.model.experiment[exp].Pinit[k].value - variances_p[i] ** 0.5, self.model.experiment[exp].Pinit[k].value + variances_p[i] ** 0.5))
                
        return None
        
    def _compute_covariance(self, hessian, variances):
        """
        Computes the covariance matrix for the paramaters taking in the Hessian
        matrix and the variances.
        Outputs the parameter confidence intervals.

        This function is not intended to be used by the users directly

        """
        self._set_up_marks(conc_only=False)
        variances_p, covariances_p = self._variances_p_calc(hessian, variances)        
        self._display_covariance(variances_p)
        
        return None
        
    def _compute_covariance_C(self, hessian, variances):
        """
        Computes the covariance matrix for the paramaters taking in the Hessian
        matrix and the variances for the problem where only C data is provided.
        Outputs the parameter confidence intervals.

        This function is not intended to be used by the users directly

        """
        self._set_up_marks()
    
        res = {}
        for exp in self.experiments:
            res.update(self._compute_residuals(self.model.experiment[exp], exp_index=exp))

        H = hessian[-self._n_params:, :]
        variances_p = np.diag(H)
        self._display_covariance(variances_p)
        
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
        nw = self._n_meas_lambdas
        nc = self._n_actual
        nparams = self._n_params

        npn = np.r_[self.n_mark[0], np.diff(list(self.n_mark.values()))]
        npt = np.r_[self.t_mark[0], np.diff(list(self.t_mark.values()))]
        npl = np.r_[self.l_mark[0], np.diff(list(self.l_mark.values()))]
        npp = np.r_[self.p_mark[0], np.diff(list(self.p_mark.values()))]
        exp_lookup = {i: exp for i, exp in enumerate(self.experiments)}
        ntheta = sum(npn*(npt + npl) + npp)
        
        exp_count = 0
        timeshift = 0 
        waveshift = 0
        
        rows = []
        cols = []
        data = []
        
        meas_times = {exp : {indx: time for indx, time in enumerate(self.model.experiment[exp].meas_times)} for exp in self.experiments}
        meas_lambdas = {exp : {indx: wave for indx, wave in enumerate(self.model.experiment[exp].meas_lambdas)} for exp in self.experiments}
        
        for i in range(nt):
            for j in range(nw):
            
                nc = npn[exp_count]
                if i == self.t_mark[exp_count] and j == self.l_mark[exp_count]:
                    exp_count += 1
                    timeshift = i
                    waveshift = j    
           
                exp = exp_lookup[exp_count]
                
                for comp_num, comp in enumerate(self._sublist_components[exp]):
                    
                    if i - timeshift in list(range(npt[exp_count])):
                        time = meas_times[exp][i - timeshift]
                    
                    if j - waveshift in list(range(npl[exp_count])):
                        wave = meas_lambdas[exp][j - waveshift]
   
                    r_idx1 = i*nc + comp_num
                    r_idx2 = j*nc + comp_num + nc*nt
                    c_idx =  i*nw + j
                    
                    rows.append(r_idx1)
                    cols.append(c_idx)
                    data.append(-2 * self.model.experiment[exp].S[wave, comp].value / (self.variances[exp]['device']))
  
                    rows.append(r_idx2)
                    cols.append(c_idx)
                    data.append(-2 * self.model.experiment[exp].C[time, comp].value / (self.variances[exp]['device']))
                         
        B_matrix = coo_matrix((data, (rows, cols)), shape=(ntheta, nw * nt)).tocsr()
        self.B_matrix = B_matrix
        
        return B_matrix
        
    def _compute_Vd_matrix(self, variances, **kwds):
        """Builds d covariance matrix

           This method is not intended to be used by users directly

        Args:
            variances (dict): variances

        Returns:
            None
        """
        row = []
        col = []
        data = []
        nt = self._n_meas_times
        nw = self._n_meas_lambdas
        nc = self._n_actual

        v_array = np.zeros(nc)
        s_array = np.zeros(nw * nc)
        
        count = 0
        for x in self.experiments:
            for k, c in enumerate(self._sublist_components[x]):
                v_array[count] = variances[x][c]
                count += 1
        
        kshift = 0
        jshift = 0
        knum = 0
        jnum=0
        count=0
        exp_count = 0
        
        for x in self.experiments:
            kshift += knum
            jshift += jnum
            if exp_count != 0:
                kshift+=1

            for j, l in enumerate(self.model.experiment[x].meas_lambdas):
                for k, c in enumerate(self._sublist_components[x]):
                    
                    if exp_count == 0:
                        nc = self.n_mark[exp_count]
                        nt = self.t_mark[exp_count]
                        nw = self.l_mark[exp_count]
                    else: 
                        nt = (self.t_mark[exp_count] - self.t_mark[exp_count - 1] )
                        nw = (self.l_mark[exp_count] - self.l_mark[exp_count - 1])

                    s_array[(j+jshift) * nc + (k+kshift)] = self.model.experiment[x].S[l, c].value
                    idx = (j+jshift) * nc + (k+kshift)
                    knum = max(knum,k)
                    count += 1
                
                jnum = max(jnum,j)
            exp_count += 1

        row = []
        col = []
        data = []
        nt = self._n_meas_times
        nw = self._n_meas_lambdas
        nd = nt * nw
        v_device = list()
        
        for x in self.experiments:
            v_device.append(variances[x]['device']) 
        
        exp_count = 0
        v_device_exp = v_device[exp_count]
        for i in range(nt):
            for j in range(nw):
                if i == self.t_mark[exp_count] and j == self.l_mark[exp_count]:
                    exp_count += 1
                    v_device_exp = v_device[exp_count]
                    
                val = sum(v_array[k] * s_array[j * nc + k] ** 2 for k in range(nc)) + v_device_exp
                row.append(i * nw + j)
                col.append(i * nw + j)
                data.append(val)
                for p in range(nw):
                    if j != p:
                        val = sum(v_array[k] * s_array[j * nc + k] * s_array[p * nc + k] for k in range(nc))
                        row.append(i * nw + j)
                        col.append(i * nw + p)
                        data.append(val)

        Vd_matrix = coo_matrix((data, (row, col)), shape=(nd, nd)).tocsr()
        
        self.Vd_matrix = Vd_matrix
    
        return Vd_matrix
  
    def run_simulation(self, builder, **kwds):
        """ Runs simulation by solving nonlinear system with ipopt

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

        Returns:
            None

        """
        solver = kwds.pop('solver', str)
        FEsim = kwds.pop('FEsim', False)
        solver_opts = kwds.pop('solver_opts', dict())
        sigmas = kwds.pop('variances', dict())
        tee = kwds.pop('tee', False)
        seed = kwds.pop('seed', None)

        start_time = kwds.pop('start_time', dict())
        end_time = kwds.pop('end_time', dict())
        nfe = kwds.pop('nfe', 50)
        ncp = kwds.pop('ncp', 3)

        # spectra_problem = kwds.pop('spectra_problem', True)

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

        # This is to make sure that for either cases of 1 model for all experiments or different models,
        # we ensure we have the correct builders

        if isinstance(builder, dict):
            for key, val in builder.items():
                if key not in self.experiments:
                    raise RuntimeError("The keys for builder need to be the same as the experimental datasets")

                if not isinstance(val, TemplateBuilder):
                    raise RuntimeError('builder needs to be type TemplateBuilder')

        elif isinstance(builder, TemplateBuilder):
            builder_dict = {}
            for item in self.experiments:
                builder_dict[item] = builder

            builder = builder_dict
        else:
            raise RuntimeError("builder added needs to be a dictionary of TemplateBuilders or a TemplateBuilder")

        if solver == '':
            solver = 'ipopt'

        print("SOLVING SIMULATION FOR INDIVIDUAL DATASETS")

        sim_dict = dict()
        results_sim = dict()
        ind_p_est = dict()

        for l in self.experiments:
            print("\nsolving for dataset ", l)
            self.builder[l] = builder[l]
            # if spectra_problem==True:
            #     self._spectra_given=True
            #     self.builder[l].add_spectral_data(self.datasets[l])
            # else:
            #     self._concentration_given=True
            #     self.builder[l].add_concentration_data(self.datasets[l])
            self.opt_model[l] = self.builder[l].create_pyomo_model(start_time[l], end_time[l])
            ind_p_est[l] = ParameterEstimator(self.opt_model[l])

            self.cloneopt_model[l] = self.opt_model[l].clone()

            #fix parameters for simulation:
            for k in self.cloneopt_model[l].P.keys():
                self.cloneopt_model[l].P[k].fixed=True

            if FEsim==True: #Initialize with FEsimulator
                sim_dict = dict()
                results_sim = dict()
                sim_dict[l] = FESimulator(self.cloneopt_model[l])
                # # # defines the discrete points wanted in the concentration profile
                sim_dict[l].apply_discretization('dae.collocation', nfe=nfe, ncp=ncp, scheme='LAGRANGE-RADAU')
                inputs_sub = {}
                initexp1 = sim_dict[l].call_fe_factory(inputs_sub)

                results_sim[l] = sim_dict[l].run_sim(solver,
                                      tee=tee,
                                      solver_opts=solver_opts)
            else:
                sim_dict = dict()
                results_sim = dict()
                sim_dict[l] = PyomoSimulator(self.cloneopt_model[l])
                sim_dict[l].apply_discretization('dae.collocation', nfe=nfe, ncp=ncp, scheme='LAGRANGE-RADAU')
                results_sim[l] = sim_dict[l].run_sim(solver,
                                                 tee=tee,
                                                 solver_opts=solver_opts)

            self.sim_results[l] = results_sim[l]

        self._sim_solved = True

        return results_sim
    #################################
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
        tol = kwds.pop('tol', 5.0e-5)
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
        method = kwds.pop("method", None)
        initsigs = kwds.pop("initial_sigmas", dict())
        tolerance = kwds.pop("tolerance", dict())
        secant_point = kwds.pop("secant_point",dict())

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

        #option for non-negativity bounds for Z:
        # self.lbZ = kwds.pop('lbZ', False)

        species_list = kwds.pop('subset_components', None)
        
        if method == 'alternate':
            if not isinstance(initsigs, dict):
                raise RuntimeError("Must provide initial sigmas as a dictionary of dictionaries with each experiment")
        else:
            for item in self.experiments:
                initsigs[item] = 1e-10
            for item in self.experiments:
                tolerance[item] = tol

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
        
        #This is to make sure that for either cases of 1 model for all experiments or different models, 
        #we ensure we have the correct builders
        
        if isinstance(builder, dict):
            for key, val in builder.items():
                if key not in self.experiments:
                    raise RuntimeError("The keys for builder need to be the same as the experimental datasets")   
                    
                if not isinstance(val, TemplateBuilder):
                    raise RuntimeError('builder needs to be type TemplateBuilder')
                    
        elif isinstance(builder, TemplateBuilder):
            builder_dict = {}
            for item in self.experiments:
                builder_dict[item] = builder
            
            builder = builder_dict
            
        else:
            raise RuntimeError("builder added needs to be a dictionary of TemplateBuilders or a TemplateBuilder")
        
        if solver == '':
            solver = 'ipopt'
            
        v_est_dict = dict()
        results_variances = dict()
        sigmas = dict()
        print("SOLVING VARIANCE ESTIMATION FOR INDIVIDUAL DATASETS")
        for l in self.experiments:
            print("\nsolving for dataset ", l)
            self.builder[l] = builder[l]
            self.builder[l].add_spectral_data(self.datasets[l])
            self.opt_model[l] = self.builder[l].create_pyomo_model(start_time[l],end_time[l])
            
            v_est_dict[l] = VarianceEstimator(self.opt_model[l])
            v_est_dict[l].apply_discretization('dae.collocation',nfe=nfe,ncp=ncp,scheme='LAGRANGE-RADAU')
            if method == 'alternate':
                results_variances[l] = v_est_dict[l].run_opt(solver,
                                            initial_sigmas = initsigs[l],
                                            secant_point = secant_point[l],
                                            tolerance = tolerance[l],
                                            method = method,
                                            tee=tee,
                                            solver_opts=solver_opts,
                                            max_iter=max_iter,
                                            tol=tol,
                                            subset_lambdas = A)
            else:
                results_variances[l] = v_est_dict[l].run_opt(solver,
                                            tolerance = tolerance[l],
                                            method = method,
                                            tee=tee,
                                            solver_opts=solver_opts,
                                            max_iter=max_iter,
                                            tol=tol,
                                            subset_lambdas = A)
            print("\nThe estimated variances are:\n")
            for k, v in results_variances[l].sigma_sq.items():
                print(k, v)
            self.variance_results[l] = results_variances[l]
            # and the sigmas for the parameter estimation step are now known and fixed
            sigmas[l] = results_variances[l].sigma_sq
            self.variances[l] = sigmas[l] 
        self._variance_solved = True
        
        return results_variances

    def solve_full_problem(self, solver, **kwds):
        """What is meant here is just that the confidence intervals are calculated and the model is solved.
        Sets up the reduced hessian and all other calculations for the full problem solve.
        This is not meant to be used directly by users.
        """
        
        tee = kwds.pop('tee', False)
        weights = kwds.pop('weights', [0.0, 1.0])
        covariance = kwds.pop('covariance', False)
        warmstart = kwds.pop('warmstart', False)
        species_list = kwds.pop('subset_components', None)
        solver_opts = kwds.pop('solver_opts', dict())
        
        if solver != 'ipopt_sens' and solver != 'k_aug':
            raise RuntimeError('To get covariance matrix the solver needs to be ipopt_sens or k_aug')
        if solver == 'ipopt_sens':
            # solver_opts['linear_solver'] = 'pardiso'
            if not 'compute_red_hessian' in solver_opts.keys():
                solver_opts['compute_red_hessian'] = 'yes'
        if solver == 'k_aug':
            solver_opts['compute_inv'] = ''

        optimizer = SolverFactory(solver)
        for key, val in solver_opts.items():
            optimizer.options[key] = val
        
        self._define_reduce_hess_order_mult()
        m = self.model
                    
        if solver == 'ipopt_sens':
            self._tmpfile = "ipopt_hess"
            solver_results = optimizer.solve(m,
                                             logfile=self._tmpfile, tee=tee,
                                             report_timing=True)

            print("Done solving building reduce hessian")
            output_string = ''
            with open(self._tmpfile, 'r') as f:
                output_string = f.read()
            if os.path.exists(self._tmpfile):
                os.remove(self._tmpfile)
            # output_string = f.getvalue()
            ipopt_output, hessian_output = split_sipopt_string(output_string)
            #print hessian_output
            print("build strings")
            if tee == True:
                print(ipopt_output)
            # print(self._idx_to_variable)
            n_vars = len(self._idx_to_variable)
            #print('n_vars', n_vars)
            hessian = read_reduce_hessian(hessian_output, n_vars)
            print(hessian.size, "hessian size")
            print(hessian.shape,"hessian shape")
            # hessian = read_reduce_hessian2(hessian_output,n_vars)
            # print hessian
            self._compute_covariance(hessian, self.variances)

        elif solver == 'k_aug':
            m.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
            m.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
            m.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
    
            m.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
            m.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)
            m.dof_v = Suffix(direction=Suffix.EXPORT)  #: SUFFIX FOR K_AUG
            m.rh_name = Suffix(direction=Suffix.IMPORT)  #: SUFFIX FOR K_AUG AS WELL
    
            count_vars = 1
            for i in self.experiments:
                if not self._spectra_given:
                    pass
                else:
                    for t in m.experiment[i].meas_times:
                        for c in self._sublist_components[i]:
                            m.experiment[i].C[t, c].set_suffix_value(m.dof_v, count_vars)
    
                            count_vars += 1
    
                if not self._spectra_given:
                    pass
                else:
                    for l in m.experiment[i].meas_lambdas:
                        for c in self._sublist_components[i]:
                            m.experiment[i].S[l, c].set_suffix_value(m.dof_v, count_vars)
                            count_vars += 1
    
                for v in self.model.experiment[i].P.values():
                    if v.is_fixed():
                        continue
                    m.experiment[i].P.set_suffix_value(m.dof_v, count_vars)
                    count_vars += 1

                if hasattr(self.model.experiment[i],'Pinit'):#added for the estimation of initial conditions which have to be complementary state vars CS
                    for k, v in self.model.experiment[i].Pinit.items():
                        # print(k,v,i)
                        m.experiment[i].init_conditions[k].set_suffix_value(m.dof_v, count_vars)
                            # print(m.experiment[i].Pinit, m.dof_v, count_vars)
                        count_vars += 1
                        # m.experiment[i].init_conditions[v].set_suffix_value(m.dof_v, count_vars)
                        # count_vars += 1
            
            print("count_vars:", count_vars)
            self._tmpfile = "k_aug_hess"
            ip = SolverFactory('ipopt')
            with open("ipopt.opt", "w") as f:
                f.write("print_info_string yes")
                f.close()
                
            m.write(filename="ip.nl", format=ProblemFormat.nl)
            solver_results = ip.solve(m, tee=tee,
                                      options = solver_opts,
                                      logfile=self._tmpfile,
                                      report_timing=True)
            
            m.write(filename="ka.nl", format=ProblemFormat.nl)
            k_aug = SolverFactory('k_aug')
            # k_aug.options["compute_inv"] = ""
            m.ipopt_zL_in.update(m.ipopt_zL_out)  #: be sure that the multipliers got updated!
            m.ipopt_zU_in.update(m.ipopt_zU_out)
            # m.write(filename="mynl.nl", format=ProblemFormat.nl)
            k_aug.solve(m, tee=False)
            print("Done solving building reduce hessian")
    
            if not self.all_sigma_specified:
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
                    var_loc[v] = 0
                
            vlocsize = len(var_loc)
            unordered_hessian = np.loadtxt('result_red_hess.txt')
            if os.path.exists('result_red_hess.txt'):
                os.remove('result_red_hess.txt')
           
            hessian = self.order_k_aug_hessian(unordered_hessian, var_loc)
          
            self._compute_covariance(hessian, sigma_sq)

    def solve_conc_full_problem(self, solver, **kwds):
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
        weights = kwds.pop('weights', [0.0, 1.0])
        covariance = kwds.pop('covariance', False)
        warmstart = kwds.pop('warmstart', False)
        species_list = kwds.pop('subset_components', None)
        solver_opts = kwds.pop('solver_opts', dict())
        
        #solver_opts = dict()
        if solver != 'ipopt_sens' and solver != 'k_aug':
            raise RuntimeError('To get covariance matrix the solver needs to be ipopt_sens or k_aug')
        if solver == 'ipopt_sens':
            if not 'compute_red_hessian' in solver_opts.keys():
                solver_opts['compute_red_hessian'] = 'yes'
                # solver_opts['linear_solver'] = 'ma57'
                # solver_opts['ma57_pivot_order'] = 4
        if solver == 'k_aug':
            solver_opts['compute_inv'] = ''
            # solver_opts['linear_solver']='ma86'
        #print(solver_opts)
        optimizer = SolverFactory(solver)
        for key, val in solver_opts.items():
            optimizer.options[key] = val
            
        m = self.model
        
        self._define_reduce_hess_order_mult()
        
        if covariance and solver == 'ipopt_sens':
            self._tmpfile = "ipopt_hess"
            solver_results = optimizer.solve(m, tee=tee,
                                             logfile=self._tmpfile,
                                             report_timing=True)

            print("Done solving building reduce hessian")
            output_string = ''
            with open(self._tmpfile, 'r') as f:
                output_string = f.read()
            if os.path.exists(self._tmpfile):
                os.remove(self._tmpfile)

            ipopt_output, hessian_output = split_sipopt_string(output_string)
            #print (hessian_output)
            #print("build strings")
            #if tee == True:
                #print(ipopt_output)

            n_vars = len(self._idx_to_variable)

            hessian = read_reduce_hessian(hessian_output, n_vars)
            #print(hessian.size, "hessian size")
            # hessian = read_reduce_hessian2(hessian_output,n_vars)
            sigma_sq = self.variances
            if self._concentration_given:
                self._compute_covariance_C(hessian, sigma_sq)

            # else:
            #    self._compute_covariance(hessian, sigma_sq)

        elif covariance and solver == 'k_aug':
                
            m.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
            m.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
            m.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
            m.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
            m.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)

            m.dof_v = Suffix(direction=Suffix.EXPORT)  #: SUFFIX FOR K_AUG
            m.rh_name = Suffix(direction=Suffix.IMPORT)  #: SUFFIX FOR K_AUG AS WELL
            m.dof_v.pprint()
            m.rh_name.pprint()
            count_vars = 1
        
            if not self._spectra_given:
                pass
            else:
                for i in self.experiments:
                    for t in m.experiment[i].meas_times:
                        for c in self._sublist_components[i]:
                            m.experiment[i].C[t, c].set_suffix_value(m.dof_v, count_vars)

                            count_vars += 1
                        

            if not self._spectra_given:
                pass
            else:
                for l in self._meas_lambdas:
                    for c in self._sublist_components:
                        m.S[l, c].set_suffix_value(m.dof_v, count_vars)
                        count_vars += 1

            var_counted = list()
            
            print("globs", self.global_params)
            
            for i in self.experiments:
                for k, v in self.model.experiment[i].P.items():
                    #print(k,v)                    
                    if k not in var_counted:
                        #print(count_vars)
                        # print(k,v, i)
                        if v.is_fixed():  #: Skip the fixed ones
                            print(str(v) + '\has been skipped for covariance calculations')
                            continue
                        m.experiment[i].P.set_suffix_value(m.dof_v, count_vars)
                        # print(m.experiment[i].P, m.dof_v, count_vars)
                        count_vars += 1
                        var_counted.append(k)

                if hasattr(self.model.experiment[i], 'Pinit'):#added for the estimation of initial conditions which have to be complementary state vars CS
                    for k, v in self.model.experiment[i].Pinit.items():
                        # print(k,v,i)
                        if k not in var_counted:
                            m.experiment[i].init_conditions[k].set_suffix_value(m.dof_v, count_vars)
                            # print(m.experiment[i].Pinit, m.dof_v, count_vars)
                            count_vars += 1
                            var_counted.append(k)
                    
            self._tmpfile = "k_aug_hess"
            ip = SolverFactory('ipopt')
            solver_results = ip.solve(m, tee=tee,
                                      logfile=self._tmpfile,
                                      report_timing=True)
            # m.P.pprint()
            k_aug = SolverFactory('k_aug')
            k_aug.options['compute_inv'] = ""
            # k_aug.options["no_scale"] = ""
            m.ipopt_zL_in.update(m.ipopt_zL_out)  #: be sure that the multipliers got updated!
            m.ipopt_zU_in.update(m.ipopt_zU_out)
            # m.write(filename="mynl.nl", format=ProblemFormat.nl)
            #print("do we get here?")
            k_aug.solve(m, tee=False) #True
            print("Done solving building reduce hessian")
            #
            # if not self.all_sigma_specified:
            #     raise RuntimeError(
            #         'All variances must be specified to determine covariance matrix.\n Please pass variance dictionary to run_opt')

            n_vars = len(self._idx_to_variable)
            print("n_vars", n_vars)
            m.rh_name.pprint()
            var_loc = m.rh_name
            for v in self._idx_to_variable.values():
                try:
                    var_loc[v]
                except:
                    #print(v, "is an error")
                    var_loc[v] = 0
                    #print(v, "is thus set to ", var_loc[v])
                    #print(var_loc[v])

            vlocsize = len(var_loc)
            print("var_loc size, ", vlocsize)
            unordered_hessian = np.loadtxt('result_red_hess.txt')
            if os.path.exists('result_red_hess.txt'):
                os.remove('result_red_hess.txt')
            # hessian = read_reduce_hessian_k_aug(hessian_output, n_vars)
            # hessian =hessian_output
            # print(hessian)
            print(unordered_hessian.size, "unordered hessian size")
            #hessian = self._order_k_aug_hessian(unordered_hessian, var_loc)
            hessian = self.order_k_aug_hessian(unordered_hessian, var_loc)
            
            # if self._estimability == True:
            #     self.hessian = hessian
            if self._concentration_given:
                self._compute_covariance_C(hessian, self.variances)
        else:
            solver_results = optimizer.solve(m, tee=tee)

        m.del_component('objective')
        
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
            
            spectra_problem (bool): tells whether we have spectral data or concentration data (False if not specified)

            shared_spectra (bool): tells whether spectra are shared across datasets for species (False if not specified)

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
        
        covariance = kwds.pop('covariance', False)
        
        spectra_problem = kwds.pop('spectra_problem', True)
        
        
        #init from file:
        init_files = kwds.pop('init_files', False)
        resultY = kwds.pop('resultY', dict())
        resultX = kwds.pop('resultX', dict())
        resultZ = kwds.pop('resultZ', dict())
        resultdZdt = kwds.pop('resultdZdt', dict())
        resultC = kwds.pop('resultC', dict())


        # option for non-negativity bounds for Z:
        # self.lbZ = kwds.pop('lbZ', False)
        
        spectra_shared = kwds.pop('shared_spectra', False)
        
        #read info of unwanted G for each experiment KH.L
        unwanted_G_info = kwds.pop("unwanted_G_info", dict())
        
        # scale variances for sovling parameter estimation problem.
        # sometimes the problem will be much easiler to be solved
        # aspecially when solving multiply experiment problem with unwanted G. KH.L
        scaled_variance = kwds.pop("scaled_variance", False)
        
        if covariance:
            if solver != 'ipopt_sens' and solver != 'k_aug':
                raise RuntimeError('To get covariance matrix the solver needs to be ipopt_sens or k_aug')
            #if solver == 'ipopt_sens':
            #    raise RuntimeError('To get covariance matrix for multiple experiments, the solver needs to be k_aug')
        
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
         
            
        if isinstance(builder, dict):
            for key, val in builder.items():
                if key not in self.experiments:
                    raise RuntimeError("The keys for builder need to be the same as the experimental datasets")   
                    
                if not isinstance(val, TemplateBuilder):
                    raise RuntimeError('builder needs to be type TemplateBuilder')
                    
        elif isinstance(builder, TemplateBuilder):
            builder_dict = {}
            for item in self.experiments:
                builder_dict[item] = builder
            
            builder = builder_dict
            
        else:
            raise RuntimeError('builder needs to be type dictionary of TemplateBuilder or TemplateBuilder')

        results_pest = dict()

        list_components = {}
        for k in self.experiments:            
            list_components[k] = [j for j in builder[k]._component_names]
        self._sublist_components = list_components
        
        if bool(sigma_sq) == False:
            if bool(self.variances) == True: 
                sigma_sq = self.variances
            else:
                raise RuntimeError('Need to add variances in order to run parameter estimation')
        else:
            for key, val in sigma_sq.items():

                if key not in self.experiments:
                    raise RuntimeError("The keys for sigma_sq need to be the same as the experimental datasets")
                    
                self.all_sigma_specified = True
                keys = sigma_sq[key].keys()
                expsigma = sigma_sq[key]
                
                for k in list_components[key]:
                    if k not in keys:
                        self.all_sigma_specified = False
                        expsigma[k] = max(expsigma.values())
        
                if not 'device' in val.keys():
                    self.all_sigma_specified = False
                    expsigma['device'] = 1.0


        if self._variance_solved == False:
            self.variances = sigma_sq
            
        print("\nSOLVING PARAMETER ESTIMATION FOR INDIVIDUAL DATASETS - For initialization")

        def is_empty(any_structure): #function for cases below to check if dict is empty!
            if any_structure:
                return False
            else:
                return True


        ind_p_est = dict()
        list_params_across_blocks = list()
        list_waves_across_blocks = list()
        list_species_across_blocks = list()
        all_waves = list()
        all_params = list()
        global_params = list()
        global_waves = list()
        all_species = list()
        shared_species = list()
        
        if unwanted_G_info:
            # check which experiments have unwanted G and which don't have. KH.L
            exps_w_G = list()
            exps_wo_G = list()
            for i in self.experiments:
                if i in unwanted_G_info.keys():
                    exps_w_G.append(i)
                else:
                    exps_wo_G.append(i)
            # print(exps_w_G,exps_wo_G,"line1616")           
            
            # categorize different types of G and write in detailed_G_type. KH.L
            detailed_G_type = dict()
            unwanted_G = dict()
            time_variant_G = dict()
            time_invariant_G = dict()
            St_dict = dict()
            Z_in_dict = dict()
            for i in exps_w_G:
                if unwanted_G_info[i]["type"] == "unwanted_G":
                    detailed_G_type[i] = "time_variant_G"
                    unwanted_G[i] =  True
                    time_variant_G[i] = False
                    time_invariant_G[i]  = False
                    St_dict[i] = dict()
                    Z_in_dict[i] = dict()
                elif unwanted_G_info[i]["type"] == "time_variant_G":
                    detailed_G_type[i] = "time_variant_G"
                    unwanted_G[i] =  False
                    time_variant_G[i] = True
                    time_invariant_G[i]  = False
                    St_dict[i] = dict()
                    Z_in_dict[i] = dict()
                elif unwanted_G_info[i]["type"] == "time_invariant_G":                    
                    unwanted_G[i] =  False
                    time_variant_G[i] = False
                    time_invariant_G[i]  = True
                    if "St" in unwanted_G_info[i].keys() and "Z_in" in unwanted_G_info[i].keys():
                        St_dict[i] = unwanted_G_info[i]["St"]
                        Z_in_dict[i] = unwanted_G_info[i]["Z_in"]
                    elif "St" in unwanted_G_info[i].keys() and "Z_in" not in unwanted_G_info[i].keys():
                        St_dict[i] = unwanted_G_info[i]["St"]
                        Z_in_dict[i] = dict()
                    elif "St" not in unwanted_G_info[i].keys() and "Z_in" in unwanted_G_info[i].keys():
                        St_dict[i] = dict()
                        Z_in_dict[i] = unwanted_G_info[i]["Z_in"]
                        
                    omega_list = list()
                    for s in St_dict[i].keys():
                        omega_list.append(St_dict[i][s])
                    for t in Z_in_dict[i].keys():
                        omega_list.append(Z_in_dict[i][t])
                    omega_sub = np.array(omega_list)
                    rank = np.linalg.matrix_rank(omega_sub)
                    sha = omega_sub.shape
                    cols = sha[1]
                    rko = cols - rank
                    # print(omega_sub, rko)
                    if rko > 0:
                        detailed_G_type[i] = "time_invariant_G_decompose"
                    else:
                        detailed_G_type[i] = "time_invariant_G_no_decompose"

        for l in self.experiments:
            print("\nSolving for DATASET ", l)
            if spectra_problem == True:
                if self._variance_solved == True:
                    # then we already have inits
                    ind_p_est[l] = ParameterEstimator(self.opt_model[l])
                    
                    # if unwanted_G_info == dict() or detailed_G_type[l] != "time_variant_G":
                    ind_p_est[l].apply_discretization('dae.collocation',nfe=nfe,ncp=ncp,scheme='LAGRANGE-RADAU')
                    ind_p_est[l].initialize_from_trajectory('Z', self.variance_results[l].Z)
                    if hasattr(ind_p_est[l], 'S'):
                        ind_p_est[l].initialize_from_trajectory('S', self.sim_results[l].S)
                    ind_p_est[l].initialize_from_trajectory('C', self.variance_results[l].C)
                    # NOTICE here that we may need to add X and Y variables and DZdt vars here depending on the situtation
                    # This needs to be done based on their existence.
                    ind_p_est[l].scale_variables_from_trajectory('Z', self.variance_results[l].Z)
                    ind_p_est[l].scale_variables_from_trajectory('S', self.variance_results[l].S)
                    ind_p_est[l].scale_variables_from_trajectory('C', self.variance_results[l].C)

                    if unwanted_G_info:
                        if l in exps_w_G:
                            results_pest[l] = ind_p_est[l].run_opt('ipopt',
                                                                   tee=tee,
                                                                   solver_opts=solver_opts,
                                                                   variances=self.variances[l],
                                                                   unwanted_G = unwanted_G[l],
                                                                   time_variant_G = time_variant_G[l],
                                                                   time_invariant_G = time_invariant_G[l],
                                                                   St = St_dict[l],
                                                                   Z_in = Z_in_dict[l])
                        elif l in exps_wo_G:
                            results_pest[l] = ind_p_est[l].run_opt('ipopt',
                                                                   tee=tee,
                                                                   solver_opts=solver_opts,
                                                                   variances=self.variances[l])
                            
                    else:
                        results_pest[l] = ind_p_est[l].run_opt('ipopt',
                                                               tee=tee,
                                                               solver_opts=solver_opts,
                                                               variances=self.variances[l])
                    
                    self.initialization_model[l] = ind_p_est[l]

                    print("The estimated parameters are:")
                    for k, v in results_pest[l].P.items():
                        print(k, v)
                        if k not in all_params:
                            all_params.append(k)
                        else:
                            global_params.append(k)

                        if k not in list_params_across_blocks:
                            list_params_across_blocks.append(k)

                    if hasattr(results_pest[l],
                               'Pinit'):  # added for the estimation of initial conditions which have to be complementary state vars CS
                        print("The estimated parameters are:")
                        for k, v in results_pest[l].Pinit.items():
                            print(k, v)
                            if k not in all_params:
                                all_params.append(k)
                            else:
                                global_params.append(k)

                            if k not in list_params_across_blocks:
                                list_params_across_blocks.append(k)

                else:
                    self._spectra_given = True
                    self.builder[l]=builder[l]
                    self.builder[l].add_spectral_data(self.datasets[l])
                    self.opt_model[l] = self.builder[l].create_pyomo_model(start_time[l],end_time[l])
                    ind_p_est[l] = ParameterEstimator(self.opt_model[l])
                    ind_p_est[l].apply_discretization('dae.collocation',nfe=nfe,ncp=ncp,scheme='LAGRANGE-RADAU')
                    
                    if unwanted_G_info:
                        if l in exps_w_G:
                            results_pest[l] = ind_p_est[l].run_opt('ipopt',
                                                                   tee=tee,
                                                                   solver_opts=solver_opts,
                                                                   variances = sigma_sq[l],
                                                                   unwanted_G = unwanted_G[l],
                                                                   time_variant_G = time_variant_G[l],
                                                                   time_invariant_G = time_invariant_G[l],
                                                                   St = St_dict[l],
                                                                   Z_in = Z_in_dict[l])
                        elif l in exps_wo_G:
                            results_pest[l] = ind_p_est[l].run_opt('ipopt',
                                                                   tee=tee,
                                                                   solver_opts = solver_opts,
                                                                   variances = sigma_sq[l])
                            
                    else:
                        results_pest[l] = ind_p_est[l].run_opt('ipopt',
                                                               tee=tee,
                                                               solver_opts = solver_opts,
                                                               variances = sigma_sq[l])

                    self.initialization_model[l] = ind_p_est[l]

                    print("The estimated parameters are:")
                    for k, v in results_pest[l].P.items():
                        print(k, v)
                        if k not in all_params:
                            all_params.append(k)
                        else:
                            global_params.append(k)

                        if k not in list_params_across_blocks:
                            list_params_across_blocks.append(k)

                    if hasattr(results_pest[l],
                               'Pinit'):  # added for the estimation of initial conditions which have to be complementary state vars CS
                        print("The estimated parameters are:")
                        for k, v in results_pest[l].Pinit.items():
                            print(k, v)
                            if k not in all_params:
                                all_params.append(k)
                            else:
                                global_params.append(k)

                            if k not in list_params_across_blocks:
                                list_params_across_blocks.append(k)
                
                if spectra_shared ==True:
                    for wa in ind_p_est[l].model.meas_lambdas:
                        if wa not in all_waves:
                            all_waves.append(wa)
                        else:
                            global_waves.append(wa)

                        if wa not in list_waves_across_blocks:
                            list_waves_across_blocks.append(wa)
                if spectra_shared ==True:
                    for sp in ind_p_est[l].model.mixture_components:
                        if sp not in all_species:
                            all_species.append(sp)
                        else:
                            shared_species.append(sp)

                        if sp not in list_species_across_blocks:
                            list_species_across_blocks.append(sp)

            else:
                if self._sim_solved == True:# and spectra_problem == False:
                    # then we already have inits
                    self._spectra_given = False
                    self._concentration_given = True
                    self.builder[l] = builder[l]
                    self.builder[l].add_concentration_data(self.datasets[l])
                    self.opt_model[l] = self.builder[l].create_pyomo_model(start_time[l], end_time[l])
                    ind_p_est[l] = ParameterEstimator(self.opt_model[l])
                    ind_p_est[l].apply_discretization('dae.collocation',nfe=nfe,ncp=ncp,scheme='LAGRANGE-RADAU')
                    ind_p_est[l].initialize_from_trajectory('Z', self.sim_results[l].Z)
                    ind_p_est[l].initialize_from_trajectory('dZdt', self.sim_results[l].dZdt)
                    if hasattr(ind_p_est[l], 'C'):
                        ind_p_est[l].initialize_from_trajectory('C', self.sim_results[l].C)
                    if hasattr(ind_p_est[l], 'Y'):
                        ind_p_est[l].initialize_from_trajectory('Y', self.sim_results[l].Y)
                        # ind_p_est[l].scale_variables_from_trajectory('Y', self.sim_results[l].Y)
                    if hasattr(ind_p_est[l], 'X'):
                        ind_p_est[l].initialize_from_trajectory('X', self.sim_results[l].X)


                    results_pest[l] = ind_p_est[l].run_opt('ipopt',
                                                           tee=tee,
                                                           solver_opts=solver_opts,
                                                           variances=sigma_sq[l])
                    
                    self.initialization_model[l] = ind_p_est[l]

                    print("The estimated parameters are:")
                    for k, v in results_pest[l].P.items():
                        print(k, v)
                        if k not in all_params:
                            all_params.append(k)
                        else:
                            global_params.append(k)

                        if k not in list_params_across_blocks:
                            list_params_across_blocks.append(k)

                    if hasattr(results_pest[l],
                               'Pinit'):  # added for the estimation of initial conditions which have to be complementary state vars CS
                        print("The estimated parameters are:")
                        for k, v in results_pest[l].Pinit.items():
                            print(k, v)
                            if k not in all_params:
                                all_params.append(k)
                            else:
                                global_params.append(k)

                            if k not in list_params_across_blocks:
                                list_params_across_blocks.append(k)


                elif init_files == True:
                    # then we already have inits
                    self._spectra_given = False
                    self._concentration_given = True
                    self.builder[l] = builder[l]
                    self.builder[l].add_concentration_data(self.datasets[l])
                    self.opt_model[l] = self.builder[l].create_pyomo_model(start_time[l], end_time[l])
                    ind_p_est[l] = ParameterEstimator(self.opt_model[l])
                    ind_p_est[l].apply_discretization('dae.collocation', nfe=nfe, ncp=ncp, scheme='LAGRANGE-RADAU')

                    if is_empty(resultZ) == False:
                        ind_p_est[l].initialize_from_trajectory('Z', resultZ[l])
                    if is_empty(resultdZdt) == False:
                        ind_p_est[l].initialize_from_trajectory('dZdt', resultdZdt[l])
                    if is_empty(resultC) == False:
                        ind_p_est[l].initialize_from_trajectory('C', resultC[l])
                    if is_empty(resultX) == False:
                        ind_p_est[l].initialize_from_trajectory('X', resultX[l])
                    if is_empty(resultY) == False:
                        ind_p_est[l].initialize_from_trajectory('Y', resultY[l])

                    results_pest[l] = ind_p_est[l].run_opt('ipopt',
                                                           tee=tee,
                                                           solver_opts=solver_opts,
                                                           variances=sigma_sq[l])
                    # with open('filemodelMultexpbef.txt', 'w') as f:
                    #     ind_p_est[l].model.pprint(ostream=f)
                    #     f.close()

                    self.initialization_model[l] = ind_p_est[l]

                    print("The estimated parameters are:")
                    for k, v in results_pest[l].P.items():
                        print(k, v)
                        if k not in all_params:
                            all_params.append(k)
                        else:
                            global_params.append(k)

                        if k not in list_params_across_blocks:
                            list_params_across_blocks.append(k)

                    if hasattr(results_pest[l],
                               'Pinit'):  # added for the estimation of initial conditions which have to be complementary state vars CS
                        print("The estimated parameters are:")
                        for k, v in results_pest[l].Pinit.items():
                            print(k, v)
                            if k not in all_params:
                                all_params.append(k)
                            else:
                                global_params.append(k)

                            if k not in list_params_across_blocks:
                                list_params_across_blocks.append(k)

                    # print("all_params:" , all_params)
                    # print("global_params:", global_params)
                else:
                    self._spectra_given = False
                    self._concentration_given = True
                    self.builder[l]=builder[l]
                    self.builder[l].add_concentration_data(self.datasets[l])
                    self.opt_model[l] = self.builder[l].create_pyomo_model(start_time[l],end_time[l])

                    ind_p_est[l] = ParameterEstimator(self.opt_model[l])
                    ind_p_est[l].apply_discretization('dae.collocation',nfe=nfe,ncp=ncp,scheme='LAGRANGE-RADAU')

                    results_pest[l] = ind_p_est[l].run_opt('ipopt',
                                                         tee=tee,
                                                          solver_opts = solver_opts,
                                                          variances = sigma_sq[l])


                    self.initialization_model[l] = ind_p_est[l]

                    print("The estimated parameters are:")
                    for k,v in results_pest[l].P.items():
                        print(k, v)
                        if k not in all_params:
                            all_params.append(k)
                        else:
                            if k not in global_params:
                                global_params.append(k)
                            else:
                                pass
                        if k not in list_params_across_blocks:
                            list_params_across_blocks.append(k)

                    if hasattr(results_pest[l], 'Pinit'):#added for the estimation of initial conditions which have to be complementary state vars CS
                        print("The estimated parameters are:")
                        for k, v in results_pest[l].Pinit.items():
                            print(k, v)
                            if k not in all_params:
                                all_params.append(k)
                            else:
                                global_params.append(k)

                            if k not in list_params_across_blocks:
                                list_params_across_blocks.append(k)

                #print("all_params:" , all_params)
                #print("global_params:", global_params)
                self.global_params = global_params

                if spectra_shared ==True:
                    for wa in ind_p_est[l].model.meas_lambdas:
                        if wa not in all_waves:
                            all_waves.append(wa)
                        else:
                            global_waves.append(wa)

                        if wa not in list_waves_across_blocks:
                            list_waves_across_blocks.append(wa)

                if spectra_shared ==True:
                    for sp in ind_p_est[l].model.mixture_components:
                        if sp not in all_species:
                            #print(sp)
                            all_species.append(sp)
                        else:
                            shared_species.append(sp)

                        if sp not in list_species_across_blocks:
                            list_species_across_blocks.append(sp)
        
        print("\nSOLVING PARAMETER ESTIMATION FOR MULTIPLE DATASETS\n")
        #Now that we have all our datasets solved individually we can build our blocks and use
        #these solutions to initialize
        m = ConcreteModel()
        m.solver_opts = solver_opts
        m.tee = tee
        
        if scaled_variance == True:
            var_scaled = dict()
            for s,t in self.variances.items():
                maxx = max(list(t.values()))
                ind_var = dict()
                for i,j in t.items():
                    ind_var[i] = j/maxx
                var_scaled[s] = ind_var
            self.variances = var_scaled
        
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
            if with_d_vars and self._spectra_given:
                m.D_bar = Var(m.meas_times,
                              m.meas_lambdas)
    
                def rule_D_bar(m, t, l):
                    return m.D_bar[t, l] == sum(m.C[t, k] * m.S[l, k] for k in self.initialization_model[exp]._sublist_components)
    
                m.D_bar_constraint = Constraint(m.meas_times,
                                                m.meas_lambdas,
                                                rule=rule_D_bar)
            
            m.error = Var(bounds = (0, None))
            
            if self._spectra_given:
                def rule_objective(m):
                    expr = 0
                    for t in m.meas_times:
                        for l in m.meas_lambdas:
                            if with_d_vars:
                                if unwanted_G_info:
                                    if exp in exps_w_G:
                                        if detailed_G_type[exp] == "time_variant_G":
                                            expr += (m.D[t, l] - m.D_bar[t, l] - m.qr[t]*m.g[l]) ** 2 / (self.variances[exp]['device'])
                                        elif detailed_G_type[exp] == "time_invariant_G_decompose" or detailed_G_type[exp] == "time_invariant_G_no_decompose":
                                            if spectra_shared == True:
                                                expr += (m.D[t, l] - m.D_bar[t, l] - m.g[l]) ** 2 / (self.variances[exp]['device'])
                                            elif spectra_shared == False:
                                                if detailed_G_type[exp] == "time_invariant_G_decompose":
                                                    expr += (m.D[t, l] - m.D_bar[t, l]) ** 2 / (self.variances[exp]['device'])
                                                elif detailed_G_type[exp] == "time_invariant_G_no_decompose":
                                                    expr += (m.D[t, l] - m.D_bar[t, l] - m.g[l]) ** 2 / (self.variances[exp]['device'])
                                    elif exp in exps_wo_G:
                                        expr += (m.D[t, l] - m.D_bar[t, l]) ** 2 / (self.variances[exp]['device'])
                                else:
                                    expr += (m.D[t, l] - m.D_bar[t, l]) ** 2 / (self.variances[exp]['device'])
                            else:
                                D_bar = sum(m.C[t, k] * m.S[l, k] for k in list_components)
                                if unwanted_G_info:
                                    if exp in exps_w_G:
                                        if detailed_G_type[exp] == "time_variant_G":
                                            expr += (m.D[t, l] - D_bar - m.qr[t]*m.g[l]) ** 2 / (self.variances[exp]['device'])
                                        elif detailed_G_type_G_type[exp] == "time_invariant_G_decompose" or detailed_G_type[exp] == "time_invariant_G_no_decompose":
                                            if spectra_shared == True:
                                                expr += (m.D[t, l] - D_bar - m.g[l]) ** 2 / (self.variances[exp]['device'])
                                            elif spectra_shared == False:
                                                if detailed_G_type[exp] == "time_invariant_G_decompose":
                                                    expr += (m.D[t, l] - D_bar) ** 2 / (self.variances[exp]['device'])
                                                elif detailed_G_type[exp] == "time_invariant_G_no_decompose":
                                                    expr += (m.D[t, l] - D_bar - m.g[l]) ** 2 / (self.variances[exp]['device'])
                                    elif exp in exps_wo_G:
                                        expr += (m.D[t, l] - D_bar) ** 2 / (self.variances[exp]['device'])            
                                else:
                                    expr += (m.D[t, l] - D_bar) ** 2 / (self.variances[exp]['device'])
                    
                    #If we require weights then we would add them back in here
                    #expr *= weights[0]
                    second_term = 0.0
                    
                    # Potentially big issue with concentration data from spectra and concentration data from measurements
                    second_term = conc_objective(m, variance=self.variances[exp], source='spectra')
                    
                    #expr += weights[1] * second_term
                    expr += second_term
                    return m.error == expr
        
                m.obj_const = Constraint(rule=rule_objective)
            elif self._concentration_given:
                def rule_objective(m):
                    obj = 0
                    obj += conc_objective(m, variance=self.variances[exp])
                    # for t in m.meas_times:
                    #     obj += sum((m.C[t, k] - m.Z[t, k]) ** 2 / self.variances[exp][k] for k in list_components)
                    #print(obj)
                    return m.error == obj
        
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
            count += 1
        def param_linking_rule(m, exp, param):
            prev_exp = None
            if exp == m.first_exp:

                return Constraint.Skip
            else:
                for key, val in m.map_exp_to_count.items():
                    #print(key,val)
                    if val == exp:
                        prev_exp = m.map_exp_to_count[key-1]
                if param in global_params and prev_exp != None:
                    #This here is to check that the correct linking constraints are constructed
                    #print("this constraint is written:")
                    #print(m.experiment[exp].P[param],"=", m.experiment[prev_exp].P[param])
                    return m.experiment[exp].P[param] == m.experiment[prev_exp].P[param]
                    
                else:
                    return Constraint.Skip
        #Without the fixed ones:
        new_list_params_across_blocks=list()
        # print(self.list_fixed_params)
        list_fixed_params=list()
        for i in self.experiments:
            for param in m.experiment[i].P.keys():
                if m.experiment[i].P[param].is_fixed():
                    if param not in list_fixed_params:
                        list_fixed_params.append(param)
                    # else:
                    #     print('WARNING: The fixed parameters across experiments do not match.')
        print("Fixed parameters are: ", list_fixed_params)
        for k in list_params_across_blocks:
            if k not in list_fixed_params:
                new_list_params_across_blocks.append(k)
        m.parameter_linking = Constraint(self.experiments, new_list_params_across_blocks, rule = param_linking_rule)
        
        def wavelength_linking_rule(m, exp, wave, comp):
            prev_exp = None
            if exp == m.first_exp:
                return Constraint.Skip
            else:
                for key, val in m.map_exp_to_count.items():
                    #print(key,val)
                    if val == exp:
                        prev_exp = m.map_exp_to_count[key-1]
                if wave in list_waves_across_blocks and prev_exp != None:
                    #This here is to check that the correct linking constraints are constructed

                    #print(self.initialization_model[exp]._mixture_components)
                    #print(comp)
                    #print("this constraint is written:")
                    #print(m.experiment[exp].S[wave,comp],"=", m.experiment[prev_exp].S[wave,comp])
                    if comp in m.experiment[prev_exp].mixture_components and comp in m.experiment[exp].mixture_components:
                        return m.experiment[exp].S[wave,comp] == m.experiment[prev_exp].S[wave,comp]
                    else:
                        return Constraint.Skip
                else:
                    return Constraint.Skip

        if spectra_shared == True:
            #print(list_components[self.experiments])
            #print(list_waves_across_blocks)
            #print(self.experiments)
            #print(list_species_across_blocks)
            m.spectra_linking = Constraint(self.experiments, list_waves_across_blocks, list_species_across_blocks, rule = wavelength_linking_rule)
        
        m.obj = Objective(sense = minimize, expr=sum(b.error for b in m.experiment[:]))
        self.model = m
        
        if covariance and solver == 'k_aug' and self._spectra_given:
            #Not yet working
            
            self.solve_full_problem(solver, tee = tee, solver_opts = solver_opts)
            #solver_opts['compute_inv'] = ''
            
        elif covariance and solver == 'ipopt_sens' and self._spectra_given:
            #Not yet working
            
            self.solve_full_problem(solver, tee = tee, solver_opts = solver_opts)
            #solver_opts['compute_inv'] = ''    
            
        elif self._spectra_given:
            #Working
            # with open("multimodel_E2E5.txt","w") as f:
            #     m.pprint(ostream = f)
            # print('line2203')
            optimizer = SolverFactory('ipopt')
            solver_results = optimizer.solve(m, options = solver_opts,tee=tee)

        elif covariance and solver == 'k_aug' and self._concentration_given:   
            self.solve_conc_full_problem(solver, covariance = covariance, tee=tee, solver_opts=solver_opts)
            
        elif covariance and solver == 'ipopt_sens' and self._concentration_given:   
            self.solve_conc_full_problem(solver, covariance = covariance, tee=tee, solver_opts=solver_opts)
            
        elif self._concentration_given:
            #Working
            optimizer = SolverFactory('ipopt')
            solver_results = optimizer.solve(m, options = solver_opts,tee=tee)
            
        solver_results = dict()   
        
        # loading the results, notice that we return a dictionary
        for i in m.experiment:
            solver_results[i] = ResultsObject()
            if hasattr(m.experiment[i], 'Pinit'):#added for the estimation of initial conditions which have to be complementary state vars CS
                solver_results[i].load_from_pyomo_model(m.experiment[i],
                                                        to_load=['Z', 'dZdt', 'X', 'dXdt', 'C', 'S', 'Y', 'P', 'Pinit'])
            else:
                solver_results[i].load_from_pyomo_model(m.experiment[i],
                                                        to_load=['Z', 'dZdt', 'X', 'dXdt', 'C', 'S', 'Y', 'P'])
        
        return solver_results