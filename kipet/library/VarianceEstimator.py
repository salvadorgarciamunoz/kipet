from __future__ import print_function
from __future__ import division
from pyomo.environ import *
from pyomo.dae import *
from pyomo.core import *
from pyomo.opt import (ReaderFactory,
                       ResultsFormat)
from kipet.library.Optimizer import *
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
#import pyutilib.subprocess
import matplotlib.pylab as plt
import subprocess
import time
import copy
import sys
import os
import re
import math
from pyomo.core.expr import current as EXPR
from pyomo.core.expr.numvalue import NumericConstant
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import six

import pdb


class VarianceEstimator(Optimizer):
    """Optimizer for variance estimation.

    Attributes:

        model (model): Pyomo model.

    """
    def __init__(self, model):
        super(VarianceEstimator, self).__init__(model)
        add_warm_start_suffixes(self.model)

        if not self._spectra_given:
            raise NotImplementedError("Variance estimator requires spectral data in model as model.D[ti,lj]")
        self._is_D_deriv = False
    def run_sim(self, solver, **kwds):
        raise NotImplementedError("VarianceEstimator object does not have run_sim method. Call run_opt")

    def run_opt(self, solver, **kwds):

        """Solves estimation following either the original Chen etal (2016) procedure or via the 
        maximum likelihood estimation with unknown covariance matrix. Chen's method solves a sequence 
        of optimization problems to determine variances and initial guesses for parameter estimation.
        The maximum likelihood estimation with unknown covariance matrix also solves a sequence of optimization
        problems is a more robust and reliable method, albeit somputationally costly.

        Args:

            solver_opts (dict, optional): options passed to the nonlinear solver
            
            method (str, optional): default is "Chen", other options are "max_likelihood" and "iterative_method"
            
            initial_sigmas (dict, optional): Required for "iterative_method"

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

            report_time (bool, optional): True if variance estimation is timed. Default False
            
            fixed_device_variance (float, optional): If the device variance is known in advanced we can fix it here.
                                                Only to be used in conjunction with lsq_ipopt = True.

        Returns:

            Results from the optimization (pyomo model)

        """

        solver_opts = kwds.pop('solver_opts', dict())
        sigma_sq = kwds.pop('variances', dict())
        init_sigmas = kwds.pop('initial_sigmas', dict())
        tee = kwds.pop('tee', False)
        method = kwds.pop('method', str())
        norm_order = kwds.pop('norm',np.inf)
        max_iter = kwds.pop('max_iter', 400)
        tol = kwds.pop('tolerance', 5.0e-5)
        A = kwds.pop('subset_lambdas', None)
        lsq_ipopt = kwds.pop('lsq_ipopt', False)
        init_C = kwds.pop('init_C', None)
        report_time = kwds.pop('report_time', False)

        # additional arguments for inputs CS
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
        
        # Modified variance estimation procedures arguments
        fixed_device_var = kwds.pop('fixed_device_variance', None)
        device_range = kwds.pop('device_range', None)
        num_points = kwds.pop('num_points', None)
        
        if method not in ['Chen', "max_likelihood", "iterative_method", "direct_sigmas"]:
            method = 'Chen'
            print("Method not set, so assumed that the Chen method is chosen")
            
        if method not in ['Chen', "max_likelihood", "iterative_method", "direct_sigmas"]:
            raise RuntimeError("method must be either \"Chen\", \"max_likelihood\", \"direct_sigmas\" or \"iterative_method\"")
        
        if not self.model.time.get_discretization_info():
            raise RuntimeError('apply discretization first before initializing')

        self._create_tmp_outputs()
        
        # deactivates objective functions                
        objectives_map = self.model.component_map(ctype=Objective, active=True)
        active_objectives_names = []
        for obj in six.itervalues(objectives_map):
            name = obj.cname()
            active_objectives_names.append(name)
            obj.deactivate()

        list_components = []
        if species_list is None:
            list_components = [k for k in self._mixture_components]
        else:
            for k in species_list:
                if k in self._mixture_components:
                    list_components.append(k)
                else:
                    warnings.warn("Ignored {} since is not a mixture component of the model".format(k))

        self._sublist_components = list_components

        if hasattr(self.model, 'non_absorbing'):
            warnings.warn("Overriden by non_absorbing")
            list_components = [k for k in self._mixture_components if k not in self._non_absorbing]
            self._sublist_components = list_components
        
        if hasattr(self.model, 'known_absorbance'):
            warnings.warn("Overriden by species with known absorbance")
            list_components = [k for k in self._mixture_components if k not in self._known_absorbance]
            self._sublist_components = list_components
        
        #############################
        """inputs section""" # additional section for inputs from trajectory and fixed inputs, CS
        self.fixedtraj = fixedtraj
        self.fixedy = fixedy
        self.inputs_sub = inputs_sub
        self.yfix = yfix
        self.yfixtraj = yfixtraj
        if self.inputs_sub!=None:
            for k in self.inputs_sub.keys():
                if not isinstance(self.inputs_sub[k], list):
                    print("wrong type for inputs_sub {}".format(type(self.inputs_sub[k])))
                    # raise Exception
                for i in self.inputs_sub[k]:
                    if self.fixedtraj==True or self.fixedy==True:
                        if self.fixedtraj==True:
                            for j in self.yfixtraj.keys():
                                for l in self.yfixtraj[j]:
                                    if i==l:
                                        # print('herel:fixedy', l)
                                        if not isinstance(self.yfixtraj[j], list):
                                            print("wrong type for yfixtraj {}".format(type(self.yfixtraj[j])))
                                        reft = trajectories[(k, i)]
                                        self.fix_from_trajectory(k, i, reft)
                        if self.fixedy==True:
                            for j in self.yfix.keys():
                                for l in self.yfix[j]:
                                    if i==l:
                                        # print('herel:fixedy',l)
                                        if not isinstance(self.yfix[j], list):
                                            print("wrong type for yfix {}".format(type(self.yfix[j])))
                                        for key in self.model.time.value:
                                            vark=getattr(self.model,k)
                                            vark[key, i].set_value(key)
                                            vark[key, i].fix()# since these are inputs we need to fix this
                    else:
                        print("A trajectory or fixed input is missing for {}\n".format((k, i)))
        """/end inputs section"""

        if jump:
            self.disc_jump_v_dict = var_dic
            self.jump_times_dict = jump_times  # now dictionary
            self.feed_times_set = feed_times
            if not isinstance(self.disc_jump_v_dict, dict):
                print("disc_jump_v_dict is of type {}".format(type(self.disc_jump_v_dict)))
                raise Exception  # wrong type
            if not isinstance(self.jump_times_dict, dict):
                print("disc_jump_times is of type {}".format(type(self.jump_times_dict)))
                raise Exception  # wrong type
            count = 0
            for i in six.iterkeys(self.jump_times_dict):
                for j in six.iteritems(self.jump_times_dict[i]):
                    count += 1
            if len(self.feed_times_set) > count:
                raise Exception("Error: Check feed time points in set feed_times and in jump_times again.\n"
                            "There are more time points in feed_times than jump_times provided.")
            self.load_discrete_jump()
        ######################################################

        if report_time:
            start = time.time()
        if method == 'Chen':    
            # solves formulation 18
            if init_C is None:
                self._solve_initalization(solver, subset_lambdas=A, solver_opts = solver_opts, tee=tee)
            else:
                for t in self._meas_times:
                    for k in self._mixture_components:
                        self.model.C[t, k].value = init_C[k][t]
                        self.model.Z[t, k].value = init_C[k][t]
    
                s_array = self._solve_S_from_DC(init_C)
                S_frame = pd.DataFrame(data=s_array,
                                       columns=self._mixture_components,
                                       index=self._meas_lambdas)
                # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
                if hasattr(self, '_abs_components'):
                    for l in self._meas_lambdas:
                        for k in self._abs_components:
                            self.model.S[l, k].value = S_frame[k][l]  # 1e-2
                            #: Some of these are gonna be non-zero
                            # if hasattr(self.model, 'non_absorbing'):
                            #     if k in self.model.non_absorbing:
                            #         self.model.S[l, k].value = 0.0
    
                            if hasattr(self.model, 'known_absorbance'):
                                if k in self.model.known_absorbance:
                                    self.model.S[l, k].value = self.model.known_absorbance_data[k][l]
                else:
                    for l in self._meas_lambdas:
                        for k in self._mixture_components:
                            self.model.S[l, k].value = S_frame[k][l]  # 1e-2
                            #: Some of these are gonna be non-zero
                            # if hasattr(self.model, 'non_absorbing'):
                            #     if k in self.model.non_absorbing:
                            #         self.model.S[l, k].value = 0.0
    
                            if hasattr(self.model, 'known_absorbance'):
                                if k in self.model.known_absorbance:
                                    self.model.S[l, k].value = self.model.known_absorbance_data[k]
    
            #start looping
            #print("{: >11} {: >20} {: >16} {: >16}".format('Iter','|Zi-Zi+1|','|Ci-Ci+1|','|Si-Si+1|'))
            print("{: >11} {: >20}".format('Iter', '|Zi-Zi+1|'))
            logiterfile = "iterations.log"
            if os.path.isfile(logiterfile):
                os.remove(logiterfile)
    
            # backup
            if lsq_ipopt:
                self._build_s_model()
                self._build_c_model()
            else:
                if species_list is None:
                    self._build_scipy_lsq_arrays()
                else:
                    lsq_ipopt = True
                    self._build_s_model()
                    self._build_c_model()
                
            for it in range(max_iter):
                
                rb = ResultsObject()
                # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
                if hasattr(self, '_abs_components'):
                    rb.load_from_pyomo_model(self.model, to_load=['Z', 'C', 'Cs', 'S', 'Y'])
                else:
                    rb.load_from_pyomo_model(self.model, to_load=['Z', 'C', 'S', 'Y'])
                
                self._solve_Z(solver)
    
                if lsq_ipopt:
                    self._solve_S(solver)
                    self._solve_C(solver)
                else:
                    solved_s = self._solve_s_scipy()
                    solved_c = self._solve_c_scipy()
                    
                #pdb.set_trace()
                
                ra=ResultsObject()    
                # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
                if hasattr(self, '_abs_components'):
                    ra.load_from_pyomo_model(self.model, to_load=['Z','C','Cs','S'])
                else:
                    ra.load_from_pyomo_model(self.model, to_load=['Z', 'C', 'S'])
                
                r_diff = compute_diff_results(rb,ra)
    
                
                Z_norm = r_diff.compute_var_norm('Z',norm_order)
                #C_norm = r_diff.compute_var_norm('C',norm_order)
                #S_norm = r_diff.compute_var_norm('S',norm_order)
                if it>0:
                    #print("{: >11} {: >20} {: >16} {: >16}".format(it,Z_norm,C_norm,S_norm))
                    print("{: >11} {: >20}".format(it, Z_norm))
                self._log_iterations(logiterfile, it)
                if Z_norm<tol and it >= 1:
                    break
                    
            results = ResultsObject()
            
            # retriving solutions to results object  
            # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
            if hasattr(self, '_abs_components'):
                results.load_from_pyomo_model(self.model,
                                          to_load=['Z', 'dZdt', 'X', 'dXdt', 'C', 'Cs', 'S', 'Y'])
            else:
                results.load_from_pyomo_model(self.model,
                                              to_load=['Z', 'dZdt', 'X', 'dXdt', 'C', 'S', 'Y'])
    
            print('Iterative optimization converged. Estimating variances now')
            # compute variances
            solved_variances = self._solve_variances(results, fixed_dev_var = fixed_device_var)
            
            self.compute_D_given_SC(results)
            
            param_vals = dict()
            for name in self.model.parameter_names:
                param_vals[name] = self.model.P[name].value
    
            results.P = param_vals
    
            # removes temporary files. This needs to be changes to work with pyutilib
            if os.path.exists(self._tmp2):
                os.remove(self._tmp2)
            if os.path.exists(self._tmp3):
                os.remove(self._tmp3)
            if os.path.exists(self._tmp4):
                os.remove(self._tmp4)
        elif method == 'max_likelihood':
            # First we initialize the problem by solving with our fixed variances
            #self._solve_iterative_init(solver, variances = init_sigmas, solver_options = solver_opts, tee = tee)
            # Solving the maximum likelihood variance estimation
            sigma_sq['device'] = self._solve_max_likelihood_init(solver, subset_lambdas= A, solver_opts = solver_opts, tee=tee)
            
            self._solve_max_likelihood(solver, subset_lambdas= A, solver_opts = solver_opts, tee=tee)
        elif method == 'direct_sigmas':            
            print("Solving for sigmas assuming known device variances")
            direct_or_it = "it"
            if device_range:
                if not isinstance(device_range, tuple):
                    print("device_range is of type {}".format(type(device_range)))
                    print("It should be a tuple")
                    raise Exception
                else:
                    print("Device range means that we will solve iteratively for different delta values in that range")
            if device_range and not num_points:
                print("Need to specify the number of points that we wish to evaluate in the device range")
            if not num_points:
                pass
            elif not isinstance(num_points, int):  
                print("num_points needs to be an integer!")
                raise Exception
            if not device_range:
                device_range = list() 
            if not device_range and not num_points:
                direct_or_it = "direct"
                print("assessing for the value of delta provided")
                if not fixed_device_var:
                    print("If iterative method not selected then need to provide fixed device variance (delta**2)")
                    raise Exception
                else:
                    if not isinstance(fixed_device_var, float):
                        raise Exception("fixed device variance needs to be of type float")
            print("attempt to solve directly for delta assuming no sigmas")
            sigma_sq['device'] = self._solve_max_likelihood_init(solver, subset_lambdas= A, solver_opts = solver_opts, tee=tee)
            print(sigma_sq['device'])
            delta = sigma_sq['device']
            
            
            
            if direct_or_it in ["it"]:
                print("Now that we know the maximum delta value, we can solve from there within a range")
                device_range.append(delta*0.00001)
                device_range.append(delta)

                dist = abs((device_range[1] - device_range[0])/num_points)
                
                max_likelihood_vals = []
                delta_vals = []
                iteration_counter = []
                delta = device_range[0]
                residual_vals = []
                #max_likelihood_val, sigma_vals, stop_it= self._solve_sigma_given_delta(solver, subset_lambdas= A, solver_opts = solver_opts, tee=tee,delta = delta)
                count = 0
                while delta < device_range[1]:
                    print("0000000000000000000000000000000000000000000000000000000")
                    print("000000000      ITERATION", count, "     0000000000000000")
                    print("solving for this value of delta:", delta)
                    print("0000000000000000000000000000000000000000000000000000000")
                    max_likelihood_val, sigma_vals, stop_it= self._solve_sigma_given_delta(solver, subset_lambdas= A, solver_opts = solver_opts, tee=tee,delta = delta)
                    if max_likelihood_val >= 5000:
                        max_likelihood_vals.append(5000)
                        delta_vals.append(log(delta))
                        iteration_counter.append(count)
                    else:
                        max_likelihood_vals.append(max_likelihood_val)
                        delta_vals.append(log(delta))
                        iteration_counter.append(count)
                        
                    if max_likelihood_val != 0:
                        sigma_vals['device'] = delta
                        residuals = self._solve_param_est(solver, solver_opts = solver_opts, tee=tee, variances = sigma_vals)
                        residual_vals.append(residuals)  
                    else:
                        residual_vals.append(0) 
                    print("objective function,iteration, delta",max_likelihood_val, count, delta )
                    delta = delta + dist
                    count += 1
                    
                #delta = self._solve_max_likelihood_init(solver, variances = sigmas, subset_lambdas= A, solver_opts = solver_opts, tee=tee)
                
                plt.plot(iteration_counter, max_likelihood_vals)
                plt.plot(iteration_counter, residuals)
                plt.show() 
                #sys.exit()
                #delta = 1e-11
                #sigmas = self._solve_phi_given_delta(solver, solver_opts = solver_opts, tee = tee, delta = delta)
                results = ResultsObject()
            
                # retriving solutions to results object  
                # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
                if hasattr(self, '_abs_components'):
                    results.load_from_pyomo_model(self.model,
                                              to_load=['Z', 'dZdt', 'X', 'dXdt', 'C', 'Cs', 'S', 'Y'])
                else:
                    results.load_from_pyomo_model(self.model,
                                                  to_load=['Z', 'dZdt', 'X', 'dXdt', 'C', 'S', 'Y'])
                
                param_vals = dict()
                for name in self.model.parameter_names:
                    param_vals[name] = self.model.P[name].value
        
                results.P = param_vals
                results.sigma_sq = sigma_vals
                results.sigma_sq['device'] = delta
            else:
                max_likelihood_val, sigma_vals, stop_it= self._solve_sigma_given_delta(solver, subset_lambdas= A, solver_opts = solver_opts, tee=tee,delta = fixed_device_var)
                # retriving solutions to results object  
                # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
                results = ResultsObject()
                if hasattr(self, '_abs_components'):
                    results.load_from_pyomo_model(self.model,
                                              to_load=['Z', 'dZdt', 'X', 'dXdt', 'C', 'Cs', 'S', 'Y'])
                else:
                    results.load_from_pyomo_model(self.model,
                                                  to_load=['Z', 'dZdt', 'X', 'dXdt', 'C', 'S', 'Y'])
                
                param_vals = dict()
                for name in self.model.parameter_names:
                    param_vals[name] = self.model.P[name].value
        
                results.P = param_vals
                results.sigma_sq = sigma_vals
                results.sigma_sq['device'] = delta
        else:
            print("Solving for variances using the iterative method")
            all_sigma_specified = True
            
            keys = init_sigmas.keys()
            for k in list_components:
                if k not in keys:
                    all_sigma_specified = False
                    init_sigmas[k] = max(init_sigmas.values())
    
            if not 'device' in init_sigmas.keys():
                all_sigma_specified = False
                init_sigmas['device'] = 1.0
            
            # First we initialize the problem by solving with our fixed variances
            #self._solve_iterative_init(solver, variances = init_sigmas, solver_options = solver_opts, tee = tee)
            
            # Then we get delta with fixed variances
            #delta = self._solve_delta_given_sigma(solver, solver_opts = solver_opts, tee = tee, variances = init_sigmas)
            delta = init_sigmas['device']
            print(init_sigmas)
            new_sigmas = init_sigmas
            
            for k in list_components:
                new_sigmas[k] = new_sigmas[k]*100
                
            #This part gets a rough scale
            residuals = dict()
            sigmas_it = dict()
            
            for i in range(10):
                print(i)
                print(new_sigmas)
                sigmas_it[i] = dict()
                for k in list_components:
                    sigmas_it[i][k] = new_sigmas[k]
                sigmas_it[i]['device'] = delta
                print(sigmas_it)
                print(sigmas_it[i])
                residuals[i] = self._solve_Psi_given_var(solver, solver_opts = solver_opts, tee = tee, variances = new_sigmas)
                if i != 0:
                    if residuals[i] >= residuals[i-1]*2:
                        print("we have found the right scale of sigmas")
                        break
                for k in list_components:
                    new_sigmas[k] = new_sigmas[k]/4
            print(residuals)
            print(sigmas_it)
            residuals_2 = dict()
            sigma_val_it = dict()
            count=0
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            # and then here we solve the parametric problem
            for k in list_components:
                print(k)
                orig_val = sigmas_it[i-1][k]
                sigmas_it[i-1][k] = sigmas_it[i-1][k]/20
                
                for j in range(22):
                    print(sigmas_it[i-1])                    
                    residuals_2[k,count] = self._solve_Psi_given_var(solver, solver_opts = solver_opts, tee = tee, variances = sigmas_it[i-1])
                    print(residuals_2)
                    print(j,count)
                    if count != 0:
                        if residuals_2[k,count] > residuals_2[k,count-1]:
                            print("we have found the right scale of sigmas,", k)
                            break
                    if j==29:
                        break
                    else:
                        sigma_val_it[k, count] = sigmas_it[i-1][k]
                        sigmas_it[i-1][k] = sigmas_it[i-1][k]*1.4
                    count+=1
                    print(count,j, (count+j))
                print("end of the search for sigma_sq", k)
                sigmas_it[i-1][k] = orig_val
                print(residuals_2)
                count = 0
            print(sigmas_it[i-1])
            print(residuals_2)
            print(sigma_val_it)
            plot_dicta = dict()
            plot_dictb = dict()
            plot_dictc = dict()
            for a,b in residuals_2.items():
                print(a,b)
                print(a[0],a[1])
                print(sigma_val_it[a[0],a[1]])
                if a[0] == 'A':
                    plot_dicta[b] = sigma_val_it['A',a[1]]
                elif a[0] == 'B':
                    plot_dictb[b] = sigma_val_it['B',a[1]]   
                elif a[0] == 'C':
                    plot_dictc[b] = sigma_val_it['C',a[1]] 
            
            lists = sorted(plot_dicta.items()) # sorted by key, return a list of tuples

            x, y = zip(*lists) # unpack a list of pairs into two tuples

            plt.plot(y,x)
            plt.show() 
            lists = sorted(plot_dictb.items()) # sorted by key, return a list of tuples

            x, y = zip(*lists) # unpack a list of pairs into two tuples

            plt.plot(y,x)
            plt.show()
            lists = sorted(plot_dictc.items()) # sorted by key, return a list of tuples

            x, y = zip(*lists) # unpack a list of pairs into two tuples

            plt.plot(y,x)
            plt.show()
            #This part then looks to find a better idea of which species should be 
            
            #sigmas = self._solve_phi_given_delta(solver, solver_opts = solver_opts, tee = tee, delta = delta)
            #sigmas = self._solve_sigma_given_delta(solver, solver_opts = solver_opts, tee = tee, sigmas = init_sigmas)
            #sys.exit()
            results = ResultsObject()
            
            # retriving solutions to results object  
            # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
            if hasattr(self, '_abs_components'):
                results.load_from_pyomo_model(self.model,
                                          to_load=['Z', 'dZdt', 'X', 'dXdt', 'C', 'Cs', 'S', 'Y'])
            else:
                results.load_from_pyomo_model(self.model,
                                              to_load=['Z', 'dZdt', 'X', 'dXdt', 'C', 'S', 'Y'])
            
            param_vals = dict()
            for name in self.model.parameter_names:
                param_vals[name] = self.model.P[name].value
    
            results.P = param_vals
            
        if report_time:
            end = time.time()
            print("Total execution time in seconds for variance estimation:", end - start)
        
        return results

    def _solve_initalization(self, solver, **kwds):
        """Solves formulation 19 in weifengs paper

           This method is not intended to be used by users directly

        Args:
            sigma_sq (dict): variances 
        
            tee (bool,optional): flag to tell the optimizer whether to stream output
            to the terminal or not
        
            profile_time (bool,optional): flag to tell pyomo to time the construction and solution of the model. 
            Default False
        
            subset_lambdas (array_like,optional): Set of wavelengths to used in initialization problem 
            (Weifeng paper). Default all wavelengths.

        Returns:

            None

        """
        solver_opts = kwds.pop('solver_opts', dict())
        tee = kwds.pop('tee', True)
        set_A = kwds.pop('subset_lambdas', list())
        profile_time = kwds.pop('profile_time', False)
        sigmas_sq = kwds.pop('variances', dict())

        if not set_A:
            set_A = self._meas_lambdas
        
        keys = sigmas_sq.keys()
        # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
        if hasattr(self, '_abs_components'):
            for k in self._abs_components:
                if k not in keys:
                    sigmas_sq[k] = 0.0
        else:
            for k in self._sublist_components:
                if k not in keys:
                    sigmas_sq[k] = 0.0

        print("Solving Initialization Problem\n")
        
        # Need to check whether we have negative values in the D-matrix so that we do not have
        #non-negativity on S

        for t in self._meas_times:
            for l in self._meas_lambdas:
                if self.model.D[t, l] >= 0:
                    pass
                else:
                    self._is_D_deriv = True
        if self._is_D_deriv == True:
            print("Warning! Since D-matrix contains negative values Kipet is relaxing non-negativity on S")

        # the ones that are not in set_A are going to be stale and wont go to the optimizer
        
        # build objective
        obj = 0.0
        # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
        if hasattr(self, '_abs_components'):
            for t in self._meas_times:
                for l in set_A:
                    D_bar = sum(self.model.Z[t, k] * self.model.S[l, k] for k in self._abs_components)
                    obj += (self.model.D[t, l] - D_bar) ** 2
        else:
            for t in self._meas_times:
                for l in set_A:
                    D_bar = sum(self.model.Z[t, k] * self.model.S[l, k] for k in self._sublist_components)
                    obj += (self.model.D[t, l] - D_bar) ** 2
        self.model.init_objective = Objective(expr=obj)
        
        opt = SolverFactory(solver)

        for key, val in solver_opts.items():
            opt.options[key]=val
        solver_results = opt.solve(self.model,
                                   tee=tee,
                                   report_timing=profile_time)

        for t in self._meas_times:
            for k in self._mixture_components:
                if k in sigmas_sq and sigmas_sq[k] > 0.0:
                    self.model.C[t, k].value = np.random.normal(self.model.Z[t, k].value, sigmas_sq[k])
                else:
                    self.model.C[t, k].value = self.model.Z[t, k].value
                
        self.model.del_component('init_objective')

    
    def _solve_max_likelihood_init(self, solver, **kwds):
        """Solves the maximum likelihood initialization with (C-Z) = 0 and delta as a variable

           This method is not intended to be used by users directly

        Args:
            sigma_sq (dict): variances 
        
            tee (bool,optional): flag to tell the optimizer whether to stream output
            to the terminal or not
        
            profile_time (bool,optional): flag to tell pyomo to time the construction and solution of the model. 
            Default False
        
            subset_lambdas (array_like,optional): Set of wavelengths to used in initialization problem 
            (Weifeng paper). Default all wavelengths.

        Returns:

            None

        """   
        solver_opts = kwds.pop('solver_opts', dict())
        tee = kwds.pop('tee', True)
        set_A = kwds.pop('subset_lambdas', list())
        profile_time = kwds.pop('profile_time', False)
        sigmas_sq = kwds.pop('variances', dict())

        if not set_A:
            set_A = self._meas_lambdas
        
        keys = sigmas_sq.keys()
        # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
        if hasattr(self, '_abs_components'):
            for k in self._abs_components:
                if k not in keys:
                    sigmas_sq[k] = 0.0
        else:
            for k in self._sublist_components:
                if k not in keys:
                    sigmas_sq[k] = 0.0

        print("Solving Initialization Problem\n")
        
        # Need to check whether we have negative values in the D-matrix so that we do not have
        #non-negativity on S

        for t in self._meas_times:
            for l in self._meas_lambdas:
                if self.model.D[t, l] >= 0:
                    pass
                else:
                    self._is_D_deriv = True
        if self._is_D_deriv == True:
            print("Warning! Since D-matrix contains negative values Kipet is relaxing non-negativity on S")
        '''   
        def D_SZTranspose(m, t, l, set_A, countt, countl):
            #print(m,t,l)
            countt1 = 0
            countl1 = 0
            DSZ = dict()
            for i in m._meas_times:
                #print(i)
                for j in set_A:
                    #print(j)
                    #Make sure this is ok for abs and non-abs
                    D_bar = sum(m.model.Z[i, k] * m.model.S[j, k] for k in m._sublist_components)
                    DSZ[j,i] = m.model.D[i, j] - D_bar
                    countl1 += 1
                countl1 = 0
                countt1 += 1
            #print(countl1,countt1,countt,countl,j,i,t,l)
            DSZTel = DSZ[l, t]
            return DSZTel
        '''
        self.DSZ = dict()
        
        def D_SZTranspose(m, t, l, set_A, countt, countl, absorb):
            #print(m,t,l)
            countt1 = 0
            countl1 = 0
            if absorb:
                D_bar = sum(m.model.Z[t, k] * m.model.S[l, k] for k in m._sublist_components)
            else:
                D_bar = sum(m.model.Z[t, k] * m.model.S[l, k] for k in m._sublist_components)
            self.DSZ[l,t] = m.model.D[t, l] - D_bar
            countl1 += 1
            countl1 = 0
            countt1 += 1
            #print(countl1,countt1,countt,countl,j,i,t,l)
            DSZTel = self.DSZ[l, t]
            return DSZTel
        # build objective
        
        
        obj = 0.0
        # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
        countt = 0
        countl = 0
        ntp = len(self._meas_times)
        if hasattr(self, '_abs_components'):
            for t in self._meas_times:
                for l in set_A:
                    D_bar = sum(self.model.Z[t, k] * self.model.S[l, k] for k in self._abs_components)
                    DSZTransp = D_SZTranspose(self, t, l,set_A, countt,countl, absorb = True)
                    obj += (self.model.D[t, l] - D_bar)**2
                    countl += 1
                countt += 1
        else:
            for t in self._meas_times:
                for l in set_A:
                    D_bar = sum(self.model.Z[t, k] * self.model.S[l, k] for k in self._sublist_components)
                    DSZTransp = D_SZTranspose(self, t, l, set_A, countt,countl, absorb = False)
                    obj += (self.model.D[t, l] - D_bar)**2
                    countl += 1
                countt += 1                    
        newobj = ntp/2*log(obj/ntp)            
        self.model.init_objective = Objective(expr=newobj)
        
        opt = SolverFactory(solver)

        for key, val in solver_opts.items():
            opt.options[key]=val
        solver_results = opt.solve(self.model,
                                   tee=tee,
                                   report_timing=profile_time)
        
        delta = (value(self.model.init_objective))
        for k,v in six.iteritems(self.model.P):
            print(k, v)
            print(v.value)
        
        etaTeta = 0
        if hasattr(self, '_abs_components'):
            for t in self._meas_times:
                for l in set_A:
                    D_bar = sum(value(self.model.Z[t, k]) * value(self.model.S[l, k]) for k in self._abs_components)
                    #DSZTransp = D_SZTranspose(self, t, l,set_A, countt,countl, absorb = True)
                    etaTeta += (value(self.model.D[t, l])- D_bar)**2
                    countl += 1
                countt += 1
        else:
            for t in self._meas_times:
                for l in set_A:
                    D_bar = sum(value(self.model.Z[t, k]) * value(self.model.S[l, k]) for k in self._sublist_components)
                    #DSZTransp = D_SZTranspose(self, t, l, set_A, countt,countl, absorb = False)
                    etaTeta += (value(self.model.D[t, l]) - D_bar)**2
                    countl += 1
                countt += 1  
        deltasq = etaTeta/ntp
        print("the real deltasq?", deltasq)
        for t in self._meas_times:
            for k in self._mixture_components:
                if k in sigmas_sq and sigmas_sq[k] > 0.0:
                    self.model.C[t, k].value = np.random.normal(self.model.Z[t, k].value, sigmas_sq[k])
                else:
                    self.model.C[t, k].value = self.model.Z[t, k].value
                
        self.model.del_component('init_objective')
        print("delta squared: ", delta**2)
        
        print(delta/ntp)
        print((delta/ntp)**2)
        return deltasq

    def _solve_max_likelihood(self, solver, **kwds):
        """Solves the maximum likelihood initialization with (C-Z) = 0 and delta as a variable

           This method is not intended to be used by users directly

        Args:
            sigma_sq (dict): variances 
        
            tee (bool,optional): flag to tell the optimizer whether to stream output
            to the terminal or not
        
            profile_time (bool,optional): flag to tell pyomo to time the construction and solution of the model. 
            Default False
        
            subset_lambdas (array_like,optional): Set of wavelengths to used in initialization problem 
            (Weifeng paper). Default all wavelengths.

        Returns:

            None

        """   
        solver_opts = kwds.pop('solver_opts', dict())
        tee = kwds.pop('tee', True)
        set_A = kwds.pop('subset_lambdas', list())
        profile_time = kwds.pop('profile_time', False)
        sigmas_sq = kwds.pop('variances', dict())

        if not set_A:
            set_A = self._meas_lambdas
        
        keys = sigmas_sq.keys()
        # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
        if hasattr(self, '_abs_components'):
            for k in self._abs_components:
                if k not in keys:
                    sigmas_sq[k] = 0.0
        else:
            for k in self._sublist_components:
                if k not in keys:
                    sigmas_sq[k] = 0.0

        print("Solving Initialization Problem\n")
        
        # Need to check whether we have negative values in the D-matrix so that we do not have
        #non-negativity on S

        for t in self._meas_times:
            for l in self._meas_lambdas:
                if self.model.D[t, l] >= 0:
                    pass
                else:
                    self._is_D_deriv = True
        if self._is_D_deriv == True:
            print("Warning! Since D-matrix contains negative values Kipet is relaxing non-negativity on S")
        
        # First we form the variables in the Lower triangular matrix for 
            
            
            
    def _solve_Psi_given_var(self, solver, **kwds):
        """Solves the maximum likelihood initialization with (C-Z) = 0 and delta as a variable

           This method is not intended to be used by users directly

        Args:
            sigma_sq (dict): variances 
        
            tee (bool,optional): flag to tell the optimizer whether to stream output
            to the terminal or not
        
            profile_time (bool,optional): flag to tell pyomo to time the construction and solution of the model. 
            Default False
        
            subset_lambdas (array_like,optional): Set of wavelengths to used in initialization problem 
            (Weifeng paper). Default all wavelengths.

        Returns:

            None

        """   
        solver_opts = kwds.pop('solver_opts', dict())
        tee = kwds.pop('tee', True)
        set_A = kwds.pop('subset_lambdas', list())
        profile_time = kwds.pop('profile_time', False)
        sigmas_sq = kwds.pop('variances', dict())

        if not set_A:
            set_A = self._meas_lambdas
        
        keys = sigmas_sq.keys()
        # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
        if hasattr(self, '_abs_components'):
            for k in self._abs_components:
                if k not in keys:
                    sigmas_sq[k] = 0.0
        else:
            for k in self._sublist_components:
                if k not in keys:
                    sigmas_sq[k] = 0.0

        print("Solving delta from given sigmas\n")
        
        # Need to check whether we have negative values in the D-matrix so that we do not have
        #non-negativity on S

        for t in self._meas_times:
            for l in self._meas_lambdas:
                if self.model.D[t, l] >= 0:
                    pass
                else:
                    self._is_D_deriv = True
        if self._is_D_deriv == True:
            print("Warning! Since D-matrix contains negative values Kipet is relaxing non-negativity on S")

        self.DSZ = dict()
        self.CZtransp = dict()
        
        def D_SZTranspose(m, t, l, set_A, countt, countl, absorb):
            if absorb:
                D_bar = sum(m.model.C[t, k] * m.model.S[l, k] for k in self._abs_components)
            else:
                D_bar = sum(m.model.C[t, k] * m.model.S[l, k] for k in m._sublist_components)
            self.DSZ[l,t] = m.model.D[t, l] - D_bar
            DSZTel = self.DSZ[l, t]
            return DSZTel
        # build objective
        def CZTranspose(m, t, k, set_A, countt, countl, absorb):
            if absorb:
                CZtransp = sum(m.model.C[t, k] - m.model.Z[t, k] for k in self._abs_components)
            else:
                CZtransp = sum(m.model.C[t, k] - m.model.Z[t, k] for k in m._sublist_components)
            self.CZtransp[k,t] = CZtransp
            CZT_el = self.CZtransp[k,t] 
            return CZT_el        
        
        obj = 0.0
        # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
        countt = 0
        countl = 0
        
        if hasattr(self, '_abs_components'):
            for t in self._meas_times:
                for l in set_A:
                    D_bar = sum(self.model.C[t, k] * self.model.S[l, k] for k in self._abs_components)
                    DSZTransp = D_SZTranspose(self, t, l,set_A, countt,countl, absorb = True)
                    obj += 0.5*(self.model.D[t, l] - D_bar)**2/sigmas_sq['device']
                    countl += 1
                countt += 1
        else:
            for t in self._meas_times:
                for l in set_A:
                    D_bar = sum(self.model.C[t, k] * self.model.S[l, k] for k in self._sublist_components)
                    DSZTransp = D_SZTranspose(self, t, l, set_A, countt,countl, absorb = False)
                    obj += 0.5*(self.model.D[t, l] - D_bar)**2/sigmas_sq['device']
                    countl += 1
                countt += 1                    
        
        for t in self._meas_times:
            for k in self._sublist_components:
                CZtransp = CZTranspose(self, t, k, set_A, countt,countl, absorb = False)
                obj += 0.5*((self.model.C[t, k] - self.model.Z[t, k])**2)/sigmas_sq[k]
                        
        self.model.init_objective = Objective(expr=obj)
        
        opt = SolverFactory(solver)

        for key, val in solver_opts.items():
            opt.options[key]=val
        solver_results = opt.solve(self.model,
                                   tee=tee,
                                   report_timing=profile_time)
        residuals = (value(self.model.init_objective))
        print(sigmas_sq['device'])
        print(sigmas_sq)
        print("residuals: ", residuals)
        epsilonTepsilon = 0
        for t in self._meas_times:
            for k in self._sublist_components:
                CZtransp = CZTranspose(self, t, k, set_A, countt,countl, absorb = False)
                epsilonTepsilon += 0.5*((value(self.model.C[t, k]) - value(self.model.Z[t, k]))**2)/sigmas_sq[k]
        etaTeta = 0
        if hasattr(self, '_abs_components'):
            for t in self._meas_times:
                for l in set_A:
                    D_bar = sum(value(self.model.C[t, k]) * value(self.model.S[l, k]) for k in self._abs_components)
                    #DSZTransp = D_SZTranspose(self, t, l,set_A, countt,countl, absorb = True)
                    etaTeta += 0.5*(value(self.model.D[t, l])- D_bar)**2/sigmas_sq['device']
                    countl += 1
                countt += 1
        else:
            for t in self._meas_times:
                for l in set_A:
                    D_bar = sum(value(self.model.C[t, k]) * value(self.model.S[l, k]) for k in self._sublist_components)
                    #DSZTransp = D_SZTranspose(self, t, l, set_A, countt,countl, absorb = False)
                    etaTeta += 0.5*(value(self.model.D[t, l]) - D_bar)**2/sigmas_sq['device']
                    countl += 1
                countt += 1 
        print("D - CS: etaTeta :", etaTeta)
        print("C-Z: epsilonTepsilon :", epsilonTepsilon)
        #print(value(self.model.deltasq_inv))
        #print(sqrt(1/value(self.model.deltasq_inv)))
        for k,v in six.iteritems(self.model.P):
            print(k, v)
            print(v.value)
        results = ResultsObject()
            
            # retriving solutions to results object  
            # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
        if hasattr(self, '_abs_components'):
            results.load_from_pyomo_model(self.model,
                                          to_load=['Z', 'dZdt', 'X', 'dXdt', 'C', 'Cs', 'S', 'Y'])
        else:
            results.load_from_pyomo_model(self.model,
                                              to_load=['Z', 'dZdt', 'X', 'dXdt', 'C', 'S', 'Y'])                
        self.model.del_component('init_objective')
        #self.model.del_component('init_objective')  
        results.C.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile")

        results.S.plot.line(legend=True)
        plt.xlabel("Wavelength (cm)")
        plt.ylabel("Absorbance (L/(mol cm))")
        plt.title("Absorbance  Profile")
        results.Z.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile")
   
        plt.show() 
        return residuals
    
    def _solve_param_est(self, solver, **kwds):
        """Solves the maximum likelihood initialization with (C-Z) = 0 and delta as a variable

           This method is not intended to be used by users directly

        Args:
            sigma_sq (dict): variances 
        
            tee (bool,optional): flag to tell the optimizer whether to stream output
            to the terminal or not
        
            profile_time (bool,optional): flag to tell pyomo to time the construction and solution of the model. 
            Default False
        
            subset_lambdas (array_like,optional): Set of wavelengths to used in initialization problem 
            (Weifeng paper). Default all wavelengths.

        Returns:

            None

        """   
        solver_opts = kwds.pop('solver_opts', dict())
        tee = kwds.pop('tee', True)
        set_A = kwds.pop('subset_lambdas', list())
        profile_time = kwds.pop('profile_time', False)
        sigmas_sq = kwds.pop('variances', dict())

        if not set_A:
            set_A = self._meas_lambdas
        
        keys = sigmas_sq.keys()
        # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
        if hasattr(self, '_abs_components'):
            for k in self._abs_components:
                if k not in keys:
                    sigmas_sq[k] = 0.0
        else:
            for k in self._sublist_components:
                if k not in keys:
                    sigmas_sq[k] = 0.0

        print("Solving pramaeter estimation problem")
        
        # Need to check whether we have negative values in the D-matrix so that we do not have
        #non-negativity on S

        for t in self._meas_times:
            for l in self._meas_lambdas:
                if self.model.D[t, l] >= 0:
                    pass
                else:
                    self._is_D_deriv = True
        if self._is_D_deriv == True:
            print("Warning! Since D-matrix contains negative values Kipet is relaxing non-negativity on S")

    
        
        obj = 0.0
        # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
        countt = 0
        countl = 0
        
        if hasattr(self, '_abs_components'):
            for t in self._meas_times:
                for l in set_A:
                    D_bar = sum(self.model.C[t, k] * self.model.S[l, k] for k in self._abs_components)
                    obj += (self.model.D[t, l] - D_bar)**2/sigmas_sq['device']
                    countl += 1
                countt += 1
        else:
            for t in self._meas_times:
                for l in set_A:
                    D_bar = sum(self.model.C[t, k] * self.model.S[l, k] for k in self._sublist_components)
                    obj += (self.model.D[t, l] - D_bar)**2/sigmas_sq['device']
                    countl += 1
                countt += 1                    
        
        for t in self._meas_times:
            for k in self._sublist_components:
                obj += ((self.model.C[t, k] - self.model.Z[t, k])**2)/sigmas_sq[k]
                        
        self.model.init_objective = Objective(expr=obj)
        
        opt = SolverFactory(solver)

        for key, val in solver_opts.items():
            opt.options[key]=val
        solver_results = opt.solve(self.model,
                                   tee=tee,
                                   report_timing=profile_time)
        
        residuals = (value(self.model.init_objective))
        print("residuals: ", residuals)


        for k,v in six.iteritems(self.model.P):
            print(k, v)
            print(v.value)
        results = ResultsObject()
            
            # retriving solutions to results object  
            # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
        if hasattr(self, '_abs_components'):
            results.load_from_pyomo_model(self.model,
                                          to_load=['Z', 'dZdt', 'X', 'dXdt', 'C', 'Cs', 'S', 'Y'])
        else:
            results.load_from_pyomo_model(self.model,
                                              to_load=['Z', 'dZdt', 'X', 'dXdt', 'C', 'S', 'Y'])                
        self.model.del_component('init_objective')
        #self.model.del_component('init_objective')  
        '''
        results.C.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile")

        results.S.plot.line(legend=True)
        plt.xlabel("Wavelength (cm)")
        plt.ylabel("Absorbance (L/(mol cm))")
        plt.title("Absorbance  Profile")
        results.Z.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile")
   
        plt.show() 
        '''
        return residuals
    
    def _solve_delta_given_sigma(self, solver, **kwds):
        """Solves the maximum likelihood initialization with (C-Z) = 0 and delta as a variable

           This method is not intended to be used by users directly

        Args:
            sigma_sq (dict): variances 
        
            tee (bool,optional): flag to tell the optimizer whether to stream output
            to the terminal or not
        
            profile_time (bool,optional): flag to tell pyomo to time the construction and solution of the model. 
            Default False
        
            subset_lambdas (array_like,optional): Set of wavelengths to used in initialization problem 
            (Weifeng paper). Default all wavelengths.

        Returns:

            None

        """   
        solver_opts = kwds.pop('solver_opts', dict())
        tee = kwds.pop('tee', True)
        set_A = kwds.pop('subset_lambdas', list())
        profile_time = kwds.pop('profile_time', False)
        sigmas_sq = kwds.pop('variances', dict())

        if not set_A:
            set_A = self._meas_lambdas
        
        keys = sigmas_sq.keys()
        # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
        if hasattr(self, '_abs_components'):
            for k in self._abs_components:
                if k not in keys:
                    sigmas_sq[k] = 0.0
        else:
            for k in self._sublist_components:
                if k not in keys:
                    sigmas_sq[k] = 0.0

        print("Solving delta from given sigmas\n")
        
        # Need to check whether we have negative values in the D-matrix so that we do not have
        #non-negativity on S

        for t in self._meas_times:
            for l in self._meas_lambdas:
                if self.model.D[t, l] >= 0:
                    pass
                else:
                    self._is_D_deriv = True
        if self._is_D_deriv == True:
            print("Warning! Since D-matrix contains negative values Kipet is relaxing non-negativity on S")

        self.DSZ = dict()
        self.CZtransp = dict()
        
        def D_SZTranspose(m, t, l, set_A, countt, countl, absorb):
            if absorb:
                D_bar = sum(m.model.C[t, k] * m.model.S[l, k] for k in self._abs_components)
            else:
                D_bar = sum(m.model.C[t, k] * m.model.S[l, k] for k in m._sublist_components)
            self.DSZ[l,t] = m.model.D[t, l] - D_bar
            DSZTel = self.DSZ[l, t]
            return DSZTel
        # build objective
        def CZTranspose(m, t, k, set_A, countt, countl, absorb):
            if absorb:
                CZtransp = sum(m.model.C[t, k] - m.model.Z[t, k] for k in self._abs_components)
            else:
                CZtransp = sum(m.model.C[t, k] - m.model.Z[t, k] for k in m._sublist_components)
            self.CZtransp[k,t] = CZtransp
            CZT_el = self.CZtransp[k,t] 
            return CZT_el        
        
        obj = 0.0
        # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
        countt = 0
        countl = 0
        self.model.deltasq_inv = Var(within = NonNegativeReals)
        if hasattr(self, '_abs_components'):
            for t in self._meas_times:
                for l in set_A:
                    D_bar = sum(self.model.C[t, k] * self.model.S[l, k] for k in self._abs_components)
                    DSZTransp = D_SZTranspose(self, t, l,set_A, countt,countl, absorb = True)
                    obj += self.model.deltasq_inv*0.5*(self.model.D[t, l] - D_bar)*DSZTransp
                    countl += 1
                countt += 1
        else:
            for t in self._meas_times:
                for l in set_A:
                    D_bar = sum(self.model.C[t, k] * self.model.S[l, k] for k in self._sublist_components)
                    DSZTransp = D_SZTranspose(self, t, l, set_A, countt,countl, absorb = False)
                    obj += self.model.deltasq_inv*0.5*(self.model.D[t, l] - D_bar)*DSZTransp
                    countl += 1
                countt += 1                    
        
        for t in self._meas_times:
            for k in self._sublist_components:
                CZtransp = CZTranspose(self, t, k, set_A, countt,countl, absorb = False)
                obj += 0.5*((self.model.C[t, k] - self.model.Z[t, k])*CZtransp)/sigmas_sq[k]
                        
        self.model.init_objective = Objective(expr=obj)
        
        opt = SolverFactory(solver)

        for key, val in solver_opts.items():
            opt.options[key]=val
        solver_results = opt.solve(self.model,
                                   tee=tee,
                                   report_timing=profile_time)
        residuals = (value(self.model.init_objective))
        
        print("residuals: ", residuals)
        print(value(self.model.deltasq_inv))
        print(sqrt(1/value(self.model.deltasq_inv)))
        for k,v in six.iteritems(self.model.P):
            print(k, v)
            print(v.value)
                
        self.model.del_component('init_objective')
        #self.model.del_component('init_objective')        
        return 1/value(self.model.deltasq_inv)
    
    def _solve_sigma_given_delta(self, solver, **kwds):
        """Solves the maximum likelihood initialization with (C-Z) = 0 and delta as a variable

           This method is not intended to be used by users directly

        Args:
            sigma_sq (dict): variances 
        
            tee (bool,optional): flag to tell the optimizer whether to stream output
            to the terminal or not
        
            profile_time (bool,optional): flag to tell pyomo to time the construction and solution of the model. 
            Default False
        
            subset_lambdas (array_like,optional): Set of wavelengths to used in initialization problem 
            (Weifeng paper). Default all wavelengths.

        Returns:

            None

        """   
        solver_opts = kwds.pop('solver_opts', dict())
        tee = kwds.pop('tee', True)
        set_A = kwds.pop('subset_lambdas', list())
        profile_time = kwds.pop('profile_time', False)
        sigmas_sq = kwds.pop('variances', dict())
        delta = kwds.pop('delta', dict())
        #sigmas = kwds.pop('sigmas', dict())

        if not set_A:
            print("subset lambdas not selected. Solving full problem.")
            set_A = self._meas_lambdas
        
        keys = sigmas_sq.keys()
        # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
        if hasattr(self, '_abs_components'):
            for k in self._abs_components:
                if k not in keys:
                    sigmas_sq[k] = 0.0
        else:
            for k in self._sublist_components:
                if k not in keys:
                    sigmas_sq[k] = 0.0

        print("Solving sigmas from the given delta\n")
        
        # Need to check whether we have negative values in the D-matrix so that we do not have
        #non-negativity on S

        for t in self._meas_times:
            for l in self._meas_lambdas:
                if self.model.D[t, l] >= 0:
                    pass
                else:
                    self._is_D_deriv = True
        if self._is_D_deriv == True:
            print("Warning! Since D-matrix contains negative values Kipet is relaxing non-negativity on S")
        ntp = len(self._meas_times) 
        self.DSZ = dict()
        self.CZtransp = dict()
        
        def D_SZTranspose(m, t, l, set_A, countt, countl, absorb):
            if absorb:
                D_bar = sum(m.model.C[t, k] * m.model.S[l, k] for k in self._abs_components)
            else:
                D_bar = sum(m.model.C[t, k] * m.model.S[l, k] for k in m._sublist_components)
            self.DSZ[l,t] = m.model.D[t, l] - D_bar
            DSZTel = self.DSZ[l, t]
            return DSZTel
        # build objective
        def CZTranspose(m, t, k, set_A, countt, countl, absorb):
            if absorb:
                CZtransp = sum(m.model.C[t, k] - m.model.Z[t, k] for k in self._abs_components)
            else:
                CZtransp = sum(m.model.C[t, k] - m.model.Z[t, k] for k in m._sublist_components)
            self.CZtransp[k,t] = CZtransp
            CZT_el = self.CZtransp[k,t] 
            return CZT_el        
        m = self.model.clone()
        obj = 0.0
        # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
        countt = 0
        countl = 0
        #sigma_magnitude = 1000
        #self.model.sigmas = Var(self.model.mixture_components, initialize = 0.3, bounds = (0.00001,1))
        #def sigm_ratio(m):
        #    return sum(m.sigmas[k] for k in self._sublist_components) == 1
            
        #self.model.sigma_ratio = Constraint(rule=sigm_ratio)    
        if hasattr(self, '_abs_components'):
            for t in self._meas_times:
                for l in set_A:
                    D_bar = sum(self.model.C[t, k] * self.model.S[l, k] for k in self._abs_components)
                    DSZTransp = D_SZTranspose(self, t, l,set_A, countt,countl, absorb = True)
                    obj += 1/(2*delta)*(self.model.D[t, l] - D_bar)**2
                    countl += 1
                countt += 1
        else:
            for t in self._meas_times:
                for l in set_A:
                    D_bar = sum(self.model.C[t, k] * self.model.S[l, k] for k in self._sublist_components)
                    DSZTransp = D_SZTranspose(self, t, l, set_A, countt,countl, absorb = False)
                    obj += 1/(2*delta)*(self.model.D[t, l] - D_bar)**2
                    countl += 1
                countt += 1 
        inlog = 0  
        self.model.eps = Param(initialize = 1e-8)                 
        variancesdict = dict()
        for k in self._sublist_components:
            variancesdict[k] = 0
        for t in self._meas_times:
            for k in self._sublist_components:
                CZtransp = CZTranspose(self, t, k, set_A, countt,countl, absorb = False)
                #obj += ((self.model.C[t, k] - self.model.Z[t, k])**2)
                inlog += ((self.model.C[t, k] - self.model.Z[t, k])**2)
                
        obj += (ntp/2)*log((inlog/ntp)+self.model.eps)

                        
        self.model.init_objective = Objective(expr=obj)
        
        opt = SolverFactory(solver)

        for key, val in solver_opts.items():
            opt.options[key]=val
        try:
            solver_results = opt.solve(self.model,
                                   tee=tee,
                                   report_timing=profile_time)
            residuals = (value(self.model.init_objective))
        
        
            print("objective function: ", residuals)
            CZobj = 0
            CZobj1 = 0
            for t in self._meas_times:
                for k in self._sublist_components:
                    #CZtransp = CZTranspose(self, t, k, set_A, countt,countl, absorb = False)
                    CZobj += ((value(self.model.C[t, k]) - value(self.model.Z[t, k]))**2)
                    CZobj1 += ((value(self.model.C[t, k]) - value(self.model.Z[t, k]))**2)
                    variancesdict[k] += 1/ntp*((value(self.model.C[t, k]) - value(self.model.Z[t, k]))**2)
            DCSobj = 0        
            for t in self._meas_times:
                for l in set_A:
                    D_bar = sum(value(self.model.C[t, k]) * value(self.model.S[l, k]) for k in self._sublist_components)
                    DSZTransp = D_SZTranspose(self, t, l, set_A, countt,countl, absorb = False)
                    DCSobj += 1/(2*delta)*(value(self.model.D[t, l]) - D_bar)**2              
            
            print("printing C-Z residuals without log",(CZobj))
            print("printing D-CS residuals (1/(2*delta)*(sum(D-CS))**2)",DCSobj)
            print("printing C-Z residuals with log",ntp/2*log(CZobj1/ntp))
            for k in self._sublist_components:
                print(k, variancesdict[k])
        
            for k,v in six.iteritems(self.model.P):
                print(k, v)
                print(v.value)
             
               
            res = 0
            for t in self._meas_times:
                for k in self._sublist_components:
                    #CZtransp = CZTranspose(self, t, k, set_A, countt,countl, absorb = False)
                    res += (value(self.model.C[t, k]) - value(self.model.Z[t, k]))**2
            max_likelihood_value = 0.5*ntp*log((res))
            pes = 0
            for t in self._meas_times:
                for l in set_A:
                    D_bar = sum(value(self.model.C[t, k]) * value(self.model.S[l, k]) for k in self._sublist_components)
                    pes += (value(self.model.D[t, l]) - D_bar)**2  
            pes = pes/(2*delta)
            max_likelihood_value1 = max_likelihood_value + pes
            #print('max_likelihood_value eta only',max_likelihood_value)
            #print('max_likelihood_value with epsilon no log',max_likelihood_value1)
            pes = 0
            for t in self._meas_times:
                for l in set_A:
                    D_bar = sum(value(self.model.C[t, k]) * value(self.model.S[l, k]) for k in self._sublist_components)
                    pes += (value(self.model.D[t, l]) - D_bar)**2  
            pes = 0.5*ntp*log((pes))
            max_likelihood_value2 = max_likelihood_value + pes
            #print('max_likelihood_value with log',max_likelihood_value2)
            stop_it = False
        except:
            print("no value found")
            max_likelihood_value2 = 0
            max_likelihood_value1 = 0
            variancesdict = None
            residuals = 0
            stop_it = True
            self.model = m
        self.model.del_component('init_objective')
        self.model.del_component('eps')
        #self.model.del_component('sigm_ratio')        
        #return residuals, variancesdict, stop_it
        return residuals, variancesdict, stop_it

    def _solve_sigma_given_delta_ratiovariable(self, solver, **kwds):
        """Solves the maximum likelihood initialization with (C-Z) = 0 and delta as a variable

           This method is not intended to be used by users directly

        Args:
            sigma_sq (dict): variances 
        
            tee (bool,optional): flag to tell the optimizer whether to stream output
            to the terminal or not
        
            profile_time (bool,optional): flag to tell pyomo to time the construction and solution of the model. 
            Default False
        
            subset_lambdas (array_like,optional): Set of wavelengths to used in initialization problem 
            (Weifeng paper). Default all wavelengths.

        Returns:

            None

        """   
        solver_opts = kwds.pop('solver_opts', dict())
        tee = kwds.pop('tee', True)
        set_A = kwds.pop('subset_lambdas', list())
        profile_time = kwds.pop('profile_time', False)
        sigmas_sq = kwds.pop('variances', dict())
        delta = kwds.pop('delta', dict())
        sigmas = kwds.pop('delta', dict())

        if not set_A:
            set_A = self._meas_lambdas
        
        keys = sigmas_sq.keys()
        # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
        if hasattr(self, '_abs_components'):
            for k in self._abs_components:
                if k not in keys:
                    sigmas_sq[k] = 0.0
        else:
            for k in self._sublist_components:
                if k not in keys:
                    sigmas_sq[k] = 0.0

        print("Solving sigmas from the given delta\n")
        
        # Need to check whether we have negative values in the D-matrix so that we do not have
        #non-negativity on S

        for t in self._meas_times:
            for l in self._meas_lambdas:
                if self.model.D[t, l] >= 0:
                    pass
                else:
                    self._is_D_deriv = True
        if self._is_D_deriv == True:
            print("Warning! Since D-matrix contains negative values Kipet is relaxing non-negativity on S")

        self.DSZ = dict()
        self.CZtransp = dict()
        
        def D_SZTranspose(m, t, l, set_A, countt, countl, absorb):
            if absorb:
                D_bar = sum(m.model.C[t, k] * m.model.S[l, k] for k in self._abs_components)
            else:
                D_bar = sum(m.model.C[t, k] * m.model.S[l, k] for k in m._sublist_components)
            self.DSZ[l,t] = m.model.D[t, l] - D_bar
            DSZTel = self.DSZ[l, t]
            return DSZTel
        # build objective
        def CZTranspose(m, t, k, set_A, countt, countl, absorb):
            if absorb:
                CZtransp = sum(m.model.C[t, k] - m.model.Z[t, k] for k in self._abs_components)
            else:
                CZtransp = sum(m.model.C[t, k] - m.model.Z[t, k] for k in m._sublist_components)
            self.CZtransp[k,t] = CZtransp
            CZT_el = self.CZtransp[k,t] 
            return CZT_el        
        
        obj = 0.0
        # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
        countt = 0
        countl = 0
        sigma_magnitude = 1000
        self.model.sigmas = Var(self.model.mixture_components, initialize = 0.3, bounds = (0.00001,1))
        def sigm_ratio(m):
            return sum(m.sigmas[k] for k in self._sublist_components) == 1
            
        self.model.sigma_ratio = Constraint(rule=sigm_ratio)    
        if hasattr(self, '_abs_components'):
            for t in self._meas_times:
                for l in set_A:
                    D_bar = sum(self.model.C[t, k] * self.model.S[l, k] for k in self._abs_components)
                    DSZTransp = D_SZTranspose(self, t, l,set_A, countt,countl, absorb = True)
                    obj += 1/(2*1)*(self.model.D[t, l] - D_bar)**2
                    countl += 1
                countt += 1
        else:
            for t in self._meas_times:
                for l in set_A:
                    D_bar = sum(self.model.C[t, k] * self.model.S[l, k] for k in self._sublist_components)
                    DSZTransp = D_SZTranspose(self, t, l, set_A, countt,countl, absorb = False)
                    obj += 1/(2*1)*(self.model.D[t, l] - D_bar)**2
                    countl += 1
                countt += 1                    
        
        for t in self._meas_times:
            for k in self._sublist_components:
                CZtransp = CZTranspose(self, t, k, set_A, countt,countl, absorb = False)
                obj += 0.5*((self.model.C[t, k] - self.model.Z[t, k])**2)*self.model.sigmas[k]*sigma_magnitude
                        
        self.model.init_objective = Objective(expr=obj)
        
        opt = SolverFactory(solver)

        for key, val in solver_opts.items():
            opt.options[key]=val
        solver_results = opt.solve(self.model,
                                   tee=tee,
                                   report_timing=profile_time)
        residuals = (value(self.model.init_objective))
        
        print("residuals: ", residuals)
        print(self.model.sigmas)
        for k in self._sublist_components:
            print(k, value(self.model.sigmas[k]))
            print(1/(sigma_magnitude*value(self.model.sigmas[k])))
            sigmas_sq[k] = 1/(sigma_magnitude*value(self.model.sigmas[k]))
        for k,v in six.iteritems(self.model.P):
            print(k, v)
            print(v.value)
                
        self.model.del_component('init_objective')
        self.model.del_component('sigm_ratio')        
        return sigmas_sq

    def _solve_phi_given_delta(self, solver, **kwds):
        """Solves the maximum likelihood initialization with (C-Z) = 0 and delta as a variable

           This method is not intended to be used by users directly

        Args:
            sigma_sq (dict): variances 
        
            tee (bool,optional): flag to tell the optimizer whether to stream output
            to the terminal or not
        
            profile_time (bool,optional): flag to tell pyomo to time the construction and solution of the model. 
            Default False
        
            subset_lambdas (array_like,optional): Set of wavelengths to used in initialization problem 
            (Weifeng paper). Default all wavelengths.

        Returns:

            None

        """   
        solver_opts = kwds.pop('solver_opts', dict())
        tee = kwds.pop('tee', True)
        set_A = kwds.pop('subset_lambdas', list())
        profile_time = kwds.pop('profile_time', False)
        sigmas_sq = kwds.pop('variances', dict())
        delta = kwds.pop('delta', float())

        if not set_A:
            set_A = self._meas_lambdas
        
        keys = sigmas_sq.keys()
        # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
        if hasattr(self, '_abs_components'):
            for k in self._abs_components:
                if k not in keys:
                    sigmas_sq[k] = 0.0
        else:
            for k in self._sublist_components:
                if k not in keys:
                    sigmas_sq[k] = 0.0

        print("Solving phi from the given delta\n")
        
        # Need to check whether we have negative values in the D-matrix so that we do not have
        #non-negativity on S

        for t in self._meas_times:
            for l in self._meas_lambdas:
                if self.model.D[t, l] >= 0:
                    pass
                else:
                    self._is_D_deriv = True
        if self._is_D_deriv == True:
            print("Warning! Since D-matrix contains negative values Kipet is relaxing non-negativity on S")

        self.DSZ = dict()
        self.CZtransp = dict()
        
        def D_SZTranspose(m, t, l, set_A, countt, countl, absorb):
            if absorb:
                D_bar = sum(m.model.C[t, k] * m.model.S[l, k] for k in self._abs_components)
            else:
                D_bar = sum(m.model.C[t, k] * m.model.S[l, k] for k in m._sublist_components)
            self.DSZ[l,t] = m.model.D[t, l] - D_bar
            DSZTel = self.DSZ[l, t]
            return DSZTel
        # build objective
        def CZTranspose(m, t, k, set_A, countt, countl, absorb):
            if absorb:
                CZtransp = sum(m.model.C[t, k] - m.model.Z[t, k] for k in self._abs_components)
            else:
                CZtransp = sum(m.model.C[t, k] - m.model.Z[t, k] for k in m._sublist_components)
            self.CZtransp[k,t] = CZtransp
            CZT_el = self.CZtransp[k,t] 
            return CZT_el        
        
        obj = 0.0
        # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
        countt = 0
        countl = 0
        ntp = len(self._meas_times)
        if hasattr(self, '_abs_components'):
            for t in self._meas_times:
                for l in set_A:
                    D_bar = sum(self.model.C[t, k] * self.model.S[l, k] for k in self._abs_components)
                    #DSZTransp = D_SZTranspose(self, t, l,set_A, countt,countl, absorb = True)
                    obj += 1/(2*delta)*((self.model.D[t, l] - D_bar)**2)
                    countl += 1
                countt += 1
        else:
            for t in self._meas_times:
                for l in set_A:
                    D_bar = sum(self.model.C[t, k] * self.model.S[l, k] for k in self._sublist_components)
                    #DSZTransp = D_SZTranspose(self, t, l, set_A, countt,countl, absorb = False)
                    obj += 1/(2*delta)*((self.model.D[t, l] - D_bar)**2)
                    countl += 1
                countt += 1                    
        res = 0
        for t in self._meas_times:
            for k in self._sublist_components:
                #CZtransp = CZTranspose(self, t, k, set_A, countt,countl, absorb = False)
                res += (self.model.C[t, k] - self.model.Z[t, k])**2
        obj += 0.5*ntp*log((res)/ntp)
                
                
        def non_zero_CZ_(m, t, k):

            return (m.C[t, k] - m.Z[t, k])**2 >= 0.000000000000001
                
        #self.model.non_zero_CZ = Constraint(self.model.meas_times, self.model.mixture_components, rule=non_zero_CZ_)

                 
        self.model.init_objective = Objective(expr=obj)
        
        opt = SolverFactory(solver)

        for key, val in solver_opts.items():
            opt.options[key]=val
        solver_results = opt.solve(self.model,
                                   tee=tee,
                                   report_timing=profile_time)
        residuals = (value(self.model.init_objective))
        
        print("residuals: ", residuals)
        
        for k,v in six.iteritems(self.model.P):
            print(k, v)
            print(v.value)
        sigma = dict()
        for k in self._sublist_components:
            CZtransp = CZTranspose(self, t, k, set_A, countt,countl, absorb = False)
            sigma[k] += sum(((self.model.C[t, k] - self.model.Z[t, k])**2)/ntp for t in self._meas_times)
        
        self.model.del_component('init_objective')
        print(sigma)        
        return sigma

    
    def _solve_iterative_init(self, solver, **kwds):
        """This method first fixes params and makes C = Z to solve for S. Requires fixed delta and sigmas
        following this, the parameters are freed to users bounds and full problem is solved with 

        Args:
            sigma_sq (dict): variances 
        
            tee (bool,optional): flag to tell the optimizer whether to stream output
            to the terminal or not
        
            profile_time (bool,optional): flag to tell pyomo to time the construction and solution of the model. 
            Default False
        
            subset_lambdas (array_like,optional): Set of wavelengths to used in initialization problem 
            (Weifeng paper). Default all wavelengths.

        Returns:

            None

        """   
        solver_opts = kwds.pop('solver_options', dict())
        tee = kwds.pop('tee', True)
        set_A = kwds.pop('subset_lambdas', list())
        profile_time = kwds.pop('profile_time', False)
        sigmas_sq = kwds.pop('variances', dict())

        if not set_A:
            set_A = self._meas_lambdas
        
        keys = sigmas_sq.keys()
        print(keys)
        # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
        if hasattr(self, '_abs_components'):
            for k in self._abs_components:
                print(k)
                if k not in keys:
                    sigmas_sq[k] = 0.0
        else:
            for k in self._sublist_components:
                if k not in keys:
                    sigmas_sq[k] = 0.0
        # Need to check whether we have negative values in the D-matrix so that we do not have
        #non-negativity on S

        for t in self._meas_times:
            for l in self._meas_lambdas:
                if self.model.D[t, l] >= 0:
                    pass
                else:
                    self._is_D_deriv = True
        if self._is_D_deriv == True:
            print("Warning! Since D-matrix contains negative values Kipet is relaxing non-negativity on S")
            
        list_components = []

        list_components = [k for k in self._mixture_components]
        print(sigmas_sq)                    
        print("Solving Initialization Problem with fixed parameters\n")
        original_bounds = dict()
        for v,k in six.iteritems(self.model.P):
            print(self.model.P[v].lb, self.model.P[v].ub)
            low = value(self.model.P[v].lb)
            high = value(self.model.P[v].ub)
            print(low,high)
            print(type(low))
            original_bounds[v] = (low, high)
            print(original_bounds)
            ub = value(self.model.P[v])
            lb = ub
            self.model.P[v].setlb(lb)
            self.model.P[v].setub(ub)
        # build objective
        obj = 0.0
        # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
        if hasattr(self, '_abs_components'):
            for t in self._meas_times:
                for l in set_A:
                    D_bar = sum(self.model.Z[t, k] * self.model.S[l, k] for k in self._abs_components)
                    obj += (self.model.D[t, l] - D_bar) ** 2
        else:
            for t in self._meas_times:
                for l in set_A:
                    D_bar = sum(self.model.Z[t, k] * self.model.S[l, k] for k in self._sublist_components)
                    obj += (self.model.D[t, l] - D_bar) ** 2
        self.model.init_objective = Objective(expr=obj)
        
        opt = SolverFactory(solver)

        for key, val in solver_opts.items():
            opt.options[key]=val
        solver_results = opt.solve(self.model,
                                   tee=tee,
                                   report_timing=profile_time)
        for k,v in six.iteritems(self.model.P):
            print(k, v)
            print(v.value)
        
        for t in self._meas_times:
            for k in self._mixture_components:
                if k in sigmas_sq and sigmas_sq[k] > 0.0:
                    self.model.C[t, k].value = np.random.normal(self.model.Z[t, k].value, sigmas_sq[k])
                else:
                    self.model.C[t, k].value = self.model.Z[t, k].value
                
        self.model.del_component('init_objective')
        
        for v,k in six.iteritems(self.model.P):
            #print(self.model.P[v].lb, self.model.P[v].ub)
            print(k,v)
            print(type(original_bounds[v][1]))
            ub = original_bounds[v][1]
            lb = original_bounds[v][0]
            self.model.P[v].setlb(lb)
            self.model.P[v].setub(ub)
            
        m = self.model

        m.D_bar = Var(m.meas_times,
                      m.meas_lambdas)
        # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
        if hasattr(self, '_abs_components'):
            def rule_D_bar(m, t, l):
                return m.D_bar[t, l] == sum(m.Cs[t, k] * m.S[l, k] for k in self._abs_components)
        else:
            def rule_D_bar(m, t, l):
                return m.D_bar[t, l] == sum(m.C[t, k] * m.S[l, k] for k in self._sublist_components)

        m.D_bar_constraint = Constraint(m.meas_times,
                                        m.meas_lambdas,
                                        rule=rule_D_bar)

        # estimation
        def rule_objective(m):
            expr = 0
            for t in m.meas_times:
                for l in m.meas_lambdas:

                    expr += (m.D[t, l] - m.D_bar[t, l]) ** 2 / (sigmas_sq['device'])

            second_term = 0.0
            for t in m.meas_times:
                second_term += sum((m.C[t, k] - m.Z[t, k]) ** 2 / sigmas_sq[k] for k in list_components)

            return expr

        m.objective = Objective(rule=rule_objective) 
        
        opt = SolverFactory(solver)

        for key, val in solver_opts.items():
            opt.options[key]=val
        solver_results = opt.solve(self.model,
                                   tee=tee,
                                   report_timing=profile_time)
        for k,v in six.iteritems(self.model.P):
            print(k, v)
            print(v.value)        
            print(value(v.ub))
            print(value(v.lb))
        
        m.del_component('objective')
        self.model = m
        
            
    def _solve_Z(self, solver, **kwds):
        """Solves formulation 20 in weifengs paper

           This method is not intended to be used by users directly

        Args:
        
            solver_opts (dict, optional): options passed to the nonlinear solver

            tee (bool,optional): flag to tell the optimizer whether to stream output
            to the terminal or not
        
            profile_time (bool,optional): flag to tell pyomo to time the construction and solution of the model. 
            Default False
        
            subset_lambdas (array_like,optional): Set of wavelengths to used in initialization problem 
            (Weifeng paper). Default all wavelengths.

        Returns:

            None

        """
        solver_opts = kwds.pop('solver_opts', dict())
        tee = kwds.pop('tee', False)
        profile_time = kwds.pop('profile_time', False)
        
        # assume this values were computed in beforehand
        for t in self._meas_times:
            for k in self._sublist_components:
                if hasattr(self.model, 'non_absorbing'):
                    if k in self.model.non_absorbing:
                        pass
                    else:
                        self.model.C[t, k].fixed = True

        obj = 0.0
        for k in self._sublist_components:
            x = sum((self.model.C[t, k]-self.model.Z[t, k])**2 for t in self._meas_times)
            obj += x

        self.model.z_objective = Objective(expr=obj)
        #self.model.z_objective.pprint()
        if profile_time:
            print('-----------------Solve_Z--------------------')
            
        opt = SolverFactory(solver)

        for key, val in solver_opts.items():
            opt.options[key]=val

        from pyomo.opt import ProblemFormat
        solver_results = opt.solve(self.model,
                                   logfile=self._tmp2,
                                   tee=tee,
                                   #show_section_timing=True,
                                   report_timing=profile_time)

        self.model.del_component('z_objective')

    def _solve_s_scipy(self, **kwds):
        """Solves formulation 22 in weifengs paper (using scipy least_squares)

           This method is not intended to be used by users directly

        Args:
            tee (bool,optional): flag to tell the optimizer whether to stream output
            to the terminal or not
        
            profile_time (bool,optional): flag to tell pyomo to time the construction and solution of the model. 
            Default False

            ftol (float, optional): Tolerance for termination by the change of the cost function. Default is 1e-8

            xtol (float, optional): Tolerance for termination by the change of the independent variables. Default is 1e-8

            gtol (float, optional): Tolerance for termination by the norm of the gradient. Default is 1e-8.

            loss (str, optional): Determines the loss function. The following keyword values are allowed:
                'linear' (default) : rho(z) = z. Gives a standard least-squares problem.

                'soft_l1' : rho(z) = 2 * ((1 + z)**0.5 - 1). The smooth approximation of l1 (absolute value) loss. Usually a good choice for robust least squares.

                'huber' : rho(z) = z if z <= 1 else 2*z**0.5 - 1. Works similarly to 'soft_l1'.

                'cauchy' : rho(z) = ln(1 + z). Severely weakens outliers influence, but may cause difficulties in optimization process.

                'arctan' : rho(z) = arctan(z). Limits a maximum loss on a single residual, has properties similar to 'cauchy'.
            f_scale (float, optional): Value of soft margin between inlier and outlier residuals, default is 1.0

            max_nfev (int, optional): Maximum number of function evaluations before the termination

        Returns:
            None

        """
        
        method = kwds.pop('method','trf')
        def_tol = 1.4901161193847656e-07
        ftol = kwds.pop('ftol', def_tol)
        xtol = kwds.pop('xtol', def_tol)
        gtol = kwds.pop('gtol', def_tol)
        x_scale = kwds.pop('x_scale', 1.0)
        loss = kwds.pop('loss', 'linear')
        f_scale = kwds.pop('f_scale', 1.0)
        max_nfev = kwds.pop('max_nfev', None)
        verbose = kwds.pop('verbose', 2)
        profile_time = kwds.pop('profile_time', False)
        tee = kwds.pop('tee', False)

        if profile_time:
            print('-----------------Solve_S--------------------')
            t0 = time.time()

        # assumes S has been computed in the model                
        # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
        if hasattr(self, '_abs_components'):
            n = self._nabs_components
            for j, l in enumerate(self._meas_lambdas):
                for k, c in enumerate(self._abs_components):
                    if self.model.S[l, c].value < 0.0 and self._is_D_deriv == False:  #: only less thant zero for non-absorbing
                        self._s_array[j * n + k] = 1e-2
                    else:
                        self._s_array[j * n + k] = self.model.S[l, c].value

            for j, t in enumerate(self._meas_times):
                for k, c in enumerate(self._abs_components):
                    self._z_array[j * n + k] = self.model.Z[t, c].value
        else:
            n = self._n_components
            for j, l in enumerate(self._meas_lambdas):
                for k, c in enumerate(self._mixture_components):
                    if self.model.S[l, c].value < 0.0 and self._is_D_deriv == False:  #: only less thant zero for non-absorbing
                        self._s_array[j * n + k] = 1e-2
                    else:
                        self._s_array[j * n + k] = self.model.S[l, c].value
                        
            for j, t in enumerate(self._meas_times):
                for k, c in enumerate(self._mixture_components):
                    self._z_array[j * n + k] = self.model.Z[t, c].value

        def F(x, z_array, d_array, nl, nt, nc):
            diff = np.zeros(nt*nl)
            for i in range(nt):
                for j in range(nl):
                    diff[i*nl+j] = d_array[i, j]-sum(z_array[i*nc+k]*x[j*nc+k] for k in range(nc))
            return diff

        def JF(x, z_array, d_array, nl, nt, nc):
            row = []
            col = []
            data = []
            for i in range(nt):
                for j in range(nl):
                    for k in range(nc):
                        row.append(i*nl+j)
                        col.append(j*nc+k)
                        data.append(-z_array[i*nc+k])
            return coo_matrix((data, (row, col)),
                              shape=(nt*nl, nc*nl))

        # solve
        if tee:
            # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
            if hasattr(self, '_abs_components'):
                if self._is_D_deriv == False:
                    res = least_squares(F,self._s_array,JF,
                                    (0.0,np.inf),method,
                                    ftol,xtol,gtol,
                                    x_scale,loss,f_scale,
                                    max_nfev=max_nfev,
                                    verbose=verbose,
                                    args=(self._z_array,
                                          self._d_array,
                                          self._n_meas_lambdas,
                                          self._n_meas_times,
                                          self._nabs_components))
                else:
                    res = least_squares(F,self._s_array,JF,
                                    (-np.inf,np.inf),method,
                                    ftol,xtol,gtol,
                                    x_scale,loss,f_scale,
                                    max_nfev=max_nfev,
                                    verbose=verbose,
                                    args=(self._z_array,
                                          self._d_array,
                                          self._n_meas_lambdas,
                                          self._n_meas_times,
                                          self._nabs_components))
            else:
                if self._is_D_deriv == False:
                    res = least_squares(F,self._s_array,JF,
                                    (0.0,np.inf),method,
                                    ftol,xtol,gtol,
                                    x_scale,loss,f_scale,
                                    max_nfev=max_nfev,
                                    verbose=verbose,
                                    args=(self._z_array,
                                          self._d_array,
                                          self._n_meas_lambdas,
                                          self._n_meas_times,
                                          self._n_components))
                else:
                    res = least_squares(F,self._s_array,JF,
                                    (-np.inf,np.inf),method,
                                    ftol,xtol,gtol,
                                    x_scale,loss,f_scale,
                                    max_nfev=max_nfev,
                                    verbose=verbose,
                                    args=(self._z_array,
                                          self._d_array,
                                          self._n_meas_lambdas,
                                          self._n_meas_times,
                                          self._n_components))
        else:
            if hasattr(self, '_abs_components'):
                f = StringIO()
                with stdout_redirector(f):
                    if self._is_D_deriv == False: 
                        res = least_squares(F,self._s_array,JF,
                                        (0.0,np.inf),method,
                                        ftol,xtol,gtol,
                                        x_scale,loss,f_scale,
                                        max_nfev=max_nfev,
                                        verbose=verbose,
                                        args=(self._z_array,
                                              self._d_array,
                                              self._n_meas_lambdas,
                                              self._n_meas_times,
                                              self._nabs_components))
                    else: 
                        res = least_squares(F,self._s_array,JF,
                                        (-np.inf,np.inf),method,
                                        ftol,xtol,gtol,
                                        x_scale,loss,f_scale,
                                        max_nfev=max_nfev,
                                        verbose=verbose,
                                        args=(self._z_array,
                                              self._d_array,
                                              self._n_meas_lambdas,
                                              self._n_meas_times,
                                              self._nabs_components))

                with open(self._tmp3,'w') as tf:
                    tf.write(f.getvalue())
            else:
                f = StringIO()
                with stdout_redirector(f):
                    if self._is_D_deriv == False: 
                        res = least_squares(F, self._s_array, JF,
                                        (0.0, np.inf), method,
                                        ftol, xtol, gtol,
                                        x_scale, loss, f_scale,
                                        max_nfev=max_nfev,
                                        verbose=verbose,
                                        args=(self._z_array,
                                              self._d_array,
                                              self._n_meas_lambdas,
                                              self._n_meas_times,
                                              self._n_components))
                    else:
                        res = least_squares(F, self._s_array, JF,
                                        (-np.inf, np.inf), method,
                                        ftol, xtol, gtol,
                                        x_scale, loss, f_scale,
                                        max_nfev=max_nfev,
                                        verbose=verbose,
                                        args=(self._z_array,
                                              self._d_array,
                                              self._n_meas_lambdas,
                                              self._n_meas_times,
                                              self._n_components))

                with open(self._tmp3, 'w') as tf:
                    tf.write(f.getvalue())
        
        if profile_time:
            t1 = time.time()
            print("Scipy.optimize.least_squares time={:.3f} seconds".format(t1-t0))

        # retrive solution to pyomo model
        # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
        if hasattr(self, '_abs_components'):
            for j, l in enumerate(self._meas_lambdas):
                for k, c in enumerate(self._abs_components):
                    self.model.S[l, c].value = res.x[j * n + k]  #: Some of these are not gonna be zero
                    if hasattr(self.model, 'known_absorbance'):
                        if c in self.model.known_absorbance:
                            self.model.S[l, c].set_value(self.model.known_absorbance_data[c][l])
        else:
            for j,l in enumerate(self._meas_lambdas):
                for k,c in enumerate(self._mixture_components):
                    self.model.S[l,c].value = res.x[j*n+k]  #: Some of these are not gonna be zero
                    if hasattr(self.model, 'known_absorbance'):
                        if c in self.model.known_absorbance:
                            self.model.S[l, c].set_value(self.model.known_absorbance_data[c][l])

        return res.success

    def _solve_c_scipy(self, **kwds):
        """Solves formulation 25 in weifengs paper (using scipy least_squares)

           This method is not intended to be used by users directly

        Args:
            tee (bool,optional): flag to tell the optimizer whether to stream output
            to the terminal or not
        
            profile_time (bool,optional): flag to tell pyomo to time the construction and solution of the model. 
            Default False

            ftol (float, optional): Tolerance for termination by the change of the cost function. Default is 1e-8

            xtol (float, optional): Tolerance for termination by the change of the independent variables. Default is 1e-8

            gtol (float, optional): Tolerance for termination by the norm of the gradient. Default is 1e-8.

            loss (str, optional): Determines the loss function. The following keyword values are allowed:
                'linear' (default) : rho(z) = z. Gives a standard least-squares problem.

                'soft_l1' : rho(z) = 2 * ((1 + z)**0.5 - 1). The smooth approximation of l1 (absolute value) loss. Usually a good choice for robust least squares.

                'huber' : rho(z) = z if z <= 1 else 2*z**0.5 - 1. Works similarly to 'soft_l1'.

                'cauchy' : rho(z) = ln(1 + z). Severely weakens outliers influence, but may cause difficulties in optimization process.

                'arctan' : rho(z) = arctan(z). Limits a maximum loss on a single residual, has properties similar to 'cauchy'.
            f_scale (float, optional): Value of soft margin between inlier and outlier residuals, default is 1.0

            max_nfev (int, optional): Maximum number of function evaluations before the termination

        Returns:
            None

        """
        
        method = kwds.pop('method','trf')
        def_tol = 1.4901161193847656e-07
        ftol = kwds.pop('ftol',def_tol)
        xtol = kwds.pop('xtol',def_tol)
        gtol = kwds.pop('gtol',def_tol)
        x_scale = kwds.pop('x_scale',1.0)
        loss = kwds.pop('loss','linear')
        f_scale = kwds.pop('f_scale',1.0)
        max_nfev = kwds.pop('max_nfev',None)
        verbose = kwds.pop('verbose',2)
        profile_time = kwds.pop('profile_time',False)
        tee =  kwds.pop('tee',False)
        
        if profile_time:
            print('-----------------Solve_C--------------------')
            t0 = time.time()
        # assumes S have been computed in the model
        # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
        if hasattr(self, '_abs_components'):
            n = self._nabs_components
            for i, t in enumerate(self._meas_times):
                for k, c in enumerate(self._abs_components):
                    if self.model.Cs[t, c].value <= 0.0:
                        self._c_array[i * n + k] = 1e-15
                    else:
                        self._c_array[i * n + k] = self.model.Cs[t, c].value

            for j, l in enumerate(self._meas_lambdas):
                for k, c in enumerate(self._abs_components):
                    self._s_array[j * n + k] = self.model.S[l, c].value
        else:
            n = self._n_components
            for i, t in enumerate(self._meas_times):
                for k, c in enumerate(self._mixture_components):
                    if self.model.C[t, c].value <= 0.0:
                        self._c_array[i * n + k] = 1e-15
                    else:
                        self._c_array[i * n + k] = self.model.C[t, c].value

            for j, l in enumerate(self._meas_lambdas):
                for k, c in enumerate(self._mixture_components):
                    self._s_array[j * n + k] = self.model.S[l, c].value

        def F(x,s_array,d_array,nl,nt,nc):
            diff = np.zeros(nt*nl)
            for i in range(nt):
                for j in range(nl):
                    diff[i*nl+j]=d_array[i,j]-sum(s_array[j*nc+k]*x[i*nc+k] for k in range(nc))
            return diff

        def JF(x,s_array,d_array,nl,nt,nc):
            row = []
            col = []
            data = []
            for i in range(nt):
                for j in range(nl):
                    for k in range(nc):
                        row.append(i*nl+j)
                        col.append(i*nc+k)
                        data.append(-s_array[j*nc+k])

            return coo_matrix((data, (row, col)),
                              shape=(nt*nl,nc*nt))

        # solve
        if tee:
            # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
            if hasattr(self, '_abs_components'):
                res = least_squares(F, self._c_array, JF,
                                    (0.0, np.inf), method,
                                    ftol, xtol, gtol,
                                    x_scale, loss, f_scale,
                                    max_nfev=max_nfev,
                                    verbose=verbose,
                                    args=(self._s_array,
                                          self._d_array,
                                          self._n_meas_lambdas,
                                          self._n_meas_times,
                                          self._nabs_components))
            else:
                res = least_squares(F,self._c_array,JF,
                                    (0.0,np.inf),method,
                                    ftol,xtol,gtol,
                                    x_scale,loss,f_scale,
                                    max_nfev=max_nfev,
                                    verbose=verbose,
                                    args=(self._s_array,
                                          self._d_array,
                                          self._n_meas_lambdas,
                                          self._n_meas_times,
                                          self._n_components))
        else:
            # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
            if hasattr(self, '_abs_components'):
                f = StringIO()
                with stdout_redirector(f):
                    res = least_squares(F, self._c_array, JF,
                                        (0.0, np.inf), method,
                                        ftol, xtol, gtol,
                                        x_scale, loss, f_scale,
                                        max_nfev=max_nfev,
                                        verbose=verbose,
                                        args=(self._s_array,
                                              self._d_array,
                                              self._n_meas_lambdas,
                                              self._n_meas_times,
                                              self._nabs_components))

                with open(self._tmp4, 'w') as tf:
                    tf.write(f.getvalue())
            else:
                f = StringIO()
                with stdout_redirector(f):
                    res = least_squares(F, self._c_array, JF,
                                        (0.0, np.inf), method,
                                        ftol, xtol, gtol,
                                        x_scale, loss, f_scale,
                                        max_nfev=max_nfev,
                                        verbose=verbose,
                                        args=(self._s_array,
                                              self._d_array,
                                              self._n_meas_lambdas,
                                              self._n_meas_times,
                                              self._n_components))

                with open(self._tmp4, 'w') as tf:
                    tf.write(f.getvalue())

        if profile_time:
            t1 = time.time()
            print("Scipy.optimize.least_squares time={:.3f} seconds".format(t1-t0))

        # retrive solution
        # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
        if hasattr(self, '_abs_components'):
            for j,t in enumerate(self._meas_times):
                for k,c in enumerate(self._abs_components):
                    self.model.Cs[t,c].value = res.x[j*n+k]
        else:
            for j, t in enumerate(self._meas_times):
                for k, c in enumerate(self._mixture_components):
                    self.model.C[t, c].value = res.x[j * n + k]

        return res.success

    def _solve_variances(self, results, fixed_dev_var = None):
        """Solves formulation 23 in weifengs paper (using scipy least_squares)

           This method is not intended to be used by users directly

        Args:
            results (ResultsObject): Data obtained from Weifengs procedure
            fixed_device_var(float, optional): if we have device variance we input it here

        Returns:
            bool indicated if variances were estimated succesfully.

        """
        
        nl = self._n_meas_lambdas
        nt = self._n_meas_times
        b = np.zeros((nl, 1))

        variance_dict = dict()

        # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as s
        if hasattr(self,'_abs_components'):
            nabs=len(self._abs_components)
            A = np.ones((nl, nabs + 1))
            reciprocal_nt = 1.0 / nt
            for i, l in enumerate(self._meas_lambdas):
                for j, t in enumerate(self._meas_times):
                    D_bar = 0.0
                    for w, k in enumerate(self._abs_components):
                        A[i, w] = results.S[k][l] ** 2
                        D_bar += results.S[k][l] * results.Z[k][t]
                    b[i] += (self.model.D[t, l] - D_bar) ** 2
                b[i] *= reciprocal_nt

            if fixed_dev_var == None:
                # try with a simple numpy without bounds first
                res_lsq = np.linalg.lstsq(A, b, rcond=None)
                all_nonnegative = True
                n_vars = nabs + 1

                for i in range(n_vars):
                    if res_lsq[0][i] < 0.0:
                        if res_lsq[0][i] < -1e-5:
                            all_nonnegative = False
                        else:
                            res_lsq[0][i] = abs(res_lsq[0][i])
                    res_lsq[0][i]

                if not all_nonnegative:
                    x0 = np.zeros(nabs + 1) + 1e-2
                    bb = np.zeros(nl)
                    for i in range(nl):
                        bb[i] = b[i]

                    def F(x, M, rhs):
                        return rhs - M.dot(x)

                    def JF(x, M, rhs):
                        return -M

                    res_lsq = least_squares(F, x0, JF,
                                            bounds=(0.0, np.inf),
                                            verbose=2, args=(A, bb))
                    for i, k in enumerate(self._abs_components):
                        variance_dict[k] = res_lsq.x[i]
                    variance_dict['device'] = res_lsq.x[nabs]
                    results.sigma_sq = variance_dict
                    return res_lsq.success

                else:
                    for i, k in enumerate(self._abs_components):
                        variance_dict[k] = res_lsq[0][i][0]
                    variance_dict['device'] = res_lsq[0][nabs][0]
                    results.sigma_sq = variance_dict

            if fixed_dev_var:
                bp = np.zeros((nl, 1))
                for i, l in enumerate(self._meas_lambdas):
                    bp[i] = b[i] - fixed_dev_var
                Ap = np.zeros((nl, nabs))
                for i, l in enumerate(self._meas_lambdas):
                    for j, t in enumerate(self._meas_times):
                        for w, k in enumerate(self._abs_components):
                            Ap[i, w] = results.S[k][l] ** 2

                res_lsq = np.linalg.lstsq(Ap, bp, rcond=None)
                all_nonnegative = True
                n_vars = nabs
                for i in range(n_vars):
                    if res_lsq[0][i] < 0.0:
                        if res_lsq[0][i] < -1e-5:
                            all_nonnegative = False
                        else:
                            res_lsq[0][i] = abs(res_lsq[0][i])
                    res_lsq[0][i]

                for i, k in enumerate(self._abs_components):
                    variance_dict[k] = res_lsq[0][i][0]
                variance_dict['device'] = fixed_dev_var
                results.sigma_sq = variance_dict
        else:
            nc = len(self._sublist_components)
            A = np.ones((nl, nc + 1))
            reciprocal_nt = 1.0/nt
            for i, l in enumerate(self._meas_lambdas):
                for j, t in enumerate(self._meas_times):
                    D_bar = 0.0
                    for w, k in enumerate(self._sublist_components):
                        A[i, w] = results.S[k][l]**2
                        D_bar += results.S[k][l]*results.Z[k][t]
                    b[i] += (self.model.D[t, l]-D_bar)**2
                b[i] *= reciprocal_nt

            if fixed_dev_var == None:
                # try with a simple numpy without bounds first
                res_lsq = np.linalg.lstsq(A, b, rcond=None)
                all_nonnegative = True
                n_vars = nc + 1

                for i in range(n_vars):
                    if res_lsq[0][i] < 0.0:
                        if res_lsq[0][i] < -1e-5:
                            all_nonnegative = False
                        else:
                            res_lsq[0][i] = abs(res_lsq[0][i])
                    res_lsq[0][i]

                if not all_nonnegative:
                    x0 = np.zeros(nc + 1) + 1e-2
                    bb = np.zeros(nl)
                    for i in range(nl):
                        bb[i] = b[i]

                    def F(x, M, rhs):
                        return rhs - M.dot(x)

                    def JF(x, M, rhs):
                        return -M

                    res_lsq = least_squares(F, x0, JF,
                                            bounds=(0.0, np.inf),
                                            verbose=2, args=(A, bb))
                    for i, k in enumerate(self._sublist_components):
                        variance_dict[k] = res_lsq.x[i]
                    variance_dict['device'] = res_lsq.x[nc]
                    results.sigma_sq = variance_dict
                    return res_lsq.success

                else:
                    for i, k in enumerate(self._sublist_components):
                        variance_dict[k] = res_lsq[0][i][0]
                    variance_dict['device'] = res_lsq[0][nc][0]
                    results.sigma_sq = variance_dict

            if fixed_dev_var:
                bp = np.zeros((nl, 1))
                for i, l in enumerate(self._meas_lambdas):
                    bp[i] = b[i] - fixed_dev_var
                Ap = np.zeros((nl, nc))
                for i, l in enumerate(self._meas_lambdas):
                    for j, t in enumerate(self._meas_times):
                        for w, k in enumerate(self._sublist_components):
                            Ap[i, w] = results.S[k][l] ** 2

                res_lsq = np.linalg.lstsq(Ap, bp, rcond=None)
                all_nonnegative = True
                n_vars = nc
                for i in range(n_vars):
                    if res_lsq[0][i] < 0.0:
                        if res_lsq[0][i] < -1e-5:
                            all_nonnegative = False
                        else:
                            res_lsq[0][i] = abs(res_lsq[0][i])
                    res_lsq[0][i]

                for i, k in enumerate(self._sublist_components):
                    variance_dict[k] = res_lsq[0][i][0]
                variance_dict['device'] = fixed_dev_var
                results.sigma_sq = variance_dict
        return 1

    def _build_scipy_lsq_arrays(self):
        """Creates arrays for scipy solvers

           This method is not intended to be used by users directly

        Args:

        Returns:
            None

        """
        self._d_array = np.zeros((self._n_meas_times,self._n_meas_lambdas))
        for i,t in enumerate(self._meas_times):
            for j,l in enumerate(self._meas_lambdas):
                self._d_array[i,j] = self.model.D[t,l]

        # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
        if hasattr(self,'_abs_components'):
            self._s_array = np.ones(self._n_meas_lambdas * self._nabs_components)
            self._z_array = np.ones(self._n_meas_times * self._nabs_components)
            self._c_array = np.ones(self._n_meas_times * self._nabs_components)
        else:
            self._s_array = np.ones(self._n_meas_lambdas*self._n_components)
            self._z_array = np.ones(self._n_meas_times*self._n_components)
            self._c_array = np.ones(self._n_meas_times*self._n_components)

    def _create_tmp_outputs(self):
        """Creates temporary files for loging solutions of each optimization problem

           This method is not intended to be used by users directly

        Args:

        Returns:
            None

        """
        self._tmp2 = "tmp_Z"
        self._tmp3 = "tmp_S"
        self._tmp4 = "tmp_C"
        
        with open(self._tmp2,'w') as f:
            f.write("temporary file for ipopt output")
        
        with open(self._tmp3,'w') as f:
            f.write("temporary file for ipopt output")
        
        with open(self._tmp4,'w') as f:
            f.write("temporary file for ipopt output")

    def _log_iterations(self, filename, iteration):
        """log solution of each subproblem in Weifengs procedure

           This method is not intended to be used by users directly

        Args:

        Returns:
            None

        """
        with open(filename, "a") as f:
            f.write("\n#######################Iteration {}#######################\n".format(iteration))
            with open(self._tmp2,'r') as tf:
                f.write("\n#######################Solve Z {}#######################\n".format(iteration))
                f.write(tf.read())
            with open(self._tmp3,'r') as tf:
                f.write("\n#######################Solve S {}#######################\n".format(iteration))
                f.write(tf.read())
            with open(self._tmp4,'r') as tf:
                f.write("\n#######################Solve C {}#######################\n".format(iteration))
                f.write(tf.read())

    def _build_s_model(self):
        """Builds s_model to solve formulation 22 with ipopt

           This method is not intended to be used by users directly

        Args:

        Returns:
            None

        """
        # initialization
        # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
        if hasattr(self, '_abs_components'):
            self.S_model = ConcreteModel()
            if self._is_D_deriv:
                self.S_model.S = Var(self._meas_lambdas,
                                     self._abs_components,
                                     bounds=(None, None),
                                     initialize=1.0)
            else:
                self.S_model.S = Var(self._meas_lambdas,
                                     self._abs_components,
                                     bounds=(0.0, None),
                                     initialize=1.0)
            for l in self._meas_lambdas:
                for k in self._abs_components:
                    self.S_model.S[l, k].value = self.model.S[l, k].value
                    # if hasattr(self.model, 'non_absorbing'):
                    #     if k in self.model.non_absorbing:
                    #         if self.model.S[l, k].value != 0.0:
                    #             # print("non_zero 772")
                    #             self.S_model.S[l, k].set_value(0.0)
                    #             self.S_model.S[l, k].fix()
                    if hasattr(self.model, 'known_absorbance'):
                        if k in self.model.known_absorbance:
                            if self.model.S[l, k].value != self.model.known_absorbance_data[k][l]:
                                self.model.S[l, k].set_value(self.model.known_absorbance_data[k][l])
                                self.S_model.S[l, k].fix()
        else:
            self.S_model = ConcreteModel()
            if self._is_D_deriv:
                self.S_model.S = Var(self._meas_lambdas,
                                     self._sublist_components,
                                     bounds=(None, None),
                                     initialize=1.0)
            else:
                self.S_model.S = Var(self._meas_lambdas,
                                     self._sublist_components,
                                     bounds=(0.0, None),
                                     initialize=1.0)
            for l in self._meas_lambdas:
                for k in self._sublist_components:
                    self.S_model.S[l, k].value = self.model.S[l, k].value
                    # if hasattr(self.model, 'non_absorbing'):
                    #     if k in self.model.non_absorbing:
                    #         if self.model.S[l, k].value != 0.0:
                    #             # print("non_zero 772")
                    #             self.S_model.S[l, k].set_value(0.0)
                    #             self.S_model.S[l, k].fix()
                    if hasattr(self.model, 'known_absorbance'):
                        if k in self.model.known_absorbance:
                            if self.model.S[l, k].value != self.model.known_absorbance_data[k][l]:
                                self.model.S[l, k].set_value(self.model.known_absorbance_data[k][l])
                                self.S_model.S[l, k].fix()

    def _solve_S(self, solver, **kwds):
        """Solves formulation 23 from Weifengs procedure with ipopt

           This method is not intended to be used by users directly

        Args:

        Returns:
            None

        """
        solver_opts = kwds.pop('solver_opts', dict())
        tee = kwds.pop('tee', False)
        update_nl = kwds.pop('update_nl', False)
        profile_time = kwds.pop('profile_time', False)

        # initialize
        # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
        if hasattr(self, '_abs_components'):
            for l in self._meas_lambdas:
                for c in self._abs_components:
                    self.S_model.S[l, c].value = self.model.S[l, c].value
                    # if hasattr(self.model, 'non_absorbing'):
                    #     if c in self.model.non_absorbing:
                    #         if self.model.S[l, c].value != 0.0:
                    #             # print("non_zero 800")
                    #             self.S_model.S[l, c].set_value(0.0)
                    #             self.S_model.S[l, c].fix()

                    if hasattr(self.model, 'known_absorbance'):
                        if c in self.model.known_absorbance:
                            if self.model.S[l, c].value != self.model.known_absorbance_data[c][l]:
                                self.model.S[l, c].set_value(self.model.known_absorbance_data[c][l])
                                self.S_model.S[l, c].fix()
        else:
            for l in self._meas_lambdas:
                for c in self._sublist_components:
                    self.S_model.S[l, c].value = self.model.S[l, c].value
                    # if hasattr(self.model, 'non_absorbing'):
                    #     if c in self.model.non_absorbing:
                    #         if self.model.S[l, c].value != 0.0:
                    #             # print("non_zero 800")
                    #             self.S_model.S[l, c].set_value(0.0)
                    #             self.S_model.S[l, c].fix()

                    if hasattr(self.model, 'known_absorbance'):
                        if c in self.model.known_absorbance:
                            if self.model.S[l, c].value != self.model.known_absorbance_data[c][l]:
                                self.model.S[l, c].set_value(self.model.known_absorbance_data[c][l])
                                self.S_model.S[l, c].fix()
        obj = 0.0
        # asumes base model has been solved already for Z
        # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
        if hasattr(self, '_abs_components'):
            for t in self._meas_times:
                for l in self._meas_lambdas:
                    D_bar = sum(self.S_model.S[l, k] * self.model.Z[t, k].value for k in self._abs_components)
                    obj += (D_bar - self.model.D[t, l]) ** 2
        else:
            for t in self._meas_times:
                for l in self._meas_lambdas:
                    D_bar = sum(self.S_model.S[l, k] * self.model.Z[t, k].value for k in self._sublist_components)
                    obj += (D_bar - self.model.D[t, l]) ** 2
                    
        self.S_model.objective = Objective(expr=obj)

        if profile_time:
            print('-----------------Solve_S--------------------')

        opt = SolverFactory(solver)

        for key, val in solver_opts.items():
            opt.options[key]=val
        solver_results = opt.solve(self.S_model,
                                   logfile=self._tmp3,
                                   tee=tee,
                                   #keepfiles=True,
                                   #show_section_timing=True,
                                   report_timing=profile_time)

        self.S_model.del_component('objective')
        
        #update values in main model
        # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
        if hasattr(self, '_abs_components'):
            for l in self._meas_lambdas:
                for c in self._abs_components:
                    self.model.S[l, c].value = self.S_model.S[l, c].value
                    # if hasattr(self.model, 'non_absorbing'):
                    #     if c in self.model.non_absorbing:
                    #         if self.S_model.S[l, c].value != 0.0:
                    #             # print("non_zero 837")
                    #             self.model.S[l, c].set_value(0.0)
                    #             self.model.S[l, c].fix()
                    if hasattr(self.model, 'known_absorbance'):
                        if c in self.model.known_absorbance:
                            if self.model.S[l, c].value != self.model.known_absorbance_data[c][l]:
                                self.model.S[l, c].set_value(self.model.known_absorbance_data[c][l])
                                self.S_model.S[l, c].fix()
        else:
            for l in self._meas_lambdas:
                for c in self._sublist_components:
                    self.model.S[l, c].value = self.S_model.S[l, c].value
                    # if hasattr(self.model, 'non_absorbing'):
                    #     if c in self.model.non_absorbing:
                    #         if self.S_model.S[l, c].value != 0.0:
                    #             # print("non_zero 837")
                    #             self.model.S[l, c].set_value(0.0)
                    #             self.model.S[l, c].fix()
                    if hasattr(self.model, 'known_absorbance'):
                        if c in self.model.known_absorbance:
                            if self.model.S[l, c].value != self.model.known_absorbance_data[c][l]:
                                self.model.S[l, c].set_value(self.model.known_absorbance_data[c][l])
                                self.S_model.S[l, c].fix()
                            
    def _build_c_model(self):
        """Builds s_model to solve formulation 25 with ipopt

           This method is not intended to be used by users directly

        Args:

        Returns:
            None

        """
        self.C_model = ConcreteModel()
        self.C_model.C = Var(self._meas_times,
                             self._sublist_components,
                             bounds=(0.0, None),
                             initialize=1.0)

        #add_warm_start_suffixes(self.C_model)
        #self.C_model.scaling_factor = Suffix(direction=Suffix.EXPORT)

        for l in self._meas_times:
            for k in self._sublist_components:
                self.C_model.C[l, k].value = self.model.C[l, k].value
                if hasattr(self.model, 'non_absorbing'):
                    self.C_model.C[l, k].fix()  #: this variable does not need to be part of the optimization
        
    def _solve_C(self,solver,**kwds):
        """Solves formulation 23 from Weifengs procedure with ipopt

           This method is not intended to be used by users directly

        Args:

        Returns:
            None

        """
        solver_opts = kwds.pop('solver_opts', dict())
        tee = kwds.pop('tee', False)
        update_nl = kwds.pop('update_nl', False)
        profile_time = kwds.pop('profile_time', False)
        
        obj = 0.0
        # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS)
        # equality constraint in TemplateBuilder makes Cs subset of C:
        if hasattr(self, '_abs_components'):
            # assumes that s model has been solved first
            for t in self._meas_times:
                for l in self._meas_lambdas:
                    D_bar = sum(self.model.S[l, k].value*self.C_model.C[t, k] for k in self._abs_components)
                    obj += (self.model.D[t, l]-D_bar)**2
        else:
            # assumes that s model has been solved first
            for t in self._meas_times:
                for l in self._meas_lambdas:
                    D_bar = sum(self.model.S[l, k].value*self.C_model.C[t, k] for k in self._sublist_components)
                    obj += (self.model.D[t, l]-D_bar)**2

        self.C_model.objective = Objective(expr=obj)
                
        if profile_time:
            print('-----------------Solve_C--------------------')

        opt = SolverFactory(solver)

        for key, val in solver_opts.items():
            opt.options[key]=val
        solver_results = opt.solve(self.C_model,
                                   logfile=self._tmp4,
                                   tee=tee,
                                   #keepfiles=True,
                                   report_timing=profile_time)

        self.C_model.del_component('objective')
        

        #updates values in main model
        # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
        if hasattr(self, '_abs_components'):
            for t in self._meas_times:
                for c in self._abs_components:
                    self.model.C[t, c].value = self.C_model.C[t, c].value  #: does not matter for non_abs
        else:
            for t in self._meas_times:
                for c in self._sublist_components:
                    self.model.C[t, c].value = self.C_model.C[t, c].value  #: does not matter for non_abs
        #############################################################################
    # additional for the use of model with inputs for variance estimation, CS
    def load_discrete_jump(self):
        self.jump = True

        zeit = None
        for i in self.model.component_objects(ContinuousSet):
            zeit = i
            break
        if zeit is None:
            raise Exception('no continuous_set')
        self.time_set = zeit.name

        tgt_cts = getattr(self.model, self.time_set)  ## please correct me (not necessary!)
        self.ncp = tgt_cts.get_discretization_info()['ncp']
        fe_l = tgt_cts.get_finite_elements()
        fe_list = [fe_l[i + 1] - fe_l[i] for i in range(0, len(fe_l) - 1)]

        for i in range(0, len(fe_list)):  # test whether integer elements
            self.jump_constraints(i)
        # self.jump_constraints()

    def jump_constraints(self, fe):
        # type: (int) -> None
        """ Take the current state of variables of the initializing model at fe and load it into the tgt_model
        Note that this will skip fixed variables as a safeguard.

        Args:
            fe (int): The current finite element to be patched (tgt_model).
        """
        ###########################
        if not isinstance(fe, int):
            raise Exception  # wrong type
        ttgt = getattr(self.model, self.time_set)
        ##############################
        # Inclusion of discrete jumps: (CS)
        if self.jump:
            vs = ReplacementVisitor()  #: trick to replace variables
            kn = 0
            for ki in self.jump_times_dict.keys():
                if not isinstance(ki, str):
                    print("ki is not str")
                vtjumpkeydict = self.jump_times_dict[ki]
                for l in vtjumpkeydict.keys():
                    self.jump_time = vtjumpkeydict[l]
                    # print('jumptime:',self.jump_time)
                    self.jump_fe, self.jump_cp = fe_cp(ttgt, self.jump_time)
                    if self.jump_time not in self.feed_times_set:
                        raise Exception("Error: Check feed time points in set feed_times and in jump_times again.\n"
                                        "They do not match.\n"
                                        "Jump_time is not included in feed_times.")
                    # print('jump_el, el:',self.jump_fe, fe)
                    if fe == self.jump_fe + 1:
                        # print("jump_constraints!")
                        #################################
                        for v in self.disc_jump_v_dict.keys():
                            if not isinstance(v, str):
                                print("v is not str")
                            vkeydict = self.disc_jump_v_dict[v]
                            for k in vkeydict.keys():
                                if k == l:  # Match in between two components of dictionaries
                                    var = getattr(self.model, v)
                                    dvar = getattr(self.model, "d" + v + "dt")
                                    con_name = 'd' + v + 'dt_disc_eq'
                                    con = getattr(self.model, con_name)

                                    self.model.add_component(v + "_dummy_eq_" + str(kn), ConstraintList())
                                    conlist = getattr(self.model, v + "_dummy_eq_" + str(kn))
                                    varname = v + "_dummy_" + str(kn)
                                    self.model.add_component(varname, Var([0]))  #: this is now indexed [0]
                                    vdummy = getattr(self.model, varname)
                                    vs.change_replacement(vdummy[0])   #: who is replacing.
                                    # self.model.add_component(varname, Var())
                                    # vdummy = getattr(self.model, varname)
                                    jump_delta = vkeydict[k]
                                    self.model.add_component(v + '_jumpdelta' + str(kn),
                                                             Param(initialize=jump_delta))
                                    jump_param = getattr(self.model, v + '_jumpdelta' + str(kn))
                                    if not isinstance(k, tuple):
                                        k = (k,)
                                    exprjump = vdummy[0] - var[(self.jump_time,) + k] == jump_param  #: this cha
                                    # exprjump = vdummy - var[(self.jump_time,) + k] == jump_param
                                    self.model.add_component("jumpdelta_expr" + str(kn), Constraint(expr=exprjump))
                                    for kcp in range(1, self.ncp + 1):
                                        curr_time = t_ij(ttgt, self.jump_fe + 1, kcp)
                                        if not isinstance(k, tuple):
                                            knew = (k,)
                                        else:
                                            knew = k
                                        idx = (curr_time,) + knew
                                        con[idx].deactivate()
                                        e = con[idx].expr
                                        suspect_var = e.args[0].args[1].args[0].args[0].args[1]  #: seems that
                                        # e = con[idx].expr.clone()
                                        # e.args[0].args[1] = vdummy
                                        # con[idx].set_value(e)
                                        vs.change_suspect(id(suspect_var))  #: who to replace
                                        e_new = vs.dfs_postorder_stack(e)  #: replace
                                        con[idx].set_value(e_new)
                                        conlist.add(con[idx].expr)
                    kn = kn + 1
#############################################################

def add_warm_start_suffixes(model):
    # Ipopt bound multipliers (obtained from solution)
    model.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
    model.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
    # Ipopt bound multipliers (sent to solver)
    model.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
    model.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)
    # Obtain dual solutions from first solve and send to warm start
    model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)


def compute_diff_results(results1,results2):
    diff_results = ResultsObject()
    diff_results.Z = results1.Z - results2.Z
    diff_results.S = results1.S - results2.S
    diff_results.C = results1.C - results2.C
    return diff_results

#######################additional for inputs###CS
def t_ij(time_set, i, j):
    # type: (ContinuousSet, int, int) -> float
    """Return the corresponding time(continuous set) based on the i-th finite element and j-th collocation point
    From the NMPC_MHE framework by @dthierry.

    Args:
        time_set (ContinuousSet): Parent Continuous set
        i (int): finite element
        j (int): collocation point

    Returns:
        float: Corresponding index of the ContinuousSet
    """
    if i < time_set.get_discretization_info()['nfe']:
        h = time_set.get_finite_elements()[i + 1] - time_set.get_finite_elements()[i]  #: This would work even for 1 fe
    else:
        h = time_set.get_finite_elements()[i] - time_set.get_finite_elements()[i - 1]  #: This would work even for 1 fe
    tau = time_set.get_discretization_info()['tau_points']
    fe = time_set.get_finite_elements()[i]
    time = fe + tau[j] * h
    return round(time, 6)


def fe_cp(time_set, feedtime):
    # type: (ContinuousSet, float) -> tuple
    # """Return the corresponding fe and cp for a given time
    # Args:
    #    time_set:
    #    t:
    # """
    fe_l = time_set.get_lower_element_boundary(feedtime)
    # print("fe_l", fe_l)
    fe = None
    j = 0
    for i in time_set.get_finite_elements():
        if fe_l == i:
            fe = j
            break
        j += 1
    h = time_set.get_finite_elements()[1] - time_set.get_finite_elements()[0]
    tauh = [i * h for i in time_set.get_discretization_info()['tau_points']]
    j = 0  #: Watch out for LEGENDRE
    cp = None
    for i in tauh:
        if round(i + fe_l, 6) == feedtime:
            cp = j
            break
        j += 1
    return fe, cp
###########################################################

#: This class can replace variables from an expression
class ReplacementVisitor(EXPR.ExpressionReplacementVisitor):
    def __init__(self):
        super(ReplacementVisitor, self).__init__()
        self._replacement = None
        self._suspect = None

    def change_suspect(self, suspect_):
        self._suspect = suspect_

    def change_replacement(self, replacement_):
        self._replacement = replacement_

    def visiting_potential_leaf(self, node):
        #
        # Clone leaf nodes in the expression tree
        #
        if node.__class__ in native_numeric_types:
            return True, node

        if node.__class__ is NumericConstant:
            return True, node


        if node.is_variable_type():
            if id(node) == self._suspect:
                d = self._replacement
                return True, d
            else:
                return True, node

        return False, None
