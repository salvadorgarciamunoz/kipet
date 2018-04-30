from __future__ import print_function
from __future__ import division
from pyomo.environ import *
from pyomo.dae import *
from pyomo.core import *
from pyomo.opt import (ReaderFactory,
                       ResultsFormat)
from kipet.opt.Optimizer import *
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
    def __init__(self,model):
        super(VarianceEstimator, self).__init__(model)
        add_warm_start_suffixes(self.model)

        if not self._spectra_given:
            raise NotImplementedError("Variance estimator requires spectral data in model as model.D[ti,lj]")


        
    def run_sim(self, solver, **kwds):
        raise NotImplementedError("VarianceEstimator object does not have run_sim method. Call run_opt")

    def run_opt(self, solver, **kwds):

        """Solves estimation following Weifengs procedure.
           This method solved a sequence of optimization problems
           to determine variances and initial guesses for parameter estimation.

        Args:

            solver_opts (dict, optional): options passed to the nonlinear solver
        
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
        
        solver_opts = kwds.pop('solver_opts', dict())
        variances = kwds.pop('variances', dict())
        tee = kwds.pop('tee', False)
        norm_order = kwds.pop('norm',np.inf)
        max_iter = kwds.pop('max_iter', 400)
        tol = kwds.pop('tolerance', 5.0e-5)
        A = kwds.pop('subset_lambdas', None)
        lsq_ipopt = kwds.pop('lsq_ipopt', False)
        init_C = kwds.pop('init_C', None)

        species_list = kwds.pop('subset_components', None)
        
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


        # solves formulation 18
        if init_C is None:
            self._solve_initalization(solver, subset_lambdas=A, tee=tee)
        else:
            for t in self._meas_times:
                for k in self._mixture_components:
                    self.model.C[t, k].value = init_C[k][t]
                    self.model.Z[t, k].value = init_C[k][t]

            s_array = self._solve_S_from_DC(init_C)
            S_frame = pd.DataFrame(data=s_array,
                                   columns=self._mixture_components,
                                   index=self._meas_lambdas)
            
            for l in self._meas_lambdas:
                for k in self._mixture_components:
                    self.model.S[l, k].value = S_frame[k][l] #1e-2
                    #: Some of these are gonna be non-zero
                    if hasattr(self.model, 'non_absorbing'):
                        if k in self.model.non_absorbing:
                            self.model.S[l, k].value = 0.0
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
            ra.load_from_pyomo_model(self.model, to_load=['Z','C','S'])
            
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
        results.load_from_pyomo_model(self.model,
                                      to_load=['Z', 'dZdt', 'X', 'dXdt', 'C', 'S', 'Y'])

        print('Iterative optimization converged. Estimating variances now')
        # compute variances
        solved_variances = self._solve_variances(results)
        
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
        for k in self._sublist_components:
            if k not in keys:
                sigmas_sq[k] = 0.0

        print("Solving Initialization Problem\n")

        # the ones that are not in set_A are going to be stale and wont go to the optimizer
        
        # build objective
        obj = 0.0
        for t in self._meas_times:
            for l in set_A:
                D_bar = sum(self.model.Z[t, k]*self.model.S[l, k] for k in self._sublist_components)
                obj+= (self.model.D[t, l] - D_bar)**2
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
        
        # asume this values were computed in beforehand
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
        self.model.write("whatevs.nl", format=ProblemFormat.nl, io_options={"symbolic_solver_labels": True})
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
        n = self._n_components
        for j, l in enumerate(self._meas_lambdas):
            for k, c in enumerate(self._mixture_components):
                if self.model.S[l, c].value < 0.0:  #: only less thant zero for non-absorbing
                    self._s_array[j*n+k] = 1e-2
                else:
                    self._s_array[j*n+k] = self.model.S[l, c].value

        for j,t in enumerate(self._meas_times):
            for k,c in enumerate(self._mixture_components):
                self._z_array[j*n+k] = self.model.Z[t, c].value

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
            f = StringIO()
            with stdout_redirector(f):
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

            with open(self._tmp3,'w') as tf:
                tf.write(f.getvalue())
        
        if profile_time:
            t1 = time.time()
            print("Scipy.optimize.least_squares time={:.3f} seconds".format(t1-t0))

        # retrive solution to pyomo model
        for j,l in enumerate(self._meas_lambdas):
            for k,c in enumerate(self._mixture_components):
                self.model.S[l,c].value = res.x[j*n+k]  #: Some of these are not gonna be zero
                if hasattr(self.model, 'non_absorbing'):
                    if c in self.model.non_absorbing:
                        self.model.S[l, c].set_value(0.0)


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
        n = self._n_components
        for i,t in enumerate(self._meas_times):
            for k,c in enumerate(self._mixture_components):
                if self.model.C[t,c].value <=0.0:
                    self._c_array[i*n+k] = 1e-15
                else:
                    self._c_array[i*n+k] = self.model.C[t,c].value 

        for j,l in enumerate(self._meas_lambdas):
            for k,c in enumerate(self._mixture_components):
                self._s_array[j*n+k] = self.model.S[l,c].value

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
            f = StringIO()
            with stdout_redirector(f):
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

            with open(self._tmp4,'w') as tf:
                tf.write(f.getvalue())

        if profile_time:
            t1 = time.time()
            print("Scipy.optimize.least_squares time={:.3f} seconds".format(t1-t0))

        # retrive solution
        for j,t in enumerate(self._meas_times):
            for k,c in enumerate(self._mixture_components):
                self.model.C[t,c].value = res.x[j*n+k]

        return res.success

    def _solve_variances(self, results):
        """Solves formulation 23 in weifengs paper (using scipy least_squares)

           This method is not intended to be used by users directly

        Args:
            results (ResultsObject): Data obtained from Weifengs procedure 

        Returns:
            bool indicated if variances were estimated succesfully.

        """
        nl = self._n_meas_lambdas
        nt = self._n_meas_times
        nc = len(self._sublist_components)
        A = np.ones((nl, nc+1))
        b = np.zeros((nl, 1))

        reciprocal_nt = 1.0/nt
        for i, l in enumerate(self._meas_lambdas):
            for j, t in enumerate(self._meas_times):
                D_bar = 0.0
                for w, k in enumerate(self._sublist_components):
                    A[i, w] = results.S[k][l]**2
                    D_bar += results.S[k][l]*results.Z[k][t]
                b[i] += (self.model.D[t, l]-D_bar)**2
            b[i] *= reciprocal_nt

        # try with a simple numpy without bounds first
        res_lsq = np.linalg.lstsq(A, b)
        all_nonnegative = True
        n_vars = nc+1
        
        for i in range(n_vars):
            if res_lsq[0][i] < 0.0:
                if res_lsq[0][i] < -1e-5:
                    all_nonnegative=False
                else:
                    res_lsq[0][i] = abs(res_lsq[0][i])
            res_lsq[0][i]

        variance_dict = dict()
        if not all_nonnegative:
            x0 = np.zeros(nc + 1) + 1e-2
            bb = np.zeros(nl)
            for i in range(nl):
                bb[i] = b[i]

            def F(x, M, rhs):
                return  rhs-M.dot(x)

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
        self.S_model = ConcreteModel()
        self.S_model.S = Var(self._meas_lambdas,
                             self._sublist_components,
                             bounds=(0.0, None),
                             initialize=1.0)

        # initialization
        for l in self._meas_lambdas:
            for k in self._sublist_components:
                self.S_model.S[l, k].value = self.model.S[l, k].value
                if hasattr(self.model, 'non_absorbing'):
                    if k in self.model.non_absorbing:
                        if self.model.S[l, k].value != 0.0:
                            # print("non_zero 772")
                            self.S_model.S[l, k].set_value(0.0)
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
        for l in self._meas_lambdas:
            for c in self._sublist_components:
                self.S_model.S[l, c].value = self.model.S[l, c].value
                if hasattr(self.model, 'non_absorbing'):
                    if c in self.model.non_absorbing:
                        if self.model.S[l, c].value != 0.0:
                            # print("non_zero 800")
                            self.S_model.S[l, c].set_value(0.0)
                            self.S_model.S[l, c].fix()
        
        obj = 0.0
        # asumes base model has been solved already for Z
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
        for l in self._meas_lambdas:
            for c in self._sublist_components:
                self.model.S[l, c].value = self.S_model.S[l, c].value
                if hasattr(self.model, 'non_absorbing'):
                    if c in self.model.non_absorbing:
                        if self.S_model.S[l, c].value != 0.0:
                            # print("non_zero 837")
                            self.model.S[l, c].set_value(0.0)
                            self.model.S[l, c].fix()



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
        # asumes that s model has been solved first
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
        for t in self._meas_times:
            for c in self._sublist_components:
                self.model.C[t, c].value = self.C_model.C[t, c].value  #: does not matter for non_abs
        
        
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

    
