import copy
import time

from pyomo.core import *
from pyomo.dae import *
from pyomo.environ import *

from kipet.core_methods.Optimizer import *


class VarianceEstimator(Optimizer):
    """Optimizer for variance estimation.

    Attributes:

        model (model): Pyomo model.

    """
    def __init__(self, model):
        super(VarianceEstimator, self).__init__(model)
        self.add_warm_start_suffixes(self.model)

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
                                                
            secant_point (float, optional): Provides the second point (in addition to the init_sigma) for the secant method
                                            if not provided it will use 10 times the initial point
            
            device_range (tuple, optional): when using direct sigmas approach this provides the search region for delta
            
            num_points (int, optional): This provides the number of evaluations when using direct sigmas approach
            
            init_sigmas (float, optional): this will provide an initial value for the sigmas when using the alternative variance method
            
            individual_species (bool, optional): bool to see whether the overall model variance is returned or if the individual
                                                species' variances are returned based on the obtainde value for delta
            
            with_plots (bool, optional): if with_plots is provided, it will plot the log objective versus the
                                            iterations for the direct_sigmas method.

        Returns:

            Results from the optimization (pyomo model)

        """
        run_opt_kwargs = copy.copy(kwds)
        
        solver_opts = kwds.pop('solver_opts', dict())
        sigma_sq = kwds.pop('variances', dict())
        init_sigmas = kwds.pop('initial_sigmas', float())
        tee = kwds.pop('tee', False)
        method = kwds.pop('method', str())
        norm_order = kwds.pop('norm',np.inf)
        max_iter = kwds.pop('max_iter', 400)
        tol = kwds.pop('tolerance', 5.0e-5)
        A = kwds.pop('subset_lambdas', None)
        lsq_ipopt = kwds.pop('lsq_ipopt', False)
        init_C = kwds.pop('init_C', None)
        report_time = kwds.pop('report_time', False)
        individual_species = kwds.pop('individual_species', False)

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
        secant_point2 = kwds.pop('secant_point', None)
        num_points = kwds.pop('num_points', None)
        with_plots = kwds.pop('with_plots', False)
        
        methods_available = ['originalchenetal', "direct_sigmas", "alternate"]
        
        if method not in methods_available:
            method = 'originalchenetal'
            print(f'Default method of {method} selected')
            print(f'The available methods are {", ".join(methods_available)}')
            
        if not self.model.alltime.get_discretization_info():
            raise RuntimeError('apply discretization first before initializing')

        self._create_tmp_outputs()
        
        objectives_map = self.model.component_map(ctype=Objective, active=True)
        active_objectives_names = []
        for obj in objectives_map.values():
            active_objectives_names.append(obj.cname())
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

        # Set up the correct components based on absorption
        if hasattr(self.model, 'non_absorbing'):
            warnings.warn("Overriden by non_absorbing")
            list_components = [k for k in self._mixture_components if k not in self._non_absorbing]
        
        if hasattr(self.model, 'known_absorbance'):
            warnings.warn("Overriden by species with known absorbance")
            list_components = [k for k in self._mixture_components if k not in self._known_absorbance]
        
        self._sublist_components = list_components
        
        if hasattr(self, '_abs_components'):
            self.component_set = self._abs_components
            self.component_var = 'Cs'
        else:
            self.component_set = self._sublist_components
            self.component_var = 'C'
        
        # Fixed imputs and trajectory section
        if inputs_sub is not None:
            from kipet.common.additional_inputs import add_inputs
            
            add_kwargs = dict(
                fixedtraj = fixedtraj,
                fixedy = fixedy, 
                inputs_sub = inputs_sub,
                yfix = yfix,
                yfixtraj = yfixtraj,
                trajectories = trajectories,
            )
            
            add_inputs(self, add_kwargs)

        # Dosing
        if jump:
            self.set_up_jumps(run_opt_kwargs)
        
        if report_time:
            start = time.time()
             
        # Call the chosen method
        if method == 'originalchenetal':  
            from kipet.variance_methods.chen_method import run_method
            results = run_method(self, solver, run_opt_kwargs)
            
        elif method == 'alternate':
            from kipet.variance_methods.alternate_method import run_alternate_method
            solver = 'ipopt'
            results = run_alternate_method(self, solver, run_opt_kwargs)
        
        elif method == 'direct_sigmas':
            from kipet.variance_methods.alternate_method import run_direct_sigmas_method
            results = run_direct_sigmas_method(self, solver, run_opt_kwargs)
            
        # Report time
        if report_time:
            end = time.time()
            print("Total execution time in seconds for variance estimation:", end - start)
        
        return results

    def _warn_if_D_negative(self):
    
        for t in self._meas_times:
            for l in self._meas_lambdas:
                if self.model.D[t, l] >= 0:
                    pass
                else:
                    self._is_D_deriv = True
        if self._is_D_deriv == True:
            print("Warning! Since D-matrix contains negative values Kipet is relaxing non-negativity on S")

        return None

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
            
    def solve_max_device_variance(self, solver, **kwds):
        """Solves the maximum likelihood formulation with (C-Z) = 0. solves ntp/2*log(eTe/ntp)
        and then calculates delta from the solution. See documentation for more details.
    
        Args:
            solver_opts (dict, optional): options passed to the nonlinear solver
        
            tee (bool,optional): flag to tell the optimizer whether to stream output
            to the terminal or not
        
            subset_lambdas (array_like,optional): Set of wavelengths to used in initialization problem 
            (Weifeng paper). Default all wavelengths.
    
        Returns:
    
            delta_sq (float): value of the max device variance
    
        """   
        solver_opts = kwds.pop('solver_opts', dict())
        tee = kwds.pop('tee', True)
        set_A = kwds.pop('subset_lambdas', list())
    
        if not set_A:
            set_A = self._meas_lambdas
    
        list_components = [k for k in self._mixture_components]
                    
        if hasattr(self.model, 'non_absorbing'):
            warnings.warn("Overriden by non_absorbing")
            list_components = [k for k in self._mixture_components if k not in self._non_absorbing]
        
        if hasattr(self.model, 'known_absorbance'):
            warnings.warn("Overriden by species with known absorbance")
            list_components = [k for k in self._mixture_components if k not in self._known_absorbance]
        
        self._sublist_components = list_components
            
        print("Solving For the worst possible device variance\n")
        
        self._warn_if_D_negative()
        
        obj = 0.0
        ntp = len(self._meas_times)
        nwp = len(self._meas_lambdas)
        
        if hasattr(self, '_abs_components'):
            self.component_set = self._abs_components
        else:
            self.component_set = self._sublist_components
        
        for t in self._meas_times:
            for l in set_A:
                D_bar = sum(self.model.Z[t, k] * self.model.S[l, k] for k in self.component_set)
                obj += (self.model.D[t, l] - D_bar)**2
       
        self.model.init_objective = Objective(expr=obj)
        
        opt = SolverFactory(solver)
    
        for key, val in solver_opts.items():
            opt.options[key]=val
        solver_results = opt.solve(self.model,
                                   tee=tee)
        
        print("values for the parameters in case with no model variance")
        for k, v in self.model.P.items():
            print(k, v.value)
        
        etaTeta = 0
        for t in self._meas_times:
            for l in set_A:
                D_bar = sum(value(self.model.Z[t, k]) * value(self.model.S[l, k]) for k in self.component_set)
                etaTeta += (value(self.model.D[t, l])- D_bar)**2
    
        deltasq = etaTeta/(ntp*nwp)
        
        self.model.del_component('init_objective')
        print("worst case delta squared: ", deltasq)
    
        return deltasq
    
    def solve_sigma_given_delta(self, solver, **kwds):
        """Function that solves for model variances based on a given device variance. Solves
        The log likelihood function and returns a dictionary containg all sigmas, including
        the device/delta in order to easily apply to the parameter estimation problem.
           This method is intended to be used by users directly
        Args:
            solver (str): solver to use to solve the problems (recommended "ipopt")
            
            delta (float): the device variance squared 
        
            tee (bool,optional): flag to tell the optimizer whether to stream output
            to the terminal or not
        
            solver_opts (dict,optional): Dictionary containing solver options
        
            subset_lambdas (array_like,optional): Set of wavelengths to used in problem 
            (Weifeng paper). Default all wavelengths.
        Returns:
            all_variances (dict): dictionary containg all sigmas, including the device/delta
        """   
        from kipet.variance_methods.alternate_method import _solve_sigma_given_delta
        
        solver_opts = kwds.pop('solver_opts', dict())
        tee = kwds.pop('tee', True)
        set_A = kwds.pop('subset_lambdas', list())
        delta = kwds.pop('delta', dict())
        
        residuals, sigma_vals, stop_it, results = _solve_sigma_given_delta(self, 
                                                                           solver, 
                                                                           subset_lambdas= set_A, 
                                                                           solver_opts = solver_opts, 
                                                                           tee=tee, 
                                                                           delta = delta)
        all_variances = sigma_vals
        all_variances['device'] = delta
        return all_variances