import copy
import time

from pyomo.core import *
from pyomo.dae import *
from pyomo.environ import *

from kipet.library.Optimizer import *
from kipet.library.mixins.VisitorMixins import ReplacementVisitor

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

        self._sublist_components = list_components

        if hasattr(self.model, 'non_absorbing'):
            warnings.warn("Overriden by non_absorbing")
            list_components = [k for k in self._mixture_components if k not in self._non_absorbing]
            self._sublist_components = list_components
        
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
                                        for key in self.model.alltime.value:
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
            for i in self.jump_times_dict.keys():
                for j in self.jump_times_dict[i].items():
                    count += 1
            if len(self.feed_times_set) > count:
                raise Exception("Error: Check feed time points in set feed_times and in jump_times again.\n"
                            "There are more time points in feed_times than jump_times provided.")
            self.load_discrete_jump()
        

        if report_time:
            start = time.time()
             
        if method == 'originalchenetal':  
            from kipet.library.variance_methods.chen_method import run_method
            results = run_method(self, solver, run_opt_kwargs)
            
        elif method == 'alternate':
            from kipet.library.variance_methods.alternate_method import run_alternate_method
            solver = 'ipopt'
            results = run_alternate_method(self, solver, run_opt_kwargs)
        
        elif method == 'direct_sigmas':
            from kipet.library.variance_methods.alternate_method import run_direct_sigmas_method
            results = run_direct_sigma_method(self, solver, run_opt_kwargs)
            
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
    
    def load_discrete_jump(self):
        self.jump = True

        zeit = None
        for i in self.model.component_objects(ContinuousSet):
            zeit = i
            break
        if zeit is None:
            raise Exception('No continuous_set')

        self.time_set = zeit.name

        tgt_cts = getattr(self.model, self.time_set)  ## please correct me (not necessary!)
        self.ncp = tgt_cts.get_discretization_info()['ncp']
        fe_l = tgt_cts.get_finite_elements()
        fe_list = [fe_l[i + 1] - fe_l[i] for i in range(0, len(fe_l) - 1)]

        for i in range(0, len(fe_list)):  # test whether integer elements
            self.jump_constraints(i)
        # self.jump_constraints()

    # I want to change this spaghetti
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


def add_warm_start_suffixes(model):
    # Ipopt bound multipliers (obtained from solution)
    model.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
    model.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
    # Ipopt bound multipliers (sent to solver)
    model.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
    model.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)
    # Obtain dual solutions from first solve and send to warm start
    model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)

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
    """Return the corresponding fe and cp for a given time
     Args:
        time_set:
        t:
    
    type: (ContinuousSet, float) -> tuple
             
    """
    fe_l = time_set.get_lower_element_boundary(feedtime)
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