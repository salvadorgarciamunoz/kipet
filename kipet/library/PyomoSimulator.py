import six
import sys
import warnings

from pyomo.core.expr import current as EXPR
from pyomo.dae import *
from pyomo.environ import *

from kipet.library.ResultsObject import *
from kipet.library.Simulator import *
from kipet.library.mixins.VisitorMixins import ScalingVisitor

    
class PyomoSimulator(Simulator):
    """Simulator based on pyomo.dae discretization strategies.

    Attributes:
        model (Pyomo model)

        _times (array_like): array of times after discretization

        _n_times (int): number of discretized time points

        _ipopt_scaled (bool): flag that indicates if there are
        ipopt scaling factors specified
    """

    def __init__(self, model):
        """Simulator constructor.

        Note:
            Makes a shallow copy to the model. Changes applied to
            the model within the simulator are applied to the original
            model passed to the simulator

        Args:
            model (Pyomo model)
        """
        super(PyomoSimulator, self).__init__(model)
        self._alltimes = sorted(self.model.alltime)#added for special structure CS
        #self._times = sorted(self.model.time)
        self._n_alltimes = len(self._alltimes) #added for special structure CS
        #self._n_times = len(self._times)
        self._meas_times=sorted(self.model.meas_times)
        self._allmeas_times=sorted(self.model.allmeas_times)
        self._ipopt_scaled = False
        self._spectra_given = hasattr(self.model, 'D')
        self._concentration_given = hasattr(self.model, 'C')
        self._conplementary_states_given = hasattr(self.model, 'U')
        self._absorption_given = hasattr(self.model,
                                         'S')  # added for special case of absorption data available but not concentration data CS
        self._huplc_given = hasattr(self.model, 'Chat')
        self._smoothparam_given = hasattr(self.model, 'Ps')

        # creates scaling factor suffix
        if not hasattr(self.model, 'scaling_factor'):
            self.model.scaling_factor = Suffix(direction=Suffix.EXPORT)

    def apply_discretization(self, transformation, **kwargs):
        """Discretizes the model.

        Args:
            transformation (str): TODO
            same keywords as in pyomo.dae method

        Returns:
            None
        """
        # Additional keyword - not to use alltimes as the basis for the finite elements.
        
        fixed_times = kwargs.pop('fixed_times', None)
        
        if not self.model.alltime.get_discretization_info():
            
            discretizer = TransformationFactory(transformation)
            
            if fixed_times == None:
                discretizer.apply_to(self.model, wrt=self.model.alltime, **kwargs)
            else:
                discretizer.apply_to(self.model, wrt=fixed_times, **kwargs)
            self._alltimes = sorted(self.model.alltime)
            self._n_alltimes = len(self._alltimes)

            #added for optional smoothing parameter with reading values from file CS:
            if self._smoothparam_given:
                dfps = pd.DataFrame(index=self.model.alltime, columns=self.model.smoothparameter_names)
                for t in self.model.alltime:
                    if t not in self.model.allsmooth_times:  # for points that are the same in original meas times and feed times
                        dfps.loc[t] = float(22.) #something that is not between 0 and 1
                    else:
                        ps_dict_help = dict()
                        for p in self.model.smoothparameter_names:
                            ps_dict_help[t, p] = value(self.model.smooth_param_data[t, p])
                        dfps.loc[t] = [ps_dict_help[t, p] for p in self.model.smoothparameter_names]
                dfallps = dfps
                dfallps.sort_index(inplace=True)
                dfallps.index = dfallps.index.to_series().apply(
                    lambda x: np.round(x, 6))  # time from data rounded to 6 digits

                dfallpsall = pd.DataFrame(index=self.model.alltime, columns=self.model.smoothparameter_names)
                dfsmoothdata = pd.DataFrame(index=sorted(self.model.smooth_param_datatimes), columns=self.model.smoothparameter_names)

                for t in self.model.smooth_param_datatimes:
                    dfsmoothdata.loc[t] = [value(self.model.smooth_param_data[t, p]) for p in self.model.smoothparameter_names]

                for p in self.model.smoothparameter_names:
                    for ti in self.model.alltime:
                        if float(dfallps[p][ti]) > 1:
                            valueinterp=interpolate_from_trajectory(ti, dfsmoothdata[p])
                            dfallpsall[p][ti] = float(valueinterp)
                        else:
                            dfallpsall.loc[ti] = float(dfallps[p][ti])

            self._default_initialization()
            
            if hasattr(self.model, 'K'):
                print('Scaling the parameters')
                self.scale_parameters()
        else:
            print('***WARNING: Model already discretized. Ignoring second discretization')
            
    def scale_parameters(self):
        """If scaling, this multiplies the constants in model.K to each
        parameter in model.P.
        
        I am not sure if this is necessary and will look into its importance.
        """
        #if self.model.K is not None:
        self.scale = {}
        for i in self.model.P:
            self.scale[id(self.model.P[i])] = self.model.K[i]

        for i in self.model.Z:
            self.scale[id(self.model.Z[i])] = 1
            
        for i in self.model.dZdt:
            self.scale[id(self.model.dZdt[i])] = 1
            
        for i in self.model.X:
            self.scale[id(self.model.X[i])] = 1
    
        for i in self.model.dXdt:
            self.scale[id(self.model.dXdt[i])] = 1
    
        for k, v in self.model.odes.items(): 
            scaled_expr = self.scale_expression(v.body, self.scale)
            self.model.odes[k] = scaled_expr == 0
            
    def scale_expression(self, expr, scale):
        
        visitor = ScalingVisitor(scale)
        return visitor.dfs_postorder_stack(expr)

    def fix_from_trajectory(self, variable_name, variable_index, trajectories):

        if variable_name in ['X', 'dXdt', 'Z', 'dZdt']:
            raise NotImplementedError("Fixing state variables is not allowd. Only algebraics can be fixed")

        single_traj = trajectories[variable_index]
        sim_alltimes = sorted(self._alltimes)
        var = getattr(self.model, variable_name)
        for i, t in enumerate(sim_alltimes):
            value = interpolate_from_trajectory(t, single_traj)
            var[t, variable_index].fix(value)

    def unfix_time_dependent_variable(self, variable_name, variable_index):
        var = getattr(self.model, variable_name)
        sim_times = sorted(self._alltimes)
        for i, t in enumerate(sim_times):
            var[t, variable_index].fixed = False

    # initializes the trajectories to the initial conditions
    def _default_initialization(self):
        """Initializes discreted variables model with initial condition values.

           This method is not intended to be used by users directly
        Args:
            None

        Returns:
            None
        """
        tol = 1e-4
        z_init = []
        for t in self._alltimes:
            for k in self._mixture_components:
                if abs(self.model.init_conditions[k].value) > tol:
                    z_init.append(self.model.init_conditions[k].value)
                else:
                    z_init.append(1.0)

        z_array = np.array(z_init).reshape((self._n_alltimes, self._n_components))
        z_init_panel = pd.DataFrame(data=z_array,
                                    columns=self._mixture_components,
                                    index=self._alltimes)

        c_init = []
        if self._concentration_given:
            pass
        else:
            for t in self._allmeas_times:
                for k in self._mixture_components:
                    if t in self._meas_times:
                        if abs(self.model.init_conditions[k].value) > tol:
                            c_init.append(self.model.init_conditions[k].value)
                        else:
                            c_init.append(1.0)
                    else: c_init.append(float('nan')) #added for new huplc structure!

        if self._n_allmeas_times:
            if self._concentration_given:
                pass
            else:
                c_array = np.array(c_init).reshape((self._n_allmeas_times, self._n_components))
                c_init_panel = pd.DataFrame(data=c_array,
                                        columns=self._mixture_components,
                                        index=self._allmeas_times)

                self.initialize_from_trajectory('C', c_init_panel)
                print("self._n_meas_times is true in _default_init in PyomoSim")

        x_init = []
        for t in self._alltimes:
            for k in self._complementary_states:
                if abs(self.model.init_conditions[k].value) > tol:
                    x_init.append(self.model.init_conditions[k].value)
                else:
                    x_init.append(1.0)

        x_array = np.array(x_init).reshape((self._n_alltimes, self._n_complementary_states))
        x_init_panel = pd.DataFrame(data=x_array,
                                    columns=self._complementary_states,
                                    index=self._alltimes)

        self.initialize_from_trajectory('Z', z_init_panel)
        self.initialize_from_trajectory('X', x_init_panel)

    def initialize_parameters(self, params):
        for k, v in params.items():
            self.model.P[k].value = v

    def initialize_from_trajectory(self, variable_name, trajectories):
        """Initializes discretized points with values from trajectories.

        Args:
            variable_name (str): Name of the variable in pyomo model

            trajectories (DataFrame or Series): Indexed in in the same way the pyomo
            variable is indexed. If the variable is by two sets then the first set is
            the indices of the data frame, the second set is the columns

        Returns:
            None

        """

        if not self.model.alltime.get_discretization_info():
            raise RuntimeError('apply discretization first before initializing')

        if variable_name == 'Z':
            var = self.model.Z
            inner_set = self.model.alltime
        elif variable_name == 'dZdt':
            var = self.model.dZdt
            inner_set = self.model.alltime
        elif variable_name == 'C':
            var = self.model.C
            inner_set = self._allmeas_times
        elif variable_name =='Cs':
            var = self.model.Cs
            inner_set = self._allmeas_times
        elif variable_name == 'S':
            var = self.model.S
            inner_set = self._meas_lambdas
        elif variable_name == 'X':
            var = self.model.X
            inner_set = self.model.alltime
        elif variable_name == 'U':
            var = self.model.U
            inner_set = self._allmeas_times
        elif variable_name == 'dXdt':
            var = self.model.dXdt
            inner_set = self.model.alltime
        elif variable_name == 'Y':
            var = self.model.Y
            inner_set = self.model.alltime
        elif variable_name == 'dual':  # for warm start
            var = self.model.dual
            inner_set = self.model.alltime
        elif variable_name == 'Ps':
            var = self.model.Ps
            inner_set = self.model.alltime
        elif variable_name =='Chat':
            var = self.model.Chat
            inner_set = self.model.huplctime
        else:
            raise RuntimeError('Initialization of variable {} is not supported'.format(variable_name))

        columns = trajectories.columns

        to_initialize = list()
        if variable_name in ['Z', 'dZdt', 'S', 'C']:
            for component in columns:
                if component not in self._mixture_components:
                    print(
                        'WARNING: Mixture component {} is not in model mixture components. initialization ignored'.format(
                            component))
                else:
                    to_initialize.append(component)
        #added due to new structure for non_abs species, Cs as subset of C (CS):
        if variable_name in ['Cs']:
            for component in columns:
                if component not in self._mixture_components:
                    if component not in self._abs_components:
                        print(
                            'WARNING: Absorbing component {} is not in model absorbing components. initialization ignored'.format(
                            component))
                else:
                    to_initialize.append(component)

        #added for additional huplc data CS:
        if variable_name in ['Chat']:#, 'Dhat']:
            for component in columns:
                if component not in self._huplcabs_components:
                    print(
                        'WARNING: HUPLC absorbing component {} is not in model absorbing components. initialization ignored'.format(
                            component))
                else:
                    to_initialize.append(component)

        #added for additional smoothing data CS:
        if variable_name in ['Ps']:#, 'Dhat']:
            for component in columns:
                if component not in self.model.smoothparameter_names:
                    print(
                        'WARNING: Smoothing parameter component {} is not in Smoothing parameter components. initialization ignored'.format(
                            component))
                else:
                    to_initialize.append(component)


        if variable_name in ['X', 'dXdt', 'U']:
            for component in columns:
                if component not in self._complementary_states:
                    print('WARNING: State {} is not in model complementary_states. initialization ignored'.format(
                        component))
                else:
                    to_initialize.append(component)

        if variable_name in ['Y']:
            for component in columns:
                if component not in self._algebraics:
                    print('WARNING: Algebraic {} is not in model algebraics. initialization ignored'.format(component))
                else:
                    to_initialize.append(component)
        """            
        trajectory_times = np.array(trajectories.index)
        n_ttimes = len(trajectory_times)
        first_time = trajectory_times[0]
        last_time = trajectory_times[-1]
        for component in to_initialize:
            for t in inner_set:
                if t>=first_time and t<=last_time:
                    idx = find_nearest(trajectory_times,t)
                    t0 = trajectory_times[idx]
                    if t==t0:
                        if not np.isnan(trajectories[component][t0]):
                            var[t,component].value = trajectories[component][t0]
                        else:
                            var[t,component].value = None
                    else:
                        if t0==last_time:
                            if not np.isnan(trajectories[component][t0]):
                                var[t,component].value = trajectories[component][t0]
                            else:
                                var[t,component].value = None
                        else:
                            idx1 = idx+1
                            t1 = trajectory_times[idx1]
                            x_tuple = (t0,t1)
                            y_tuple = (trajectories[component][t0],trajectories[component][t1])
                            y = interpolate_linearly(t,x_tuple,y_tuple)
                            if not np.isnan(y):
                                var[t,component].value = y
                            else:
                                var[t,component].value = None
        """
        for component in to_initialize:
            single_trajectory = trajectories[component]
            for t in inner_set:
                val = interpolate_from_trajectory(t, single_trajectory)
                if not np.isnan(val):
                    var[t, component].value = val

    def scale_variables_from_trajectory(self, variable_name, trajectories):
        """Scales discretized variables with maximum value of the trajectory.

        Note:
            This method only works with ipopt

        Args:
            variable_name (str): Name of the variable in pyomo model

            trajectories (DataFrame or Series): Indexed in in the same way the pyomo
            variable is indexed. If the variable is by two sets then the first set is
            the indices of the data frame, the second set is the columns

        Returns:
            None

        """
        # time-invariant nominal scaling
        if not self.model.alltime.get_discretization_info():
            raise RuntimeError('apply discretization first before runing simulation')

        if variable_name == 'Z':
            var = self.model.Z
            inner_set = self.model.alltime
        elif variable_name == 'dZdt':
            var = self.model.dZdt
            inner_set = self.model.alltime
        elif variable_name == 'C':
            var = self.model.C
            inner_set = self._allmeas_times
        elif variable_name =='Cs':
            var = self.model.Cs
            inner_set = self._allmeas_times
        elif variable_name == 'S':
            var = self.model.S
            inner_set = self._meas_lambdas
        elif variable_name == 'X':
            var = self.model.X
            inner_set = self.model.alltime
        elif variable_name == 'U':
            var = self.model.U
            inner_set = self._allmeas_times
        elif variable_name == 'dXdt':
            var = self.model.dXdt
            inner_set = self.model.alltime
        elif variable_name == 'Y':
            var = self.model.Y
            inner_set = self.model.alltime
        elif variable_name =='Chat':
            var = self.model.Chat
            inner_set = self.model.huplctime
        else:
            raise RuntimeError('Scaling of variable {} is not supported'.format(variable_name))

        columns = trajectories.columns
        nominal_vals = dict()
        if variable_name in ['Z', 'dZdt', 'S', 'C']:
            for component in columns:
                nominal_vals[component] = abs(trajectories[component].max())
                if component not in self._mixture_components:
                    raise RuntimeError('Mixture component {} is not in model mixture components'.format(component))

        if variable_name in ['X', 'dXdt', 'U']:
            for component in columns:
                nominal_vals[component] = abs(trajectories[component].max())
                if component not in self._complementary_states:
                    raise RuntimeError('State {} is not in model complementary_states'.format(component))

        if variable_name in ['Y']:
            for component in columns:
                nominal_vals[component] = abs(trajectories[component].max())
                if component not in self._algebraics:
                    raise RuntimeError('Algebraics {} is not in model algebraics'.format(component))

        tol = 1e-5
        for component in columns:
            if nominal_vals[component] >= tol:
                scale = 1.0 / nominal_vals[component]
                for t in inner_set:
                    self.model.scaling_factor.set_value(var[t, component], scale)

        self._ipopt_scaled = True

    def validate(self):
        """Validates model before passing to solver.

        Note:
            TODO
        """
        pass

    def run_sim(self, solver, **kwds):
        """ Runs simulation by solving nonlinear system with ipopt

        Args:
            solver (str): name of the nonlinear solver to used

            solver_opts (dict, optional): Options passed to the nonlinear solver

            variances (dict, optional): Map of component name to noise variance. The
            map also contains the device noise variance

            tee (bool,optional): flag to tell the simulator whether to stream output
            to the terminal or not

        Returns:
            None

        """
        solver_opts = kwds.pop('solver_opts', dict())
        sigmas = kwds.pop('variances', dict())
        tee = kwds.pop('tee', False)
        seed = kwds.pop('seed', None)

        if not self.model.alltime.get_discretization_info():
            raise RuntimeError('apply discretization first before runing simulation')

        # adjusts the seed to reproduce results with noise
        np.random.seed(seed)

        # variables
        Z_var = self.model.Z
        dZ_var = self.model.dZdt
        P_var = self.model.P
        X_var = self.model.X
        U_var = self.model.U
        dX_var = self.model.dXdt
        C_var = self.model.C  # added for estimation with inputs and conc data CS
        if self._huplc_given: #added for additional data CS
            Dhat_var = self.model.Dhat
            Chat_var = self.model.Chat
        # check all parameters are fixed before simulating
        for p_var_data in six.itervalues(P_var):
            if not p_var_data.fixed:
                raise RuntimeError(
                    'For simulation fix all parameters. Parameter {} is unfixed'.format(p_var_data.cname()))

        # deactivates objective functions for simulation
        if self.model.nobjectives():
            objectives_map = self.model.component_map(ctype=Objective, active=True)
            active_objectives_names = []
            for obj in six.itervalues(objectives_map):
                name = obj.cname()
                active_objectives_names.append(name)
                str_warning = 'Deactivating objective {} for simulation'.format(name)
                warnings.warn(str_warning)
                obj.deactivate()

        # Look at the output in results
        # self.model.write('f.nl')
        opt = SolverFactory(solver)

        for key, val in solver_opts.items():
            opt.options[key] = val
            
        solver_results = opt.solve(self.model, tee=tee, symbolic_solver_labels=True)
        results = ResultsObject()

        # activates objective functions that were deactivated
        if self.model.nobjectives():
            active_objectives_names = []
            objectives_map = self.model.component_map(ctype=Objective)
            for name in active_objectives_names:
                objectives_map[name].activate()

        # retriving solutions to results object
        results.load_from_pyomo_model(self.model,
                                      to_load=['Z', 'dZdt', 'X', 'dXdt', 'Y'])

        c_noise_results = []

        w = np.zeros((self._n_components, self._n_allmeas_times))
        n_sig = np.zeros((self._n_components, self._n_allmeas_times))
        # for the noise term
        if sigmas:
            for i, k in enumerate(self._mixture_components):
                if k in sigmas.keys():
                    sigma = sigmas[k] ** 0.5
                    dw_k = np.random.normal(0.0, sigma, self._n_allmeas_times)
                    n_sig[i, :] = np.random.normal(0.0, sigma, self._n_allmeas_times)
                    w[i, :] = np.cumsum(dw_k)

        # this addition is not efficient but it can be changed later

        for i, t in enumerate(self._allmeas_times):
            for j, k in enumerate(self._mixture_components):
                # c_noise_results.append(Z_var[t,k].value+ w[j,i])
                c_noise_results.append(Z_var[t, k].value + n_sig[j, i])

        c_noise_array = np.array(c_noise_results).reshape((self._n_allmeas_times, self._n_components))
        results.C = pd.DataFrame(data=c_noise_array,
                                 columns=self._mixture_components,
                                 index=self._allmeas_times)

        #added due to new structure for non_abs species, Cs as subset of C (CS):
        if hasattr(self, '_abs_components'):
            cs_noise_results=[]
            for i, t in enumerate(self._allmeas_times):
                # if i in self._meas_times:
                for j, k in enumerate(self._abs_components):
                    # c_noise_results.append(Z_var[t,k].value+ w[j,i])
                    cs_noise_results.append(Z_var[t, k].value + n_sig[j, i])

            cs_noise_array = np.array(cs_noise_results).reshape((self._n_allmeas_times, self._nabs_components))
            results.Cs = pd.DataFrame(data=cs_noise_array,
                                     columns=self._abs_components,
                                     index=self._allmeas_times)

        # addition for inputs estimation with concentration data CS:
        if self._concentration_given == True and self._absorption_given == False:
            c_noise_results = []
            for i, t in enumerate(self._allmeas_times):
                # if i in self._meas_times:
                for j, k in enumerate(self._mixture_components):
                    c_noise_results.append(C_var[t, k].value)
            c_noise_array = np.array(c_noise_results).reshape((self._n_allmeas_times, self._n_components))
            results.C = pd.DataFrame(data=c_noise_array,
                                     columns=self._mixture_components,
                                     index=self._allmeas_times)

        if self._huplc_given == True:
            results.load_from_pyomo_model(self.model,
                                      to_load=['Chat'])
        s_results = []
        # added due to new structure for non_abs species, non-absorbing species not included in S (CS):
        if hasattr(self, '_abs_components'):
            for l in self._meas_lambdas:
                for k in self._abs_components:
                    s_results.append(self.model.S[l, k].value)
        else:
            for l in self._meas_lambdas:
                for k in self._mixture_components:
                    s_results.append(self.model.S[l, k].value)

        d_results = []
        if sigmas:
            sigma_d = sigmas.get('device') ** 0.5 if "device" in sigmas.keys() else 0
        else:
            sigma_d = 0
        if s_results and c_noise_results:
            # added due to new structure for non_abs species, Cs and S as above(CS):
            if hasattr(self,'_abs_components'):
                for i, t in enumerate(self._meas_times):
                    #if t in self._meas_times:
                        # print(i, t)
                    for j, l in enumerate(self._meas_lambdas):
                        suma = 0.0
                        for w, k in enumerate(self._abs_components):
                            # print(i, self._meas_times)
                            Cs = cs_noise_results[i * self._nabs_components + w]
                            S = s_results[j * self._nabs_components + w]
                            suma += Cs * S
                        if sigma_d:
                            suma += np.random.normal(0.0, sigma_d)
                        d_results.append(suma)
                        # print(d_results)
            else:
                for i, t in enumerate(self._meas_times):
                    # # print(i, t)
                    # if t in self._meas_times:
                    for j, l in enumerate(self._meas_lambdas):
                        suma = 0.0
                        for w, k in enumerate(self._mixture_components):
                            # print(i, self._meas_times)
                            C = c_noise_results[i * self._n_components + w]
                            S = s_results[j * self._n_components + w]
                            suma += C * S
                        if sigma_d:
                            suma += np.random.normal(0.0, sigma_d)
                        d_results.append(suma)
                        # print(d_results)
        # added due to new structure for non_abs species, non-absorbing species not included in S (CS):
        if hasattr(self, '_abs_components'):
            s_array = np.array(s_results).reshape((self._n_meas_lambdas, self._nabs_components))
            results.S = pd.DataFrame(data=s_array,
                                     columns=self._abs_components,
                                     index=self._meas_lambdas)
        else:
            s_array = np.array(s_results).reshape((self._n_meas_lambdas, self._n_components))
            results.S = pd.DataFrame(data=s_array,
                                     columns=self._mixture_components,
                                     index=self._meas_lambdas)


        d_array = np.array(d_results).reshape((self._n_meas_times, self._n_meas_lambdas))
        results.D = pd.DataFrame(data=d_array,
                                 columns=self._meas_lambdas,
                                 index=self._meas_times)

        s_data_dict = dict()
        for t in self._meas_times:
            for l in self._meas_lambdas:
                s_data_dict[t, l] = float(results.D[l][t])

        # Added due to estimation with fe-factory and inputs where data already loaded to model before (CS)
        if self._spectra_given:
            self.model.del_component(self.model.D)
            self.model.del_component(self.model.D_index)
        #########

        self.model.D = Param(self._meas_times,
                             self._meas_lambdas,
                             initialize=s_data_dict)

        param_vals = dict()
        for name in self.model.parameter_names:
            param_vals[name] = self.model.P[name].value

        results.P = param_vals
        return results



