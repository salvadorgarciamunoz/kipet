from pyomo.environ import *
from pyomo.dae import *
from kipet.library.ResultsObject import *
from kipet.library.Simulator import *
import warnings
import six
import sys


class PyomoSimulator(Simulator):
    """Simulator based on pyomo.dae discretization strategies.

    Attributes:
        model (Pyomo model)
        
        _times (array_like): array of times after discretization
        
        _n_times (int): number of discretized time points
        
        _ipopt_scaled (bool): flag that indicates if there are 
        ipopt scaling factors specified 
    """

    def __init__(self,model):
        """Simulator constructor.

        Note: 
            Makes a shallow copy to the model. Changes applied to 
            the model within the simulator are applied to the original
            model passed to the simulator

        Args:
            model (Pyomo model)
        """
        super(PyomoSimulator, self).__init__(model)
        self._times = sorted(self.model.time)
        self._n_times = len(self._times)
        self._ipopt_scaled = False
        self._spectra_given = hasattr(self.model, 'D')
        self._concentration_given = hasattr(self.model, 'C')
        #creates scaling factor suffix
        if not hasattr(self.model, 'scaling_factor'):
            self.model.scaling_factor = Suffix(direction=Suffix.EXPORT)

    def apply_discretization(self,transformation,**kwargs):
        """Discretizes the model.

        Args:
            transformation (str): TODO
            same keywords as in pyomo.dae method
        
        Returns:
            None
        """
        if not self.model.time.get_discretization_info():
            discretizer = TransformationFactory(transformation)
            discretizer.apply_to(self.model,wrt=self.model.time,**kwargs)
            self._times = sorted(self.model.time)
            self._n_times = len(self._times)
            self._default_initialization()
        else:
            print('***WARNING: Model already discretized. Ignoring second discretization')

    def fix_from_trajectory(self,variable_name,variable_index,trajectories):

        if variable_name in ['X','dXdt','Z','dZdt']:
            raise NotImplementedError("Fixing state variables is not allowd. Only algebraics can be fixed")
        
        single_traj = trajectories[variable_index]
        sim_times = sorted(self._times)
        var = getattr(self.model,variable_name)
        for i,t in enumerate(sim_times):
            value = interpolate_from_trayectory(t,single_traj)
            var[t,variable_index].fix(value)

    def unfix_time_dependent_variable(self,variable_name,variable_index):
        var = getattr(self.model,variable_name)
        sim_times = sorted(self._times)
        for i,t in enumerate(sim_times):
            var[t,variable_index].fixed = False
            
    # initializes the trajectories to the initial conditions
    def _default_initialization(self):
        """Initializes discreted variables model with initial condition values.

           This method is not intended to be used by users directly
        Args:
            None
        
        Returns:
            None
        """
        tol =1e-4
        z_init = []
        for t in self._times:
            for k in self._mixture_components:
                if abs(self.model.init_conditions[k].value)>tol:
                    z_init.append(self.model.init_conditions[k].value)
                else:
                    z_init.append(1.0)

        z_array = np.array(z_init).reshape((self._n_times,self._n_components))
        z_init_panel = pd.DataFrame(data=z_array,
                                 columns=self._mixture_components,
                                 index=self._times)
        
        c_init = []
        if self._concentration_given:
            pass
        else:
            for t in self._meas_times:
                for k in self._mixture_components:
                    if abs(self.model.init_conditions[k].value)>tol:
                        c_init.append(self.model.init_conditions[k].value)
                    else:
                        c_init.append(1.0)

        if self._n_meas_times:
            if self._concentration_given:
                pass
            else:
                c_array = np.array(c_init).reshape((self._n_meas_times,self._n_components))
                c_init_panel = pd.DataFrame(data=c_array,
                                        columns=self._mixture_components,
                                        index=self._meas_times)
                self.initialize_from_trajectory('C',c_init_panel)
                print("self._n_meas_times is true in _default_init in PyomoSim")
    
        x_init = []
        for t in self._times:
            for k in self._complementary_states:
                if abs(self.model.init_conditions[k].value)>tol:
                    x_init.append(self.model.init_conditions[k].value)
                else:
                    x_init.append(1.0)

        x_array = np.array(x_init).reshape((self._n_times,self._n_complementary_states))
        x_init_panel = pd.DataFrame(data=x_array,
                                 columns=self._complementary_states,
                                 index=self._times)

        self.initialize_from_trajectory('Z',z_init_panel)
        self.initialize_from_trajectory('X',x_init_panel)

    def initialize_parameters(self,params):
        for k,v in params.items():
            self.model.P[k].value = v
            
    def initialize_from_trajectory(self,variable_name,trajectories):
        """Initializes discretized points with values from trajectories.

        Args:
            variable_name (str): Name of the variable in pyomo model
            
            trajectories (DataFrame or Series): Indexed in in the same way the pyomo 
            variable is indexed. If the variable is by two sets then the first set is
            the indices of the data frame, the second set is the columns

        Returns:
            None

        """

        if not self.model.time.get_discretization_info():
            raise RuntimeError('apply discretization first before initializing')
        
        if variable_name == 'Z':
            var = self.model.Z
            inner_set = self.model.time
        elif variable_name == 'dZdt':
            var = self.model.dZdt
            inner_set = self.model.time
        elif variable_name == 'C':
            var = self.model.C
            inner_set = self._meas_times
        elif variable_name == 'S':
            var = self.model.S
            inner_set = self._meas_lambdas
        elif variable_name == 'X':
            var = self.model.X
            inner_set = self.model.time
        elif variable_name == 'dXdt':
            var = self.model.dXdt
            inner_set = self.model.time
        elif variable_name == 'Y':
            var = self.model.Y
            inner_set = self.model.time
        else:
            raise RuntimeError('Initialization of variable {} is not supported'.format(variable_name))

        columns = trajectories.columns

        to_initialize = list()
        if variable_name in ['Z','dZdt','S','C']:
            for component in columns:
                if component not in self._mixture_components:
                    print('WARNING: Mixture component {} is not in model mixture components. initialization ignored'.format(component))
                else:
                    to_initialize.append(component)
                
        if variable_name in ['X','dXdt']:
            for component in columns:
                if component not in self._complementary_states:
                    print('WARNING: State {} is not in model complementary_states. initialization ignored'.format(component))
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
                val = interpolate_from_trayectory(t,single_trajectory)
                if not np.isnan(val):
                    var[t,component].value = val
                
    def scale_variables_from_trajectory(self,variable_name,trajectories):
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
        if not self.model.time.get_discretization_info():
            raise RuntimeError('apply discretization first before runing simulation')
        
        if variable_name == 'Z':
            var = self.model.Z
            inner_set = self.model.time
        elif variable_name == 'dZdt':
            var = self.model.dZdt
            inner_set = self.model.time
        elif variable_name == 'C':
            var = self.model.C
            inner_set = self._meas_times
        elif variable_name == 'S':
            var = self.model.S
            inner_set = self._meas_lambdas
        elif variable_name == 'X':
            var = self.model.X
            inner_set = self.model.time
        elif variable_name == 'dXdt':
            var = self.model.dXdt
            inner_set = self.model.time
        elif variable_name == 'Y':
            var = self.model.Y
            inner_set = self.model.time
        else:
            raise RuntimeError('Scaling of variable {} is not supported'.format(variable_name))

        columns = trajectories.columns
        nominal_vals = dict()
        if variable_name in ['Z','dZdt','S','C']:
            for component in columns:
                nominal_vals[component] = abs(trajectories[component].max())
                if component not in self._mixture_components:
                    raise RuntimeError('Mixture component {} is not in model mixture components'.format(component))

        if variable_name in ['X','dXdt']:
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
            if nominal_vals[component]>= tol:
                scale = 1.0/nominal_vals[component]
                for t in inner_set:
                    self.model.scaling_factor.set_value(var[t,component],scale)

        self._ipopt_scaled = True
            
    def validate(self):
        """Validates model before passing to solver.
        
        Note:
            TODO
        """
        pass
        
    def run_sim(self,solver,**kwds):
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
        sigmas = kwds.pop('variances',dict())
        tee = kwds.pop('tee',False)
        seed = kwds.pop('seed',None)
        
        if not self.model.time.get_discretization_info():
            raise RuntimeError('apply discretization first before runing simulation')

        # adjusts the seed to reproduce results with noise
        np.random.seed(seed)
        
        # variables
        Z_var = self.model.Z
        dZ_var = self.model.dZdt
        P_var = self.model.P            
        X_var = self.model.X
        dX_var = self.model.dXdt
        
        # check all parameters are fixed before simulating
        for p_var_data in six.itervalues(P_var):
            if not p_var_data.fixed:
                raise RuntimeError('For simulation fix all parameters. Parameter {} is unfixed'.format(p_var_data.cname()))

        # deactivates objective functions for simulation
        if self.model.nobjectives():
            objectives_map = self.model.component_map(ctype=Objective,active=True)
            active_objectives_names = []
            for obj in six.itervalues(objectives_map):
                name = obj.cname()
                active_objectives_names.append(name)
                str_warning = 'Deactivating objective {} for simulation'.format(name)
                warnings.warn(str_warning)
                obj.deactivate()

        # Look at the output in results
        #self.model.write('f.nl')
        opt = SolverFactory(solver)

        for key, val in solver_opts.items():
            opt.options[key]=val
            
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
                                      to_load=['Z','dZdt','X','dXdt','Y'])
        
        c_noise_results = []

        w = np.zeros((self._n_components,self._n_meas_times))
        n_sig = np.zeros((self._n_components,self._n_meas_times))
        # for the noise term
        if sigmas:
            for i,k in enumerate(self._mixture_components):
                if k in sigmas.keys():
                    sigma = sigmas[k]**0.5
                    dw_k = np.random.normal(0.0,sigma,self._n_meas_times)
                    n_sig[i,:] = np.random.normal(0.0,sigma,self._n_meas_times)
                    w[i,:] = np.cumsum(dw_k)

        # this addition is not efficient but it can be changed later
        for i,t in enumerate(self._meas_times):
            for j,k in enumerate(self._mixture_components):
                #c_noise_results.append(Z_var[t,k].value+ w[j,i])
                c_noise_results.append(Z_var[t,k].value+ n_sig[j,i])

        c_noise_array = np.array(c_noise_results).reshape((self._n_meas_times,self._n_components))
        results.C = pd.DataFrame(data=c_noise_array,
                                 columns=self._mixture_components,
                                 index=self._meas_times)
        
        s_results = []
        for l in self._meas_lambdas:
            for k in self._mixture_components:
                s_results.append(self.model.S[l,k].value)

        d_results = []
        if sigmas:
            sigma_d = sigmas.get('device')**0.5 if "device" in sigmas.keys() else 0
        else:
            sigma_d = 0
        #print(self._meas_times)
        if s_results and c_noise_results:
            for i,t in enumerate(self._meas_times):
                for j,l in enumerate(self._meas_lambdas):
                    suma = 0.0
                    for w,k in enumerate(self._mixture_components):
                        C = c_noise_results[i*self._n_components+w]
                        S = s_results[j*self._n_components+w]
                        suma+= C*S
                    if sigma_d:
                        suma+= np.random.normal(0.0,sigma_d)
                    d_results.append(suma)
        #print('len(d_results):',len(d_results))

                    
            
        s_array = np.array(s_results).reshape((self._n_meas_lambdas,self._n_components))
        #print('s_array shape:', s_array.shape)

        results.S = pd.DataFrame(data=s_array,
                                 columns=self._mixture_components,
                                 index=self._meas_lambdas)
        #print(self._n_meas_times)
                        
        d_array = np.array(d_results).reshape((self._n_meas_times,self._n_meas_lambdas))
        print('d_array shape:', d_array.shape)
        results.D = pd.DataFrame(data=d_array,
                                 columns=self._meas_lambdas,
                                 index=self._meas_times)
        #print(self._meas_times)
            
        s_data_dict = dict()
        for t in self._meas_times:
            for l in self._meas_lambdas:
                s_data_dict[t,l] = float(results.D[l][t])

        #Added due to estimation with fe-factory and inputs where data already loaded to model before
        if self._spectra_given:
            self.model.del_component(self.model.D)
            self.model.del_component(self.model.D_index)

        self.model.D = Param(self._meas_times,
                             self._meas_lambdas,
                             initialize = s_data_dict)

        param_vals = dict()
        for name in self.model.parameter_names:
            param_vals[name] = self.model.P[name].value

        results.P = param_vals
        return results
        


