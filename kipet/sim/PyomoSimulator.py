from pyomo.environ import *
from pyomo.dae import *
from ResultsObject import *
from Simulator import *
import warnings

class PyomoSimulator(Simulator):
    def __init__(self,model):
        super(PyomoSimulator, self).__init__(model)
        self._times = sorted(self.model.time)
        self._n_times = len(self._times)
        self._spectra_given = hasattr(self.model, 'D')
        self._ipopt_scaled = False
        # creates scaling factor suffix
        self.model.scaling_factor = Suffix(direction=Suffix.EXPORT)

        # this is very rigid probably need to change
        # checks model has init_conditions_set
        init_cond = self.model.init_conditions_c

        if init_cond.active == False:
            raise RuntimeError('initial condition deactivated. Please activate initial conditions')
                
    def apply_discretization(self,transformation,**kwargs):
        discretizer = TransformationFactory(transformation)
        discretizer.apply_to(self.model,wrt=self.model.time,**kwargs)
        self._times = sorted(self.model.time)
        self._n_times = len(self._times)
        self._discretized = True
        self._default_initialization()
        
    # initializes the trajectories to the initial conditions
    def _default_initialization(self):

        tol =1e-3
        z_init = []
        for t in self._times:
            for k in self._mixture_components:
                if abs(self.model.init_conditions[k])>tol:
                    z_init.append(self.model.init_conditions[k])
                else:
                    z_init.append(1.0)

        z_array = np.array(z_init).reshape((self._n_times,self._n_components))
        z_init_panel = pd.DataFrame(data=z_array,
                                 columns=self._mixture_components,
                                 index=self._times)

        c_init = []
        for t in self._meas_times:
            for k in self._mixture_components:
                if abs(self.model.init_conditions[k])>tol:
                    c_init.append(self.model.init_conditions[k])
                else:
                    c_init.append(1.0)

        if self._n_meas_times:
            c_array = np.array(c_init).reshape((self._n_meas_times,self._n_components))
            c_init_panel = pd.DataFrame(data=c_array,
                                        columns=self._mixture_components,
                                        index=self._meas_times)
            self.initialize_from_trajectory('C',c_init_panel)

    
        x_init = []
        for t in self._times:
            for k in self._complementary_states:
                if abs(self.model.init_conditions[k])>tol:
                    x_init.append(self.model.init_conditions[k])
                else:
                    x_init.append(1.0)

        x_array = np.array(x_init).reshape((self._n_times,self._n_complementary_states))
        x_init_panel = pd.DataFrame(data=x_array,
                                 columns=self._complementary_states,
                                 index=self._times)

        self.initialize_from_trajectory('Z',z_init_panel)
        self.initialize_from_trajectory('X',x_init_panel)
        
    def initialize_from_trajectory(self,variable_name,trajectories):
        if self._discretized is False:
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
        else:
            raise RuntimeError('Initialization of variable {} is not supported'.format(variable_name))

        columns = trajectories.columns

        if variable_name in ['Z','dZdt','S','C']:
            for component in columns:
                if component not in self._mixture_components:
                    raise RuntimeError('Mixture component {} is not in model mixture components'.format(component))

        if variable_name in ['X','dXdt']:
            for component in columns:
                if component not in self._complementary_states:
                    raise RuntimeError('State {} is not in model complementary_states'.format(component))

        trajectory_times = np.array(trajectories.index)
        n_ttimes = len(trajectory_times)
        first_time = trajectory_times[0]
        last_time = trajectory_times[-1]
        for component in columns:
            for t in inner_set:
                if t>=first_time and t<=last_time:
                    idx = find_nearest(trajectory_times,t)
                    t0 = trajectory_times[idx]
                    if t==t0:
                        var[t,component].value = trajectories[component][t0]
                    else:
                        if t0==last_time:
                            var[t,component].value = trajectories[component][t0]
                        else:
                            idx1 = idx+1
                            t1 = trajectory_times[idx1]
                            x_tuple = (t0,t1)
                            y_tuple = (trajectories[component][t0],trajectories[component][t1])
                            y = interpolate_linearly(t,x_tuple,y_tuple)
                            var[t,component].value = y

    def scale_variables_from_trajectory(self,variable_name,trajectories):
        # time-invariant nominal scaling
        # this method works only with ipopt
        if self._discretized is False:
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
            var = self.model.dZdt
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
        
        tol = 1e-5
        for component in columns:
            if nominal_vals[component]>= tol:
                scale = 1.0/nominal_vals[component]
                for t in inner_set:
                    self.model.scaling_factor.set_value(var[t,component],scale)

        self._ipopt_scaled = True
            
    def validate(self):
        pass
        
    def run_sim(self,solver,tee=False,solver_opts={},sigmas=None,seed=None):

        if self._discretized is False:
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
        for p_var_data in P_var.itervalues():
            if not p_var_data.fixed:
                raise RuntimeError('For simulation fix all parameters. Parameter {} is unfixed'.format(p_var_data.cname()))

        # deactivates objective functions for simulation
        if self.model.nobjectives():
            objectives_map = self.model.component_map(ctype=Objective,active=True)
            active_objectives_names = []
            for obj in objectives_map.itervalues():
                name = obj.cname()
                active_objectives_names.append(name)
                str_warning = 'Deactivating objective {} for simulation'.format(name)
                warnings.warn(str_warning)
                obj.deactivate()

        # Look at the output in results
        #self.model.write('f.nl')
        opt = SolverFactory(solver)

        for key, val in solver_opts.iteritems():
            opt.options[key]=val
            
        solver_results = opt.solve(self.model,tee=tee)
        results = ResultsObject()

        # activates objective functions that were deactivated
        if self.model.nobjectives():
            active_objectives_names = []
            objectives_map = self.model.component_map(ctype=Objective)
            for name in active_objectives_names:
                objectives_map[name].activate()


        # retriving solutions to results object
            
        c_results = []
        for t in self._times:
            for k in self._mixture_components:
                c_results.append(Z_var[t,k].value)

        c_array = np.array(c_results).reshape((self._n_times,self._n_components))
        
        results.Z = pd.DataFrame(data=c_array,
                                 columns=self._mixture_components,
                                 index=self._times)

        dc_results = []
        for t in self._times:
            for k in self._mixture_components:
                dc_results.append(dZ_var[t,k].value)

        dc_array = np.array(dc_results).reshape((self._n_times,self._n_components))
        
        results.dZdt = pd.DataFrame(data=dc_array,
                                 columns=self._mixture_components,
                                 index=self._times)

        x_results = []
        for t in self._times:
            for k in self._complementary_states:
                x_results.append(X_var[t,k].value)

        x_array = np.array(x_results).reshape((self._n_times,self._n_complementary_states))
        
        results.X = pd.DataFrame(data=x_array,
                                 columns=self._complementary_states,
                                 index=self._times)

        dx_results = []
        for t in self._times:
            for k in self._complementary_states:
                dx_results.append(dX_var[t,k].value)

        dx_array = np.array(dx_results).reshape((self._n_times,self._n_complementary_states))
        
        results.dXdt = pd.DataFrame(data=dx_array,
                                 columns=self._complementary_states,
                                 index=self._times)

        c_noise_results = []

        w = np.zeros((self._n_components,self._n_meas_times))
        # for the noise term
        if sigmas:
            for i,k in enumerate(self._mixture_components):
                if sigmas.has_key(k):
                    sigma = sigmas[k]**0.5
                    dw_k = np.random.normal(0.0,sigma,self._n_meas_times)
                    w[i,:] = np.cumsum(dw_k)

        # this addition is not efficient but it can be changed later
        for i,t in enumerate(self._meas_times):
            for j,k in enumerate(self._mixture_components):
                c_noise_results.append(Z_var[t,k].value+ w[j,i])

        c_noise_array = np.array(c_noise_results).reshape((self._n_meas_times,self._n_components))
        results.C = pd.DataFrame(data=c_noise_array,
                                 columns=self._mixture_components,
                                 index=self._meas_times)
        
        if self._spectra_given: 

            D_data = self.model.D
            
            if self._n_meas_times and self._n_meas_times<self._n_components:
                raise RuntimeError('Not enough measurements num_meas>= num_components')

            # solves over determined system
            s_array = self._solve_S_from_DC(results.C,tee=tee)

            d_results = []
            for t in self._meas_times:
                for l in self._meas_lambdas:
                    d_results.append(D_data[t,l])
            d_array = np.array(d_results).reshape((self._n_meas_times,self._n_meas_lambdas))
                        
            results.S = pd.DataFrame(data=s_array,
                                     columns=self._mixture_components,
                                     index=self._meas_lambdas)

            results.D = pd.DataFrame(data=d_array,
                                     columns=self._meas_lambdas,
                                     index=self._meas_times)

            for t in self.model.meas_times:
                for k in self._mixture_components:
                    self.model.C[t,k].value = results.C[k][t]

            for l in self.model.meas_lambdas:
                for k in self._mixture_components:
                    self.model.S[l,k].value =  results.S[k][l]
            
        else:
                    
            s_results = []
            for l in self._meas_lambdas:
                for k in self._mixture_components:
                    s_results.append(self.model.S[l,k].value)

            d_results = []
            if sigmas:
                sigma_d = sigmas.get('device')**0.5 if sigmas.has_key('device') else 0
            else:
                sigma_d = 0
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
                    
            
            s_array = np.array(s_results).reshape((self._n_meas_lambdas,self._n_components))
            results.S = pd.DataFrame(data=s_array,
                                     columns=self._mixture_components,
                                     index=self._meas_lambdas)
                        
            d_array = np.array(d_results).reshape((self._n_meas_times,self._n_meas_lambdas))
            results.D = pd.DataFrame(data=d_array,
                                     columns=self._meas_lambdas,
                                     index=self._meas_times)
            
        param_vals = dict()
        for name in self.model.parameter_names:
            param_vals[name] = self.model.P[name].value

        results.P = param_vals
        return results
        


