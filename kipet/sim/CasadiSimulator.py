import casadi as ca
from casadi.tools import *
from ResultsObject import *
from Simulator import *
import copy


class CasadiSimulator(Simulator):
    """Simulator based on pyomo.dae discretization strategies.

    Attributes:
        model (Casadi model)
        
        _times (array_like): array of times after discretization
        
        _n_times (int): number of discretized time points
        
        nfe (int): number of finite element points to split the time
        horizon. The simulator will return the solution in those points
    """
    
    def __init__(self,model):
        """Simulator constructor.

        Args:
            model (Casadi model)
        """
        super(CasadiSimulator, self).__init__(model)
        self.nfe = None
        self._times = set([t for t in model.meas_times])
        self._n_times = len(self._times)
        self._spectra_given = hasattr(self.model, 'D')
        self._fixed_variables = list()
        self._fixed_trajectories = list()
        self._fixed_variable_names = list()
        
    def apply_discretization(self,transformation,**kwargs):
        """Defines discrete points to evaluate integrator.

        Args:
            transformation (str): TODO

            **kwargs: Arbitrary keyword arguments
            nfe (int): number of points to split the time domain
        
        Returns:
            None
        """
        self.nfe = kwargs.pop('nfe',1)
        self.model.start_time
        step = (self.model.end_time - self.model.start_time)/self.nfe
        for i in xrange(0,self.nfe+1):
            self._times.add(i*step)
                
        self._n_times = len(self._times)
        self._discretized = True 
        
    def initialize_from_trajectory(self,variable_name,trajectories):
        raise NotImplementedError("CasadiSimulator does not support initialization")

    def fix_from_trajectory(self,variable_name,variable_index,trajectories):

        single_traj = trajectories[variable_index]
        times_traj = np.array(single_traj.index)
        last_time_idx = len(times_traj)-1
        sim_times = sorted(self._times)
        data = np.zeros(self._n_times)
        for i,t in enumerate(sim_times):
            idx_near = find_nearest(times_traj,t)
            if idx_near==0 or idx_near==last_time_idx:
                t_found = times_traj[idx_near]
                data[i] = single_traj[t_found]
            else:
                idx_near1 = idx_near+1
                t_found = times_traj[idx_near]
                t_found1 = times_traj[idx_near1]
                val = single_traj[t_found]
                val1 = single_traj[t_found1]
                x_tuple = (t_found,t_found1)
                y_tuple = (val,val1)
                data[i] = interpolate_linearly(t,x_tuple,y_tuple)

        var = getattr(self.model,variable_name)
        symbolic = var[variable_index]
        self._fixed_variable_names.append((variable_name,variable_index))
        self._fixed_variables.append(symbolic)
        fixed_trajectory = pd.Series(data=data,index=sim_times)
        self._fixed_trajectories.append(fixed_trajectory)
        
    
    def run_sim(self,solver,**kwds):
        """ Runs simulation by solving nonlinear system with ipopt

        Args:
            solver (str): name of the integrator to used (CVODES or IDAS)

            **kwargs: Arbitrary keyword arguments
            solver_opts (dict): Options passed to the integrator
            
            variances (dict): Map of component name to noise variance. The
            map also contains the device noise variance
            
            tee (bool): flag to tell the simulator whether to stream output
            to the terminal or not
                    
        Returns:
            None

        """
        
        solver_opts = kwds.pop('solver_opts', dict())
        sigmas = kwds.pop('variances',dict())
        tee = kwds.pop('tee',False)
        seed = kwds.pop('seed',None)
        
        # adjusts the seed to reproduce results with noise
        np.random.seed(seed)
        
        Z_var = self.model.Z
        X_var = self.model.X
        
        if self._discretized is False:
            raise RuntimeError('apply discretization first before runing simulation')
        states_l = []
        ode_l = []
        init_conditions_l = []


        for i,k in enumerate(self._mixture_components):
            states_l.append(Z_var[k])

            expr = self.model.odes[k]
            
            if isinstance(expr,ca.SX):
                representation = expr.getRepresentation()
            else:
                representation = str(expr)
            if 'nan' not in representation:
                ode_l.append(expr)
            else:
                raise RuntimeError('Mass balance expression for {} is nan.\n'.format(k)+
                'This usually happens when not using casadi.operator\n'+ 
                'e.g casadi.exp(expression)\n')
            init_conditions_l.append(self.model.init_conditions[k])

        for i,k in enumerate(self._complementary_states):
            states_l.append(X_var[k])
            expr = self.model.odes[k]
            if isinstance(expr,ca.SX):
                representation = expr.getRepresentation()
            else:
                representation = str(expr)
            if 'nan' not in representation:
                ode_l.append(expr)
            else:
                raise RuntimeError('Complementary ode expression for {} is nan.\n'.format(k)+
                'This usually happens when not using casadi.operator\n'+ 
                'e.g casadi.exp(expression)')
            init_conditions_l.append(self.model.init_conditions[k])

        states = ca.vertcat(*states_l)
        ode = ca.vertcat(*ode_l)
        x_0 = ca.vertcat(*init_conditions_l)

        system = {'x':states, 'ode':ode}
        
        step = (self.model.end_time - self.model.start_time)/self.nfe

        results = ResultsObject()

        c_results =  []
        dc_results = []

        x_results =  []
        dx_results = []

        xk = x_0
        times = sorted(self._times)
        for i,t in enumerate(times):

            sub_odes = ode
            #print sub_odes
            for s,var in enumerate(self._fixed_variables):
                value = self._fixed_trajectories[s][t]
                sub_odes = ca.substitute(sub_odes,var,value)
            #print sub_odes
            fun_ode = ca.Function("odeFunc",[states],[sub_odes])
            
            if t == self.model.start_time:
                odek = fun_ode(xk)
                for j,w in enumerate(init_conditions_l):
                    if j<self._n_components:
                        c_results.append(w)
                        dc_results.append(odek[j])
                    else:
                        x_results.append(w)
                        dx_results.append(odek[j])
            else:
                step = t - times[i-1]
                opts = {'tf':step,'print_stats':tee,'verbose':False}
                I = integrator("I",solver, system, opts)
                xk = I(x0=xk)['xf']

                # check for nan
                for j in xrange(xk.numel()):
                    if np.isnan(float(xk[j])):
                        raise RuntimeError('The iterator returned nan. exiting the program')
                    
                odek = fun_ode(xk)
                
                for j,k in enumerate(self._mixture_components):
                    c_results.append(xk[j])
                    dc_results.append(odek[j])

                for i,k in enumerate(self._complementary_states):
                    j = i+self._n_components
                    x_results.append(xk[j])
                    dx_results.append(odek[j])
        
        c_array = np.array(c_results).reshape((self._n_times,self._n_components))
        results.Z = pd.DataFrame(data=c_array,columns=self._mixture_components,index=times)
                    
        dc_array = np.array(dc_results).reshape((self._n_times,self._n_components))
        results.dZdt = pd.DataFrame(data=dc_array,columns=self._mixture_components,index=times)

        x_array = np.array(x_results).reshape((self._n_times,self._n_complementary_states))
        results.X = pd.DataFrame(data=x_array,columns=self._complementary_states,index=times)

        dx_array = np.array(dx_results).reshape((self._n_times,self._n_complementary_states))
        results.dXdt = pd.DataFrame(data=dx_array,columns=self._complementary_states,index=times)

        # get the fixed series map in the results
        for s,pair in enumerate(self._fixed_variable_names):
            var_name = pair[0]
            var_idx = pair[1]
            var = getattr(results,var_name)
            serie = var[var_idx]
            fixed_serie = self._fixed_trajectories[s] 
            for t in times:
                serie[t] = fixed_serie[t] 
        
        w = np.zeros((self._n_components,self._n_meas_times))
        # for the noise term
        if sigmas:
            for i,k in enumerate(self._mixture_components):
                if sigmas.has_key(k):
                    sigma = sigmas[k]**0.5
                    dw_k = np.random.normal(0.0,sigma,self._n_meas_times)
                    w[i,:] = np.cumsum(dw_k)
            
        c_noise_results = []
        for i,t in enumerate(self._meas_times):
            for j,k in enumerate(self._mixture_components):
                c_noise_results.append(results.Z[k][t]+w[j,i])
        
        c_noise_array = np.array(c_noise_results).reshape((self._n_meas_times,self._n_components))
        results.C = pd.DataFrame(data=c_noise_array,
                                 columns=self._mixture_components,
                                 index=self._meas_times)

        
        if self._spectra_given:
            # solves over determined system
            D_data = self.model.D
            s_array = self._solve_S_from_DC(results.C,tee=tee)

            d_results = []
            for t in self._meas_times:
                for l in self._meas_lambdas:
                    d_results.append(D_data[t,l])
                    
            d_array = np.array(d_results).reshape((self._n_meas_times,self._n_meas_lambdas))
        else:

            s_results = []
            for l in self._meas_lambdas:
                for k in self._mixture_components:
                    s_results.append(self.model.S[l,k])

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
            d_array = np.array(d_results).reshape((self._n_meas_times,self._n_meas_lambdas))
            
        # stores everything in restuls object
        results.S = pd.DataFrame(data=s_array,
                                 columns=self._mixture_components,
                                 index=self._meas_lambdas)
        results.D = pd.DataFrame(data=d_array,
                                 columns=self._meas_lambdas,
                                 index=self._meas_times)
        
        param_vals = dict()
        for name in self.model.parameter_names:
            param_vals[name] = self.model.P[name]

        results.P = param_vals
        
        return results
        
        
        
