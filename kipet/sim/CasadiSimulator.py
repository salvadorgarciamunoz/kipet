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

        if variable_name in ['X','dXdt','Z','dZdt']:
            raise NotImplementedError("Fixing state variables is not allowd. Only algebraics can be fixed")
        
        single_traj = trajectories[variable_index]
        sim_times = sorted(self._times)
        data = np.zeros(self._n_times)
        for i,t in enumerate(sim_times):
            data[i] = interpolate_from_trayectory(t,single_traj)
            
        var = getattr(self.model,variable_name)
        symbolic = var[variable_index]
        self._fixed_variable_names.append((variable_name,variable_index))
        self._fixed_variables.append(symbolic)
        fixed_trajectory = pd.Series(data=data,index=sim_times)
        self._fixed_trajectories.append(fixed_trajectory)

    def unfix_time_dependent_variable(self,variable_name,variable_index):
        var = getattr(self.model,variable_name)
        symbolic = var[variable_index]
        index = -1
        for i,v in enumerate(self._fixed_variable_names):
            if v == (variable_name,variable_index):
                index=i

        if index>0:
            del self._fixed_variable_names[index]
            del self._fixed_variables[index]
            del self._fixed_trajectories[index]
        else:
            print("WARNING: Variable {}[t,{}] not fixed".format(variable_name,variable_index))
            
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
        init_guess_y = kwds.pop('y0',dict())
        
        # adjusts the seed to reproduce results with noise
        np.random.seed(seed)
        
        Z_var = self.model.Z
        X_var = self.model.X
        Y_var = self.model.Y
        
        if self._discretized is False:
            raise RuntimeError('apply discretization first before runing simulation')
        states_l = []
        algebraics_l = []
        algebraic_eq_l = []
        ode_l = []
        init_conditions_l = []
        y_guess_l = []
        
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

        unfixed_names = list()
        fixed_names = list()
        for i,k in enumerate(self._algebraics):
            fixed = False
            for fv in self._fixed_variables:
                if ca.is_equal(fv,Y_var[k]):
                    fixed = True
                    break
            if not fixed:
                unfixed_names.append(k)
                algebraics_l.append(Y_var[k])
                if init_guess_y:
                    y_guess_l.append(init_guess_y[k])
            else:
                fixed_names.append(k)

        for eq in self.model.alg_exprs:
            algebraic_eq_l.append(eq)

        states = ca.vertcat(*states_l)
        algebraics = ca.vertcat(*algebraics_l)
        ode = ca.vertcat(*ode_l)
        alg_eq = ca.vertcat(*algebraic_eq_l)
        x_0 = ca.vertcat(*init_conditions_l)
        y_0 = ca.vertcat(*y_guess_l)

        n_unfixed = len(algebraics_l)
        
        step = (self.model.end_time - self.model.start_time)/self.nfe

        results = ResultsObject()

        c_results =  []
        dc_results = []

        x_results =  []
        dx_results = []

        y_results = []
        
        xk = x_0

        if len(y_guess_l):
            yk = y_0
        
        times = sorted(self._times)

        dummy_y = np.ones(self._n_algebraics)
        for i,t in enumerate(times):
            sub_odes = ode
            sub_algs = alg_eq
            for s,var in enumerate(self._fixed_variables):
                value = self._fixed_trajectories[s][t]
                sub_odes = ca.substitute(sub_odes,var,float(value))
                sub_algs = ca.substitute(sub_algs,var,float(value))
            fun_ode = ca.Function("odeFunc",
                                  [self.model.t,states,algebraics],
                                  [sub_odes,sub_algs])
            if i==0:
                step = 1.0e-12
                if len(y_guess_l):
                    arg = {"x0":xk, "z0":yk}
                else:
                    arg = {"x0":xk}
            else:
                step = t - times[i-1]
                arg = {"x0":xk, "z0":yk}
                
            opts = {'tf':step,'print_stats':tee,'verbose':False}
            system = {'t':self.model.t ,'x':states, 'z':algebraics, 'ode':sub_odes, 'alg':sub_algs}
            I = integrator("I",solver, system, opts)
            
            res = I(**arg)
            xk = res['xf']
            yk = res['zf']
                
            # check for nan
            for j in xrange(xk.numel()):
                if np.isnan(float(xk[j])):
                    raise RuntimeError('The iterator returned nan. exiting the program')
                    
            res_f = fun_ode(t,xk,yk)
            odek = res_f[0]
            for j,k in enumerate(self._mixture_components):
                c_results.append(xk[j])
                dc_results.append(odek[j])

            for w,k in enumerate(self._complementary_states):
                j = w+self._n_components
                x_results.append(xk[j])
                dx_results.append(odek[j])

            for j,k in enumerate(unfixed_names):
                y_results.append(yk[j])

            for j,k in enumerate(fixed_names):
                y_results.append(0.0)
                
        c_array = np.array(c_results).reshape((self._n_times,self._n_components))
        results.Z = pd.DataFrame(data=c_array,columns=self._mixture_components,index=times)
                    
        dc_array = np.array(dc_results).reshape((self._n_times,self._n_components))
        results.dZdt = pd.DataFrame(data=dc_array,columns=self._mixture_components,index=times)

        x_array = np.array(x_results).reshape((self._n_times,self._n_complementary_states))
        results.X = pd.DataFrame(data=x_array,columns=self._complementary_states,index=times)

        dx_array = np.array(dx_results).reshape((self._n_times,self._n_complementary_states))
        results.dXdt = pd.DataFrame(data=dx_array,columns=self._complementary_states,index=times)

        columns = unfixed_names + fixed_names
        y_array = np.array(y_results).reshape((self._n_times,self._n_algebraics))
        results.Y = pd.DataFrame(data=y_array,columns=columns,index=times)

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
        
        
        
