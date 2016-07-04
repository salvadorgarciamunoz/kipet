from pyomo.environ import *
from pyomo.dae import *
#from kipet.ResultsObject import *
from kipet.sim.PyomoSimulator import *
from pyomo.core.base.expr import Expr_if

import copy

class VarianceEstimator(PyomoSimulator):
    def __init__(self,model):
        super(VarianceEstimator, self).__init__(model)

        self.S_model = ConcreteModel()
        self.S_model.S = Var(self._meas_lambdas,
                             self._mixture_components,
                             bounds=(0.0,None),
                             initialize=1.0)

        self.C_model = ConcreteModel()
        self.C_model = Var(self._meas_times,
                           self._mixture_components,
                           bounds=(0.0,None),
                           initialize=1.0)
        
    def run_sim(self,solver,tee=False,solver_opts={}):
        raise NotImplementedError("VarianceEstimator object does not have run_sim method. Call run_opt")
    
    def _build_initalization_model(self,subset_lambdas=None):

        if subset_lambdas:
            set_A = set(subset_lambdas)
        else:
            set_A = copy.deepcopy(self._meas_lambdas)

        # build model
        dae = self.model
        m = ConcreteModel()
        m.A = Set(initialize=set_A)
        m.add_component('dae', dae)
        
        # build objective
        obj = 0.0
        for t in self._meas_times:
            for l in m.A:
                D_bar = sum(m.dae.Z[t,k]*m.dae.S[l,k] for k in m.dae.mixture_components)
                obj+= (m.dae.D[t,l] - D_bar)**2
        m.objective = Objective(expr=obj)
        
        return m

    def _solve_initalization(self,
                             solver,
                             tee=False,
                             solver_options={},
                             subset_lambdas=None):

        if subset_lambdas:
            set_A = set(subset_lambdas)
        else:
            set_A = self._meas_lambdas

        # build model
        dae = self.model
        m = ConcreteModel()
        m.A = Set(initialize=set_A)
        m.add_component('dae', dae)
        
        # build objective
        obj = 0.0
        for t in self._meas_times:
            for l in m.A:
                D_bar = sum(m.dae.Z[t,k]*m.dae.S[l,k] for k in m.dae.mixture_components)
                obj+= (m.dae.D[t,l] - D_bar)**2
        m.objective = Objective(expr=obj)

        opt = SolverFactory(solver)
        for key, val in solver_opts.iteritems():
            opt.options[key]=val                

        solver_results = opt.solve(self.model,tee=tee)

        for t in self._meas_times:
            for k in self._mixture_components:
                m.dae.C[t,k].value = m.dae.Z[t,k]
        
        m.del_component('dae')

    
    def _build_Z_model(self,C_trajectory):
        dae = self.model
        m = ConcreteModel()
        m.add_component('dae', dae)
        
        obj = 0.0
        reciprocal_ntp = 1.0/len(m.dae.time)
        tao = 1e-6
        gamma = 1e-6
        eta =1e-4
        for k in m.dae.mixture_components:
            x = sum((C_trajectory[k][t]-m.dae.Z[t,k])**2 for t in self._meas_times)
            x*= reciprocal_ntp
            #f_x = 0.5*gamma*(x/(x**2+eta**2)**0.5+(tao-x)/((tao-x)**2+eta**2)**0.5) + x*0.5*((x-tao)/((x-tao)**2+eta**2)**0.5+1)
            #obj+= log(f_x)            
            obj+= x

        m.objective = Objective(expr=obj)
        
        return m

    def _solve_Z(self,
                 solver,
                 C_trajectory=None,
                 tee=False,
                 solver_options={}):
        
        dae = self.model
        m = ConcreteModel()
        m.add_component('dae', dae)
        
        obj = 0.0
        reciprocal_ntp = 1.0/len(m.dae.time)
        if C_trajectory:
            for k in m.dae.mixture_components:
                x = sum((C_trajectory[k][t]-m.dae.Z[t,k])**2 for t in self._meas_times)
                x*= reciprocal_ntp
                obj+= x
        else:
            # asume this value was computed beforehand
            for t in self._meas_times:
                for k in self._mixture_components:
                    m.dae.C[t,k] = True
            
            for k in m.dae.mixture_components:
                x = sum((m.dae.C[t,k]-m.dae.Z[t,k])**2 for t in self._meas_times)
                x*= reciprocal_ntp
                obj+= x

        m.objective = Objective(expr=obj)

        opt = SolverFactory(solver)
        for key, val in solver_opts.iteritems():
            opt.options[key]=val                

        solver_results = opt.solve(self.model,tee=tee)

        if C_trajectory is not None:
            # unfixes all concentrations
            for t in self._meas_times:
                for k in self._mixture_components:
                    m.dae.C[t,k] = False
        
        m.del_component('dae')


    def _build_S_model(self,Z_trajectory):
        dae = self.model
        m = ConcreteModel()
        m.S = Var(self._meas_lambdas,
                  self._mixture_components,
                  bounds=(0.0,None),
                  initialize=1.0)

        obj = 0.0
        for t in self._meas_times:
            for l in self._meas_lambdas:
                D_bar = sum(m.S[l,k]*Z_trajectory[k][t] for k in self._mixture_components)
                obj+=(self.model.D[t,l]-D_bar)**2
            
        m.objective = Objective(expr=obj)
        return m

    def _solve_S(self,
                 Z_trajectory,
                 solver,
                 tee=False,
                 solver_options={}):
        
        obj = 0.0
        for t in self._meas_times:
            for l in self._meas_lambdas:
                D_bar = sum(self.S_model.S[l,k]*Z_trajectory[k][t] for k in self._mixture_components)
                obj+=(self.model.D[t,l]-D_bar)**2
            
        self.S_model.objective = Objective(expr=obj)

        opt = SolverFactory(solver)
        for key, val in solver_opts.iteritems():
            opt.options[key]=val                

        solver_results = opt.solve(self.S_model,tee=tee)
        self.S_model.del_component(objective)
        
    def _build_C_model(self,S_trajectory):
        m = ConcreteModel()
        m.C = Var(self._meas_times,
                  self._mixture_components,
                  bounds=(0.0,None),
                  initialize=1.0)
        
        obj = 0.0
        for t in self._meas_times:
            for l in self._meas_lambdas:
                D_bar = sum(S_trajectory[k][l]*m.C[t,k] for k in self._mixture_components)
                obj+=(self.model.D[t,l]-D_bar)**2
        m.objective = Objective(expr=obj)

        return m

    def _solve_C(self,
                 S_trajectory,
                 solver,
                 tee=False,
                 solver_options={}):

        obj = 0.0
        for t in self._meas_times:
            for l in self._meas_lambdas:
                D_bar = sum(S_trajectory[k][l]*self.C_model.C[t,k] for k in self._mixture_components)
                obj+=(self.model.D[t,l]-D_bar)**2
        
        self.C_model.objective = Objective(expr=obj)

        opt = SolverFactory(solver)
        for key, val in solver_opts.iteritems():
            opt.options[key]=val                

        solver_results = opt.solve(self.C_model,tee=tee)
        self.C_model.del_component(objective)
        


    def _solve_variances(self,S_trajectory,Z_trajectory):
        nl = self._n_meas_lambdas
        nt = self._n_meas_times
        nc = self._n_components
        A = np.ones((nl,nc+1))
        b = np.zeros((nl,1))

        reciprocal_nt = 1.0/nt
        for i,l in enumerate(self._meas_lambdas):
            for j,t in enumerate(self._meas_times):
                D_bar = 0.0
                for w,k in enumerate(self._mixture_components):
                    A[i,w] = S_trajectory[k][l]**2
                    D_bar += S_trajectory[k][l]*Z_trajectory[k][t]
                b[i] += (self.model.D[t,l]-D_bar)**2
            b[i]*=reciprocal_nt

        print A.shape
        results = np.linalg.lstsq(A, b)
        return results
            
        
    
    def run_opt(self,solver,tee=False,solver_opts={},variances={}):
        
        Z_var = self.model.Z 
        dZ_var = self.model.dZdt

        if self._discretized is False:
            raise RuntimeError('apply discretization first before runing simulation')

        # disable objectives of dae
        objectives_map = self.model.component_map(ctype=Objective,active=True)
        for obj in objectives_map.itervalues():
            obj.deactivate()

        opt = SolverFactory(solver)
        for key, val in solver_opts.iteritems():
            opt.options[key]=val
                
        init_model = self._build_initalization_model()

        solver_results = opt.solve(init_model,tee=True)
        init_model.del_component('dae')

        c_results = []
        for t in self._meas_times:
            for k in self._mixture_components:
                c_results.append(Z_var[t,k].value)

        c_array = np.array(c_results).reshape((self._n_meas_times,self._n_components))
        
        C_trajectories = pd.DataFrame(data=c_array,
                                 columns=self._mixture_components,
                                 index=self._meas_times)

        # start solving sequence of problems
        z_model = self._build_Z_model(C_trajectories)
        solver_results = opt.solve(z_model,tee=True)
        z_model.del_component('dae')

        init_c = dict()
        c_results = []
        for t in self._meas_times:
            for k in self._mixture_components:
                c_results.append(Z_var[t,k].value)
                init_c[t,k] = Z_var[t,k].value
        c_array = np.array(c_results).reshape((self._n_meas_times,self._n_components))
        
        Z_trajectory = pd.DataFrame(data=c_array,
                                    columns=self._mixture_components,
                                    index=self._meas_times)
        
        s_model = self._build_S_model(Z_trajectory)
        solver_results = opt.solve(s_model,tee=True)

        
        results =  ResultsObject()
        s_results = []
        for l in self._meas_lambdas:
            for k in self._mixture_components:
                s_results.append(s_model.S[l,k].value)

        s_array = np.array(s_results).reshape((self._n_meas_lambdas,self._n_components))
                
        S_trajectory = pd.DataFrame(data=s_array,
                                    columns=self._mixture_components,
                                    index=self._meas_lambdas)

        c_model = self._build_C_model(S_trajectory)
        solver_results = opt.solve(c_model,tee=True)
        
        c_results = []
        for t in self._times:
            for k in self._mixture_components:
                c_results.append(Z_var[t,k].value)

        c_array = np.array(c_results).reshape((self._n_times,self._n_components))
        
        results.Z = pd.DataFrame(data=c_array,
                                 columns=self._mixture_components,
                                 index=self._times)


        c_noise_results = []
        for t in self._meas_times:
            for k in self._mixture_components:
                c_noise_results.append(c_model.C[t,k].value)
        
        c_noise_array = np.array(c_noise_results).reshape((self._n_meas_times,self._n_components))

        results.C = pd.DataFrame(data=c_noise_array,
                                 columns=self._mixture_components,
                                 index=self._meas_times)

        results.S = S_trajectory

        print self._solve_variances(S_trajectory,Z_trajectory)
        
        param_vals = dict()
        for name in self.model.parameter_names:
            param_vals[name] = self.model.P[name].value

        results.P = param_vals
        
        return results

            
            
