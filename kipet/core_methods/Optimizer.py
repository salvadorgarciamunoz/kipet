"""
Base Optimizer Class
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import (
    lsqr,
    spsolve,
    ) 
from scipy.optimize import least_squares

from pyomo.dae import *
from pyomo.environ import *
from pyomo.environ import (
    Suffix,
    )

from kipet.core_methods.PyomoSimulator import *
from kipet.core_methods.ResultsObject import *
from kipet.mixins.JumpsMixin import JumpsMixin


class Optimizer(JumpsMixin, PyomoSimulator):
    """Base optimizer class.

    Note:
        This class is not intended to be used directly by users

    Attributes:
        model (model): Pyomo model.

    """
    def __init__(self, model):
        """Optimizer constructor.

        Note: 
            Makes a shallow copy to the model. Changes applied to 
            the model within the simulator are applied to the original
            model passed to the simulator

        Args:
            model (Pyomo model)
        """
        super(Optimizer, self).__init__(model)
        
    def run_sim(self, solver, **kdws):
        raise NotImplementedError("Optimizer abstract method. Call child class")       

    def run_opt(self, solver, **kwds):
        raise NotImplementedError("Optimizer abstract method. Call child class")
        
    @staticmethod
    def add_warm_start_suffixes(model, use_k_aug=False):
        """Adds suffixed variables to problem"""
        
        # Ipopt bound multipliers (obtained from solution)
        model.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
        model.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
        # Ipopt bound multipliers (sent to solver)
        model.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
        model.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)
        # Obtain dual solutions from first solve and send to warm start
        model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
        
        if use_k_aug:
            model.dof_v = Suffix(direction=Suffix.EXPORT)
            model.rh_name = Suffix(direction=Suffix.IMPORT)
            
        return None
            
    @staticmethod
    def update_warm_start(model):
        """Updates the suffixed variables for a warmstart"""
        
        model.ipopt_zL_in.update(model.ipopt_zL_out)
        model.ipopt_zU_in.update(model.ipopt_zU_out)
        
        return None
    
    def _solve_S_from_DC(self, C_dataFrame, tee=False, with_bounds=False, max_iter=200):
        """Solves a basic least squares problems with SVD.
        
        Args:
            C_dataFrame (DataFrame) data frame with concentration values
        
        Returns:
            DataFrame with estimated S_values 
        """
        D_data = self.model.D
        if self._n_meas_lambdas:
            # build Dij vector
            D_vector = np.zeros(self._n_meas_times*self._n_meas_lambdas)
            
            row  = []
            col  = []
            data = []    
            for i,t in enumerate(self._meas_times):
                for j,l in enumerate(self._meas_lambdas):
                    for k,c in enumerate(self._mixture_components):
                        row.append(i*self._n_meas_lambdas+j)
                        col.append(j*self._n_components+k)
                        data.append(C_dataFrame[c][t])
                    D_vector[i*self._n_meas_lambdas+j] = D_data[t,l]    
                
                        
            Bd = coo_matrix((data, (row, col)),
                            shape=(self._n_meas_times*self._n_meas_lambdas,
                                   self._n_components*self._n_meas_lambdas)
                            )

            if not with_bounds:
                if self._n_meas_times == self._n_components:
                    s_array = spsolve(Bd, D_vector)
                elif self._n_meas_times>self._n_components:
                    result_ls = lsqr(Bd, D_vector,show=tee)
                    s_array = result_ls[0]
                else:
                    raise RuntimeError('Need n_t_meas >= self._n_components')
            else:
                nl = self._n_meas_lambdas
                nt = self._n_meas_times
                nc = self._n_components
                x0 = np.zeros(nl*nc)+1e-2
                M = Bd.tocsr()
                
                def F(x,M,rhs):
                    return  rhs-M.dot(x)

                def JF(x,M,rhs):
                    return -M

                if tee == True:
                    verbose = 2
                else:
                    verbose = 0
                    
                res_lsq = least_squares(F,x0,JF,
                                        bounds=(0.0,np.inf),
                                        max_nfev=max_iter,
                                        verbose=verbose,args=(M,D_vector))
                s_array = res_lsq.x
                
            s_shaped = s_array.reshape((self._n_meas_lambdas,self._n_components))
        else:
            s_shaped = np.empty((self._n_meas_lambdas,self._n_components))

        return s_shaped
    
    def run_lsq_given_P(self, solver, parameters, **kwds):
        """Gives a raw estimate of S given kinetic parameters.
        
        Args:
            solver (str): name of the nonlinear solver to used
          
            solver_opts (dict, optional): options passed to the nonlinear solver
        
            variances (dict, optional): map of component name to noise variance. The
            map also contains the device noise variance
            
            tee (bool,optional): flag to tell the optimizer whether to stream output
            to the terminal or not
            initialization (bool, optional): flag indicating whether result should be 
            loaded or not to the pyomo model
        
        Returns:
            Results object with loaded results
        """
        solver_opts = kwds.pop('solver_opts', dict())
        variances = kwds.pop('variances',dict())
        tee = kwds.pop('tee',False)
        initialization = kwds.pop('initialization',False)
        wb = kwds.pop('with_bounds',True)
        max_iter = kwds.pop('max_lsq_iter',200)
        
        if not self.model.alltime.get_discretization_info():
            raise RuntimeError('apply discretization first before runing simulation')

        #self.model =copy.deepcopy(self.model)

        base_values = ResultsObject()
        base_values.load_from_pyomo_model(self.model,
                                          to_load=['Z','dZdt','X','dXdt','Y'])
        
        # fixes parameters 
        old_values = {}        
        for k,v in parameters.items():
            if self.model.P[k].fixed ==False:
                old_values[k] = self.model.P[k].value
                self.model.P[k].value = v
                self.model.P[k].fixed = True

        for k,v in self.model.P.items():
            if not v.fixed:
                print('***WARNING parameter {} is not fixed. This method expects all parameters to be fixed.'.format(k))
            
        # deactivates objective functions for simulation                
        objectives_map = self.model.component_map(ctype=Objective,active=True)
        active_objectives_names = []
        for obj in objectives_map.values():
            name = obj.cname()
            active_objectives_names.append(name)
            obj.deactivate()

            
        opt = SolverFactory(solver)
        for key, val in solver_opts.items():
            opt.options[key]=val

        solver_results = opt.solve(self.model,tee=tee)

        #unfixes the parameters that were fixed
        for k,v in old_values.items():
            if not initialization:
                self.model.P[k].value = v 
            self.model.P[k].fixed = False
            self.model.P[k].stale = False
        # activates objective functions that were deactivated
        active_objectives_names = []
        objectives_map = self.model.component_map(ctype=Objective)
        for name in active_objectives_names:
            objectives_map[name].activate()

        # unstale variables that were marked stale
        for var in self.model.component_map(ctype=Var).values():
            if not isinstance(var, DerivativeVar):
                for var_data in var.values():
                    var_data.stale=False
            else:
                for var_data in var.values():
                    var_data.stale=True

        # retriving solutions to results object  
        results = ResultsObject()
        results.load_from_pyomo_model(self.model,
                                      to_load=['Z','dZdt','X','dXdt','Y'])

        c_array = np.zeros((self._n_allmeas_times,self._n_components))
        for i,t in enumerate(self._allmeas_times):
            for j,k in enumerate(self._mixture_components):
                c_array[i,j] = results.Z[k][t]

        results.C = pd.DataFrame(data=c_array,
                                 columns=self._mixture_components,
                                 index=self._allmeas_times)
        
        D_data = self.model.D
        
        if self._n_allmeas_times and self._n_allmeas_times<self._n_components:
            raise RuntimeError('Not enough measurements num_meas>= num_components')

        # solves over determined system
        s_array = self._solve_S_from_DC(results.C,
                                        tee=tee,
                                        with_bounds=wb,
                                        max_iter=max_iter)

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

        if initialization:
            for t in self.model.allmeas_times:
                for k in self.mixture_components:
                    self.model.C[t,k].value = self.model.Z[t,k].value

            for l in self.model.meas_lambdas:
                for k in self.mixture_components:
                    self.model.S[l,k].value =  results.S[k][l]
        else:
            if not base_values.Z.empty:
                self.initialize_from_trajectory('Z',base_values.Z)
                self.initialize_from_trajectory('dZdt',base_values.dZdt)
            if not base_values.X.empty:
                self.initialize_from_trajectory('X',base_values.X)
                self.initialize_from_trajectory('dXdt',base_values.dXdt)
        
        return results