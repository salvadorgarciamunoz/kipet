"""
Common functions between PE and MEE
"""
import numpy as np

from kipet.library.common.read_hessian import *

class PEMixins(object):

    def _set_up_reduced_hessian(self, model_obj, time_set, component_set, var_name, index):
        """Method to declare the reduced hessian suffix variables"""
    
        for t in time_set:
            for c in component_set:
                v = getattr(model_obj, var_name)[t, c]
                self._idx_to_variable[index] = v
                self.model.red_hessian[v] = index
                index += 1
       
        return index
                        
    
    def _order_k_aug_hessian(self, unordered_hessian, var_loc):
        """
        not meant to be used directly by users. Takes in the inverse of the reduced hessian
        outputted by k_aug and uses the rh_name to find the locations of the variables and then
        re-orders the hessian to be in a format where the other functions are able to compute the
        confidence intervals in a way similar to that utilized by sIpopt.
        """
        vlocsize = len(var_loc)
        n_vars = len(self._idx_to_variable)
        hessian = np.zeros((n_vars, n_vars))
        
        for i, vi in enumerate(self._idx_to_variable.values()):
            for j, vj in enumerate(self._idx_to_variable.values()):
                if n_vars == 1:
                    h = unordered_hessian
                    hessian[i, j] = h
                else:
                    h = unordered_hessian[(var_loc[vi]), (var_loc[vj])]
                    hessian[i, j] = h
        print(hessian.size, "hessian size")
        return hessian
    
    def _compute_residuals(self, model_obj, exp_index=None):
        """4
        Computes the square of residuals between the optimal solution (Z) and the concentration data (C)
        Note that this returns a matrix of time points X components and it has not been divided by sigma^2

        This method is not intended to be used by users directly
        """
        nt = self._n_meas_times
        nc = self._n_actual
        residuals = dict()
        
        for index, value in model_obj.C.items():
            
            res_index = list(index)
            if exp_index:
                res_index.insert(0, exp_index)
                
            res_index = tuple(res_index)
            
            residuals[res_index] = (value.value - model_obj.Z[index].value)**2 
             
        return residuals
    
    @staticmethod
    def _get_nparams(model_obj, isSkipFixed=True):
        """Returns the number of unfixed parameters"""
        
        nparams = 0

        for v in model_obj.P.values():
            if v.is_fixed():
                print(str(v) + '\has been skipped for covariance calculations')
                continue
            nparams += 1
            
        if hasattr(model_obj, 'Pinit'):
            for v in model_obj.Pinit.values():
                if isSkipFixed:
                    if v.is_fixed():
                        print(str(v) + '\has been skipped for covariance calculations')
                        continue
                nparams += 1

        return nparams
    
    def _variances_p_calc(self, hessian, variances):
        """Computes the covariance for post calculation anaylsis
        
        """        
        nparams = self._n_params

        H = hessian[-nparams:, :]
 
        B = self._compute_B_matrix(variances)
        Vd = self._compute_Vd_matrix(variances)
        
        print(f'B: {B.shape}')
        print(f'H: {H.shape}')
        print(f'Vd: {Vd.shape}')
        
        R = B.T @ H.T
        A = Vd @ R
        L = H @ B
        V_theta = (A.T @ L.T).T
        variances_p = np.diag(V_theta)
        
        if hasattr(self, '_eigredhess2file') and self._eigredhess2file==True:
            save_eig_red_hess(V_theta)
        
        return variances_p, V_theta

