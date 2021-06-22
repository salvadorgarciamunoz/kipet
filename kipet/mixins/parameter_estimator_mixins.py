"""
Common functions between PE and MEE
"""
# Third party imports
import numpy as np
from pyomo.environ import Suffix

# KIPET library imports
from kipet.input_output.read_hessian import save_eig_red_hess


class PEMixins(object):

    """This mixin class takes some of the methods used by both the ParameterEstimator and the MEE classes
    and placed them into a single location

    This is not meant to be used by the user directly.
    """

    def _set_up_reduced_hessian(self, model_obj, time_set, component_set, var_name, index):
        """Method to declare the reduced hessian suffix variables

        :param expression model_obj: The model objective
        :param index time_set: The time index
        :param index component_set: The list of components
        :param str var_name: The model variable name
        :param int index: The current index number

        :return index: Updated index number
        :rtype: int

        """
        for t in time_set:
            for c in component_set:
                v = getattr(model_obj, var_name)[t, c]
                self._idx_to_variable[index] = v
                self.model.red_hessian[v] = index
                index += 1
       
        return index
    
    def _order_k_aug_hessian(self, unordered_hessian, var_loc):
        """ This is not meant to be used directly by users. Takes in the inverse of the reduced hessian
        outputted by k_aug and uses the rh_name to find the locations of the variables and then
        re-orders the hessian to be in a format where the other functions are able to compute the
        confidence intervals in a way similar to that utilized by sIpopt.

        :param np.ndarray unordered_hessian: The raw Hessian
        :param array-like var_loc: The index of the variable location

        :return hessian: The ordered Hessian
        :rtype: numpy.ndarray

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
        
        # n_vars = len(self._idx_to_variable)
        
        # if n_vars == 1:
        #     hessian = np.array(unordered_hessian).reshape(1, 1)
        
        # hessian = np.zeros((n_vars, n_vars))
        
        # for i, vi in enumerate(self._idx_to_variable.values()):
        #     for j, vj in enumerate(self._idx_to_variable.values()):
        #         if j < i:
        #             continue
        #         hessian[i, j] = unordered_hessian[(var_loc[vi]), (var_loc[vj])]
                    
        # hessian = hessian + np.tril(hessian.T, -1)
                    
        # return hessian
    
    def _compute_residuals(self, model_obj, exp_index=None):
        """
        Computes the square of residuals between the optimal solution (Z) and the concentration data (C)
        Note that this returns a matrix of time points X components and it has not been divided by sigma^2

        This method is not intended to be used by users directly

        :param expression model_obj: The model objective
        :param exp_index: None

        :return residuals:
        :rtype: dict

        """
        residuals = dict()
        
        conc_data = ['C', 'Cm']
        
        for model_var in conc_data:     
            if hasattr(model_obj, model_var) and getattr(model_obj, model_var) is not None:     
                for index, value in getattr(model_obj, model_var).items():
                    res_index = list(index)
                    if exp_index:
                        res_index.insert(0, exp_index)                 
                    res_index = tuple(res_index)             
                    residuals[res_index] = (value.value - model_obj.Z[index].value)**2 
             
        return residuals
    
    @staticmethod
    def _get_nparams(model_obj, isSkipFixed=True):
        """Returns the number of unfixed parameters in the objective.

        :param expression model_obj: The model objective
        :param bool isSkipFixed: Not used at the moment

        """
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
                
        if hasattr(model_obj, 'time_step_change'):
            for v in model_obj.time_step_change.values():
                if isSkipFixed:
                    if v.is_fixed():
                        print(str(v) + '\has been skipped for covariance calculations')
                        continue
                nparams += 1

        return nparams
    
    def _variances_p_calc(self, H, variances):
        """Computes the covariance for post calculation anaylsis

        :param np.ndarray hessian: The Hessian matrix
        :param dict variances: The parameter variances

        :return numpy.ndarray variances_p: The diagonal of the reduced Hessian
        :return numpy.ndarray V_theta: The reduced Hessian

        """        
        B = self._compute_B_matrix(variances)
        Vd = self._compute_Vd_matrix(variances)
        
        R = B.T @ H.T
        A = Vd @ R
        L = H @ B
        V_theta = (A.T @ L.T).T
        variances_p = np.diag(V_theta)
        
        if hasattr(self, '_eigredhess2file') and self._eigredhess2file==True:
            save_eig_red_hess(V_theta)
        
        return variances_p, V_theta

    @staticmethod
    def add_warm_start_suffixes(model, use_k_aug=False):
        """Adds suffixed variables to problem

        :param ConcreteModel model: The Pyomo model considered
        :param bool use_k_aug: Indicates if k_aug is being used

        :return: None

        """
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
        """Updates the suffixed variables for a warmstart

        :param ConcreteModel model: The Pyomo model considered

        :return: None

        """
        model.ipopt_zL_in.update(model.ipopt_zL_out)
        model.ipopt_zU_in.update(model.ipopt_zU_out)
        
        return None
