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


