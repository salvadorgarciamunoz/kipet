# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from pyomo.environ import *
from pyomo.dae import *
from kipet.library.ParameterEstimator import *
from pyomo import *
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import numpy as np
import scipy
import six
import copy
import re
import os

__author__ = 'Michael Short'  #: November 2018

class EstimabilityAnalyzer(ParameterEstimator):
    """This class is for estimability analysis. For now it will be used to select the parameter set that
    is suitable for estimation based on a mean squared error (MSE) approach first described by Wu, McLean,
    Harris, and McAuley (2011). This, in time, will be expanded to be able to do estimability analysis 
    for spectral data problems as well. The class will contain a number of functions that will perform the 
    estimability analysis. 

    Parameters
    ----------
    model : TemplateBuilder
        The full model TemplateBuilder problem needs to be fed into the Estimability Analyzer as this is 
        needed in order to build the sensitivities for ranking parameters as well as for constructing the 
        simplified models
    """

    def __init__(self, model):
        super(EstimabilityAnalyzer, self).__init__(model)
        self.param_ranks = dict()
        
    def run_sim(self, solver, **kdws):
        raise NotImplementedError("EstimabilityAnalyzer object does not have run_sim method. Call run_analyzer")

    def run_opt(self, solver, **kdws):
        raise NotImplementedError("EstimabilityAnalyzer object does not have run_opt method. Call run_analyzer")

    def get_sensitivities_for_params(self, **kwds):
        """ Obtains the sensitivities (dsdp) using k_aug. This function only works for
        concentration-only problems and obtains the sensitivities based on the initial parameter
        values and how they affect the output (Z).
        
        Args:        
            sigmasq (dict): map of component name to noise variance. The
            map also contains the device noise variance
            
            tee (bool,optional): flag to tell the optimizer whether to stream output
            to the terminal or not
            
        Returns:
            dsdp (numpy matrix):  sensitivity matrix with columns being parameters and rows the Z vars
            idx_to_params (dict): dictionary that maps the columns to the parameters
        """               
        if not self.model.alltime.get_discretization_info():
            raise RuntimeError('apply discretization first before running the estimability')
            
        sigma_sq = kwds.pop('sigmasq', dict())
        tee = kwds.pop('tee', False)
        species_list = kwds.pop('subset_components', None)

        list_components = []
        if species_list is None:
            list_components = [k for k in self._mixture_components]
        else:
            for k in species_list:
                if k in self._mixture_components:
                    list_components.append(k)
                else:
                    warnings.warn("Ignored {} since is not a mixture component of the model".format(k))

        if not self._concentration_given:
            raise NotImplementedError("In order to use the estimability analysis from concentration data requires concentration data model.C[ti,cj]")

        if self._huplc_given:
            raise NotImplementedError("Estimability analysis for additional huplc data is not implemented yet.")

        all_sigma_specified = True

        keys = sigma_sq.keys()
        for k in list_components:
            if k not in keys:
                all_sigma_specified = False
                sigma_sq[k] = max(sigma_sq.values())
                
        if not all_sigma_specified:
            raise RuntimeError(
                'All variances must be specified to determine sensitivities.\n Please pass variance dictionary to run_opt')
        
        m = self.model

        # estimation
        def rule_objective(m):
            obj = 0
            for t in m.allmeas_times:
                # if t in m.meas_times:
                obj += sum((m.C[t, k] - m.Z[t, k]) ** 2 / sigma_sq[k] for k in list_components)
            return obj
            
        m.objective = Objective(rule=rule_objective)

        #set dummy variables for k_aug to do sensitivities
        paramcount = 0
        paramlist=list()
        varcount = 0
        varlist = list()
        for k,v in six.iteritems(m.P):
            if v.is_fixed():
                paramcount +=1
                paramlist.append(k)
            else:
                varcount += 1
                varlist.append(k)
                
        if paramcount >= 1:
            m.dpset = Set(initialize = paramlist)
            m.dummyP= Var(m.dpset)
            for i in paramlist:
                #print("dummyP", i)
                m.dummyP[i] = m.P[i].value
                
        if varcount >= 1:
            m.dvset = Set(initialize = varlist)
            m.dummyV= Param(m.dvset, mutable =True)
            for i in varlist:
                #print("dummyV", i)
                #print(m.P[i].value)
                m.dummyV[i] = m.P[i].value
        
        #set dummy constraints   
        def dummy_constraints(m,p):
            if p in varlist:
                return 0 == m.dummyV[p] - m.P[p] 
            if p in paramlist:
                return 0 == m.dummyP[p] - m.P[p] 
           
        m.dummyC = Constraint(m.parameter_names, rule=dummy_constraints)     
        
        #set up suffixes for Ipopt that are required for k_aug
        m.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
        m.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
        m.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
        m.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
        m.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)

        #: K_AUG SUFFIXES  
        m.dcdp = Suffix(direction=Suffix.EXPORT)  #: the dummy constraints
        m.var_order = Suffix(direction=Suffix.EXPORT)  #: Important variables (primal)
        
        # set which are the variables and which are the parameters for k_aug       
        count_vars = 1
        #print("count_vars:",count_vars)
        if not self._spectra_given:
            pass
        else:
            for t in self._allmeas_times:
                for c in self._sublist_components:
                    m.C[t, c].set_suffix_value(m.var_order,count_vars)
                    count_vars += 1
        
        if not self._spectra_given:
            pass
        else:
            for l in self._meas_lambdas:
                for c in self._sublist_components:
                    m.S[l, c].set_suffix_value(m.var_order,count_vars)
                    count_vars += 1
                        
        if self._concentration_given:
            for t in self._allmeas_times:
                for c in self._sublist_components:
                    m.Z[t, c].set_suffix_value(m.var_order,count_vars)
                    count_vars += 1

        if self._huplc_given:
            raise RuntimeError('Estimability Analysis for additional huplc data is not included as a feature yet!')
                    
        count_dcdp = 1

        idx_to_param = dict()
        for p in m.parameter_names:            
            m.dummyC[p].set_suffix_value(m.dcdp,count_dcdp)
            idx_to_param[count_dcdp]=p
            count_dcdp+=1
          
        #: Clear this file
        with open('ipopt.opt', 'w') as f:
            f.close()
                
        #first solve with Ipopt
        ip = SolverFactory('ipopt')
        solver_results = ip.solve(m, tee=False,
                                  report_timing=False)

        m.ipopt_zL_in.update(m.ipopt_zL_out)
        m.ipopt_zU_in.update(m.ipopt_zU_out) 
        
        k_aug = SolverFactory('k_aug')
        k_aug.options['dsdp_mode'] = ""  #: sensitivity mode!
        #solve with k_aug in sensitivity mode
        k_aug.solve(m, tee=True)
        print("Done solving sensitivities")

        dsdp = np.loadtxt('dxdp_.dat')
        
        if os.path.exists('dxdp_.dat'):
            os.remove('dxdp_.dat')
        print(idx_to_param)
        
        return dsdp , idx_to_param

    def rank_params_yao(self, param_scaling = None, meas_scaling = None, sigmas = None):
        """This function ranks parameters in the method described in Yao (2003) by obtaining the 
        sensitivities related to the parameters in the model through solving the original NLP model 
        for concentrations, getting the sensitivities relating to each paramater, and then using 
        them to predict the next sensitivity. User must provide scaling factors as defined in the 
        paper. These are in the form of dictionaries, relating the confidences to the initial
        guesses for the parameters as well as for the confidence in the measurements.

        Args:
        ----------
        param_scaling (dictionary): dictionary including each parameter and their relative uncertainty.
        e.g. a value of 0.5 means that the value for the real parameter is within 50% of the guessed value
    
        meas_scaling (scalar): scalar value showing the certainty of the measurement, obtained from the device 
        manufacturer or general knowledge of process
        
        sigmasq (dict): map of component name to noise variance. The map also contains the device noise variance.
        
        returns:
            list with order of parameters
        """
        
        if param_scaling == None:
            param_scaling ={}
            print("WARNING: No scaling provided by user, so uncertainties based on the bounds provided by the user is assumed.")
            # uncertainties calculated based on bounds given
            for p in self.model.P:
                lb = self.model.P[p].lb
                ub = self.model.P[p].ub
                init = (ub-lb)/2
                param_scaling[p] = init/(ub-lb)
                print("automated param_scaling", param_scaling)
        elif param_scaling != None:
            if type(param_scaling) is not dict:
                raise RuntimeError('The param_scaling must be type dict')
        
        if meas_scaling == None:
            meas_scaling = 0.001
            print("WARNING: No scaling for measurments provided by user, so uncertainties based on measurements will be set to 0.01")
        elif meas_scaling != None:
            if isinstance(meas_scaling, int) or isinstance(meas_scaling, float):
                print("meas_scaling", meas_scaling)
            else:
                raise RuntimeError('The meas_scaling must be type int')
         
        if sigmas == None:
            sigmas ={}
            print("WARNING: No variances provided by user, so variances are assumed to be 1.")
            # sigmas need to be specified
            for p in self.model.P:
                sigmas[p] = 1
                print("automated sigmas", sigmas)
                
        elif sigmas != None:
            if type(param_scaling) is not dict:
                raise RuntimeError('The param_scaling must be type dict')
            
            else:
                keys = sigmas.keys()
                list_components = [k for k in self._mixture_components]
                all_sigma_specified = True
                for k in list_components:
                    if k not in keys:
                        all_sigma_specified = False
                        sigmas[k] = max(sigmas.values())
                
                if not all_sigma_specified:
                    raise RuntimeError(
                            'All variances must be specified to determine sensitivities.\n Please pass variance dictionary to rank_params_yao')        
        # k_aug is used to get the sensitivities. The full model is solved with dummy
        # parameters and variables at the initial values for the parameters
        self.cloned_before_k_aug = self.model.clone()
        dsdp, idx_to_param = self.get_sensitivities_for_params(tee=True, sigmasq=sigmas)
        #print("idx_to_param",idx_to_param )
        nvars = np.size(dsdp,0)
        #print("nvars,", nvars)
        nparams = 0
        for v in six.itervalues(self.model.P):
            if v.is_fixed():
                print(v, end='\t')
                print("is fixed")
                continue
            nparams += 1

        dsdp_scaled = np.zeros_like(dsdp)

        # scale the sensitivities
        i=0
        for k, p in self.model.P.items():
            if p.is_fixed():
                continue            
            for row in range(len(dsdp)):                    
                dsdp_scaled[row][i] = dsdp[row][i]*param_scaling[k]/meas_scaling
            i += 1
        #print("idx_to_param",idx_to_param )
        #print("dsdp_scaled:", dsdp_scaled)
        # euclidean norm for each column of Hessian relating parameters to outputs
        eucnorm = dict()
        eucnorm_scaled = dict()
        count=0
        for i in range(nparams):
            total = 0
            totalscaled=0
            for row in range(len(dsdp)):
                total += dsdp[row][count]**2 
                totalscaled += dsdp_scaled[row][count]**2
            float(total)
            total = np.asscalar(total)            
            sqr = (total)**(0.5)
            eucnorm[count]=sqr
            
            totals = np.asscalar(totalscaled)
            sqrs = (totals)**(0.5)
            eucnorm_scaled[count]=sqrs
            count+=1
        #print("eucnormscaled",eucnorm_scaled)
        
        # sort the norms and link them to the relevant parameters
        sorted_euc = sorted(eucnorm_scaled.values(), reverse=True)

        #print("sorted_euc,", sorted_euc)
        count=0
        ordered_params = dict()
        for p in idx_to_param:
            for t in idx_to_param:
                if sorted_euc[p-1]==eucnorm_scaled[t-1]:
                    ordered_params[count] = t-1
            count +=1
        #print("ordered_params", ordered_params)
        # set the first ranked parameter as the one with highest norm
        iter_count=0
        self.param_ranks[1] = idx_to_param[ordered_params[0]+1]
            
        #The ranking strategy of Yao, where the X and Z matrices are formed
        next_est = dict()
        X= None
        kcol = None

        for i in range(nparams-1):

            if i==0:
                X = np.zeros((nvars,1))
            else:
                X = np.append(X,np.zeros([len(X),1]),1)
            #print(X)
            # Form the appropriate matrix
            for x in range(i+1):
                #print(self.param_ranks)
                paramhere = self.param_ranks[(x+1)]
                #print(paramhere)

                for key, value in six.iteritems(self.param_ranks):
                    for idx, val in six.iteritems(idx_to_param):
                        if value ==paramhere:
                            if value == val:
                                #print(key, val, idx)
                                which_col = (idx-1) 
                                #print(which_col)
                kcol = dsdp_scaled[:,which_col].T
                recol= kcol.reshape((nvars,1))
                #print("x",x)
                #if x >= 1:
                #    X = np.append(X,np.zeros([len(X),1]),1)
                #    print("why?")
                #    print("X_before 2 loop",X)
                for n in range(nvars):
                    X[n][x] = recol[n][0]
                #print(x)
                #print("X",X)
                
            #print("X_afterloop",X)
            # Use Ordinary Least Squares to use X to predict Z
            # try is here to catch any error resulting from a singular matrix
            # perhaps not the most elegant way of checking for this
            try:
                A = X.T.dot(X)
                B= np.linalg.inv(A)
                C = B.dot(X.T)
                D=X.dot(C)
                Z = dsdp.T
                Zbar=D.dot(Z.T)
                #Get residuals of prediction
                Res = Z.T - Zbar
            except:
                print("There was an error during the OLS prediction. Most likely caused by a singular matrix. Unable to continue the procedure")
                break
            
            # Calculate the magnitude of residuals
            magres = dict()
            counter=0
            for i in range(nparams):
                total = 0
                for row in range(len(Res)):
                    total += Res[row][counter]**2
                float(total)
                total = np.asscalar(total)
                sqr = (total)**(0.5)
                magres[counter]=sqr
                counter +=1

            # Sort the residuals and ensure the params are correctly assigned
            sorted_magres = sorted(magres.values(), reverse=True)
            count2=0
            #next_est = dict()
            for p in idx_to_param:
                for t in idx_to_param:
                    if sorted_magres[p-1]==magres[t-1]:
                        #print('p,t', p,t,count2)
                        next_est[count2] = t
                        #print(next_est[count2])
                count2 += 1
            #print(sorted_magres)
            #print("next_est", next_est)
            # Add next most estimable param to the ranking list  
            self.param_ranks[(iter_count+2)]=idx_to_param[next_est[0]]
            iter_count += 1
            #print("parameter ranks!", self.param_ranks)
            
            #print("======================PARAMETER RANKED======================")
            if len(self.param_ranks) == nparams - 1:
                print("All parameters have been ranked")
                break
        
        #adding the unranked parameters to the list
        #NOTE: if param appears here then it was not evaluated (i.e. it was the least estimable)
        count = 0
        self.unranked_params = {}
        for v,p in six.iteritems(self.model.P):
            if p.is_fixed():
                print(v, end='\t')
                print("is fixed")
                continue            
            if v in self.param_ranks.values():
                continue
            else:
                self.unranked_params[count]=v
                count += 1

        print("The parameters are ranked in the following order from most estimable to least estimable:")
        count = 0
        for i in self.param_ranks:
            print("Number ", i, "is ", self.param_ranks[i])
            count+=1
        
        print("The least estimable parameters are as follows: ")
        if len(self.unranked_params) == 0:
            print("All parameters ranked")
            
        for i in self.unranked_params:
            count+=1
            print("unranked ", (count), "is ", self.unranked_params[i])
        
        #preparing final list to return to user
        self.ordered_params = list()
        count = 0
        for i in self.param_ranks:
            self.ordered_params.append(self.param_ranks[i])
            count += 1
        for i in self.unranked_params:
            self.ordered_params.append(self.unranked_params[i])
            count += 1
        
        return self.ordered_params

    def run_analyzer(self, method = None, parameter_rankings = None, meas_scaling = None, variances = None):
        """This function performs the estimability analysis. The user selects the method to be used. 
        The default will be selected based on the type of data selected. For now, only the method of 
        Wu, McLean, Harris, and McAuley (2011) using the means squared error is used. Other estimability 
        analysis tools will be added in time. The parameter rankings need to be included as well and 
        this can be done using various methods, however for now, only the Yao (2003) method is used.

        Args:
        ----------
        method: string
            The estimability method to be used. Default is Wu, et al (2011) for concentrations. Others 
            to be added
    
        parameter_rankings: list
            A list containing the parameter rankings in order from most estimable to least estimable. 
            Can be obtained using one of Kipet's parameter ranking functions.
            
        meas_scaling: scalar 
            value showing the certainty of the measurement obtained from the device manufacturer or
             general knowledge of process. Same as used in the parameter ranking algorithm.
        
        variances: dict
            variances are required, as needed by the parameter estimator.
        
        returns: list
            list of parameters that should remain in the parameter estimation, while all other 
            parameters should be fixed.
        """
        if method == None:
            method = "Wu"
            print("The method to be used is that of Wu, et al. 2011")
        elif method != "Wu":
            print("The only supported method for estimability analysis is that of Wu, et al., 2011, currently")
        else:
            method = "Wu"
            
        if parameter_rankings == None:
            raise RuntimeError('The parameter rankings need to be provided in order to run the estimability analysis chosen')
            
        elif parameter_rankings != None:
            if type(parameter_rankings) is not list:
                raise RuntimeError('The parameter_rankings must be type dict')   
                
        for v,k in six.iteritems(self.model.P): 
            if v in parameter_rankings:
                continue
            else:
                print("Warning, %s is not included in the parameter rankings algorithm" % v)
                
        for v in parameter_rankings:
            if v not in self.model.P:
                raise RuntimeError("parameter %s is not in the model! Either remove the parameter from the list or add it to the model" % v)
        
        if meas_scaling == None:
            meas_scaling = 0.001
            print("WARNING: No scaling for measurments provided by user, so uncertainties based on measurements will be set to 0.01")
        elif meas_scaling != None:
            if isinstance(meas_scaling, int) or isinstance(meas_scaling, float):
                pass
            else:
                raise RuntimeError('The meas_scaling must be type int')
                
        if variances == None:
            variances ={}
            print("WARNING: No variances provided by user, so variances are assumed to be 1.")
            # sigmas need to be specified
            for p in self.model.P:
                variances[p] = 1
                print("automated sigmas", variances)
            variances["device"] = 1
        elif variances != None:
            if type(variances) is not dict:
                raise RuntimeError('The sigmas must be type dict')
        
        if method == "Wu":
            estimable_params = self.wu_estimability(parameter_rankings, meas_scaling, variances)
            return estimable_params
        else:
            raise RuntimeError("the estimability method must be 'Wu' as this is the only supported method as of now")

    def wu_estimability(self, parameter_rankings = None, meas_scaling = None, sigmas = None):
        """This function performs the estimability analysis of Wu, McLean, Harris, and McAuley (2011) 
        using the means squared error. 

        Args:
        ----------
        parameter_rankings: list
            A list containing the parameter rankings in order from most estimable to least estimable. 
            Can be obtained using one of Kipet's parameter ranking functions.
            
        meas_scaling: int
            measurement scaling as used to scale the sensitivity matrix during param ranking
        
        sigmas: dict
            dictionary containing all the variances as required by the parameter estimator
        
        Returns:
        -----------
            list of parameters that should remain in the parameter estimation, while all other parameters should be fixed.
        """
        
        J = dict()
        params_estimated = list()
        cloned_full_model = dict()
        cloned_pestim = dict()
        results = dict()
        # For now, instead of using Levenberg-Marquardt least squares, we will use Kipet to perform the estimation
        # of every model. Here we generate each of the simplified models.
        # first we clone the main model so that we work with the full model at every iteration
        count = 1
        for p in parameter_rankings:
            cloned_full_model[count] = self.cloned_before_k_aug.clone()
            count += 1
        count = 1
        # Then we go create each simplified model, fixing remaining variables
        for p in parameter_rankings:
            params_estimated.append(p)            
            #print("performing parameter estimation for: ", params_estimated)
            for v,k in six.iteritems(cloned_full_model[count].P):
                if v in params_estimated:
                    continue
                else:
                    #print("fixing the parameters for:",v,k)
                    #fix parameters not in simplified model
                    ub = value(cloned_full_model[count].P[v])
                    lb = ub
                    cloned_full_model[count].P[v].setlb(lb)
                    cloned_full_model[count].P[v].setub(ub)
            # We then solve the Parameter estimaion problem for the SM
            options = dict()            
            cloned_pestim[count] = ParameterEstimator(cloned_full_model[count])
            results[count] = cloned_pestim[count].run_opt('ipopt',
                                        tee=False,
                                        solver_opts = options,
                                        variances=sigmas
                                        )
            #for v,k in six.iteritems(results[count].P):                
                #print(v,k)
            # Then compute the scaled residuals to obtain the Jk in the Wu et al paper   
            J [count] = self._compute_scaled_residuals(results[count], meas_scaling)
            count += 1            
        #print(J)
        count = count - 1
        # Since the estimability procedure suggested by Wu will always skip the last
        # parameter, we should check whether all parameters can be estimated
        # For now this is done by checking that the final parameter does not provide a massive decrease
        # the residuals
        low_MSE = J[1]
        #print("J",J)
        #print("low_MSE", low_MSE)
        listMSE = list()
        for k in J:
            listMSE.append(J[k]) 
            if J[k] <= low_MSE:
                low_MSE = J[k]
            else:
                continue
        #print(count)
        #print(J[count])
        #print(low_MSE)
        if J[count] == low_MSE:
            print("Lowest MSE is given by the lowest ranked parameter, therefore the full model should suffice")
            listMSE.sort()
            print("list of ordered mean squared errors of each :")
            print(listMSE)
            if listMSE[0]*10 <= listMSE[1]:
                print("all parameters are estimable! No need to reduce the model")
                return parameter_rankings

        # Now we move the the final steps of the algorithm where we compute critical ratio
        # and the corrected critical ratio
        # first we need the total number of responses
        N = 0
        for c in self._sublist_components:
            for t in self._allmeas_times:
                N += 1 
                
        crit_rat = dict()
        cor_crit_rat = dict()
        for k in J:
            if k == count:
                break
            crit_rat[k] = (J[k] - J[count])/(count - k)
            crit_rat_Kub = max(crit_rat[k]-1,crit_rat[k]*(2/(count - k + 2)))
            cor_crit_rat[k] = (count - k)/N * (crit_rat_Kub - 1)
        
        #Finally we select the value of k with the lowest corrected critical value
        params_to_select = min(cor_crit_rat, key = lambda x: cor_crit_rat.get(x) )
        print("The number of estimable parameters is:", params_to_select)
        print("optimization should be run wih the following parameters as variables and all others fixed")
        estimable_params = list()
        count=1
        for p in parameter_rankings:
            print(p)
            estimable_params.append(p)
            if count >= params_to_select:
                break
            count += 1
        return estimable_params
    
    def _compute_scaled_residuals(self, model, meas_scaling = None):
        """
        Computes the square of residuals between the optimal solution (Z) and the concentration data (C)
        
        Args:
            model (pyomo results object): solved pyomo model results object
            meas_scaling (dict): parameter scaling, defined in Wu, needs to be the same as used to rank 
                    parameters (scale sensitivity matrix)

        returns:
            value of sum of squared scaled residuals
        This method is not intended to be used by users directly
        """        
        nt = self._n_allmeas_times
        nc = self._n_actual
        self.residuals = dict()
        count_c = 0
        for c in self._sublist_components:
            count_t = 0
            for t in self._allmeas_times:
                a = model.C[c][t]
                b = model.Z[c][t]
                r = ((a - b) ** 2)
                self.residuals[t, c] = r
                count_t += 1
            count_c += 1
        E = 0           
        for c in self._sublist_components:
            for t in self._allmeas_times:
                E += self.residuals[t, c] / (meas_scaling ** 2)

        return E
    
    def obtain_estimable_params(self, epsilon1 = None, epsilon2 = None, eta = None, **kwds):
        '''
        Computes which parameters should be estimable based off of the method from Chen which
        uses the reduced Hessian to obtain the estimability.
        
        Args:
            epsilon1 (float): the "tentative" threshold value for eigenvalues for which we
                    expect corresponding parameters to be estimable
            epsilon2 (float): the "defensive" threshold where if the eigenvalue is less than
                    this value then it is likely to be a dependent parameter (inestimable)
            eta: the tolerance in relation to r_p (ratio of standard deviations)
        
        Returns:
            list of estimable parameters
        '''
        solver_opts = kwds.pop('solver_opts', dict())
        sigmas = kwds.pop('sigmas', dict())
        tee = kwds.pop('tee', False)
        seed = kwds.pop('seed', None)

        if not self.model.alltime.get_discretization_info():
            raise RuntimeError('apply discretization first before runing simulation')
        
        #NEEDS ERROR MESSAGES AND WARNINGS REGARDING THE INPUTS
        # First we run the problem with fixed parameters in order to obtain the reduced hessian
        list_components = [k for k in self._mixture_components]

        # Run this problem using a clone so as not to store the results, bounds, objective etc.
        init_hess = self.model.clone()        
        def rule_objective(m):
            obj = 0
            for t in m.allmeas_times:
                # if t in m.meas_times:
                obj += sum((m.C[t, k] - m.Z[t, k]) ** 2 / sigmas[k] for k in list_components)
            return obj
            
        init_hess.objective = Objective(rule=rule_objective)
        
        #Here we just give the parameters some very small bounds to run the parameter estimation
        for v,k in six.iteritems(init_hess.P):
            #self.model.P[v].fix()
            ub = value(init_hess.P[v])
            lb = ub
            lb = lb - 1e-12            
            init_hess.P[v].setlb(lb)
            init_hess.P[v].setub(ub)
            #print(v,k)
            #print(init_hess.P[v]._lb)
            #print(init_hess.P[v]._ub)
        optimizer = SolverFactory('ipopt_sens')
        if not 'compute_red_hessian' in solver_opts.keys():
            solver_opts['compute_red_hessian'] = 'yes'
        for key, val in solver_opts.items():
            optimizer.options[key] = val
        
        init_hess.red_hessian = Suffix(direction=Suffix.IMPORT_EXPORT)
        count_vars = 1
        for v in six.itervalues(init_hess.P):
            self._idx_to_variable[count_vars] = v
            init_hess.red_hessian[v] = count_vars
            #print(v,count_vars)
            count_vars += 1
            
        self._tmpfile = "ipopt_hess"
        solver_results = optimizer.solve(init_hess, tee=False,
                                         logfile=self._tmpfile,
                                         report_timing=True)

        #init_hess.red_hessian.pprint
        print("Done solving building reduce hessian")
        output_string = ''
        with open(self._tmpfile, 'r') as f:
            output_string = f.read()
        if os.path.exists(self._tmpfile):
            os.remove(self._tmpfile)
        # output_string = f.getvalue()
        ipopt_output, hessian_output = split_sipopt_string(output_string)
        #print(hessian_output)
        #print("build strings")

        #print(ipopt_output)

        n_vars = len(self._idx_to_variable)
        # print('n_vars', n_vars)
        hessian = read_reduce_hessian2(hessian_output, n_vars)
        
        print(hessian.size, "hessian size")
        print("The reduced Hessian")
        print(hessian)
        #v, w, ut = np.linalg.svd(hessian)
        w1,u = np.linalg.eig(hessian)
        print("U.T")
        print(u.T)
        print("eigenvalues:")
        print(w1)
        #print("ranking")
        temp = np.argsort(np.argsort(w1))
        #UTarranged = np.zeros_like(ut)
        UT2arranged = np.zeros_like(u.T)
        #temp = np.flip(temp)
        #print(temp)
        count = 0
        for a in range(len(w1),0,-1):
            #print("a",a-1)
            cor= np.where(temp == a-1)
            #print(cor)
            #print(ut[cor,:] )
            UT2arranged[count,:] = u.T[cor,:] 
            #print(u.T[cor,:] )
            count +=1
        #print("SORTED U")
        #print(UT2arranged)
        #print(temp)
        # print(w1)
        #print("SORTED EIG")
        earr = np.sort(w1)
        #print("eaar:", earr)
        eig_arranged = earr[::-1]
        #for a in range(len(w1)):
        #    b = temp[a]
        #    eig_arranged.append(w1[b])
        #print(eig_arranged)    
        #print(self._idx_to_variable)
        
        x, sig = gaussian_elim(UT2arranged, eig_arranged)
        #S, sig  = gaussian_elim(u.T, w)
        print(x)
        print(sig)
            
        for i in temp:
            #print(i)
            a = temp[i]
            b = w1[i]
            #print(b, self._idx_to_variable[i+1])
            #print(b, self._idx_to_variable[i+1].name)
        #print("Results from Gaussian elim:")
        #print(S)
        
        #print(w1)
        for i in range(len(w1)):
            #print(i)
            #a = temp[i]
            b = w1[i]
            #print(b, self._idx_to_variable[i+1])
        # stack containing inestimable parameters
        stack1 = list()
        # stack containing estimable parameters
        stack2 = list()
        for i in range(len(w1)):
            if w1[i] <= epsilon1:
                stack1.append(w1[i])
            else:
                stack2.append(w1[i])
        print("eigenvalues of inestimable params:", stack1)
        print("eigenvalues of estimable params:", stack2)
        ineststack = list()
        eststack = list()
        for i in temp:
            #print(i)
            for j in stack1:
                #print(j)
                if j == w1[i]:
                    #print("jitty", self._idx_to_variable[i+1].name)
                    ineststack.append(self._idx_to_variable[i+1].name)
            for k in stack2:
                #print(k)
                if k ==w1[i]:
                    #print(type(self._idx_to_variable[i+1].name))
                    eststack.append(self._idx_to_variable[i+1].name)
            #print(i)
            a = temp[i]
            b = w1[i]
            #print(b, self._idx_to_variable[i+1])
            
        print("estimable params:", eststack)
        print("inestimable params:", ineststack)
        
        for v,k in six.iteritems(self.model.P):
            print(v,k.value)
        first_solve = self.model.clone()
        
        def rule_objective(m):
            obj = 0
            for t in m.allmeas_times:
                # if t in m.meas_times:
                obj += sum((m.C[t, k] - m.Z[t, k]) ** 2 / sigmas[k] for k in list_components)
            return obj
            
        first_solve.objective = Objective(rule=rule_objective)
        
        for v,k in six.iteritems(self.model.P):
            #print(v,k.value, type(k.value), k, type(k))
            if self.model.P[v].name in ineststack:
                #print("do we get here?", k.value)               
                ub = value(first_solve.P[v])
                lb = ub
                lb = lb - 1e-12            
                first_solve.P[v].setlb(lb)
                first_solve.P[v].setub(ub)
                #print(v,k)
                #print(first_solve.P[v]._lb)
                #print(first_solve.P[v]._ub)
        optimizer = SolverFactory('ipopt')
        solver_results = optimizer.solve(first_solve, tee=True,
                                         logfile=self._tmpfile,
                                         report_timing=True)
        #solver_results = opt.solve(self.model, tee=True, symbolic_solver_labels=True)
        #sim = PyomoSimulator(m)
    
        # defines the discrete points wanted in the concentration profile
    
        #this will allow for the fe_factory to run the element by element march forward along 
        #the elements and also automatically initialize the PyomoSimulator model, allowing
        #for the use of the run_sim() function as before. We only need to provide the inputs 
        #to this function as an argument dictionary
        for v,k in six.iteritems(first_solve.P):
            print(v,k.value)
        #    m.P[v].fix()
        active_bounds = False
        for v,k in six.iteritems(self.model.P):
            #print(v,k.value, type(k.value), k, type(k))
            if self.model.P[v].name in eststack:
                #print("do we get here?", k.value)               
                ub = value(first_solve.P[v])
                lb = ub
                lb = lb - 1e-12            
                if value(first_solve.P[v]) == first_solve.P[v]._lb:
                    print("Active bounds on:", self.model.P[v].name)
                    active_bounds = True
                elif value(first_solve.P[v]) == first_solve.P[v]._ub:
                    print("Active bounds on:", self.model.P[v].name)
                    active_bounds = True
                #print(v,k)
                #print(first_solve.P[v]._lb)
                #print(first_solve.P[v]._ub)
        
        #init = sim.run_sim(solver = 'ipopt', tee = True)
        results = ResultsObject()

        # activates objective functions that were deactivated
        if self.model.nobjectives():
            active_objectives_names = []
            objectives_map = self.model.component_map(ctype=Objective)
            for name in active_objectives_names:
                objectives_map[name].activate()

        # retriving solutions to results object
        results.load_from_pyomo_model(first_solve,
                                      to_load=['Z','C', 'dZdt', 'X', 'dXdt', 'Y'])    
        results.Z.plot.line(legend=True)
        plt.xlabel("time (min)")
        plt.ylabel("Concentration ((mol /cm3))")
        plt.title("Concentration  Profile")
        plt.show()
        results.C.plot.line(legend=True)
        plt.xlabel("time (min)")
        plt.ylabel("Concentration ((mol /cm3))")
        plt.title("Concentration  Profile")
        plt.show()
            
        print("Simulation is done")

def read_reduce_hessian(hessian_string, n_vars):
    hessian = np.zeros((n_vars, n_vars))
    for i, line in enumerate(hessian_string.split('\n')):
        if i > 0:  # ignores header
            if line not in ['', ' ', '\t']:
                hess_line = line.split(']=')
                if len(hess_line) == 2:
                    value = float(hess_line[1])
                    column_line = hess_line[0].split(',')
                    col = int(column_line[1])
                    row_line = column_line[0].split('[')
                    row = int(row_line[1])
                    hessian[row, col] = float(value)
                    hessian[col, row] = float(value)
    return hessian  
      
def read_reduce_hessian2(hessian_string, n_vars):
    hessian_string = re.sub('RedHessian unscaled\[', '', hessian_string)
    hessian_string = re.sub('\]=', ',', hessian_string)

    hessian = np.zeros((n_vars, n_vars))
    for i, line in enumerate(hessian_string.split('\n')):
        if i > 0:
            hess_line = line.split(',')
            if len(hess_line) == 3:
                row = int(hess_line[0])
                col = int(hess_line[1])
                hessian[row, col] = float(hess_line[2])
                hessian[col, row] = float(hess_line[2])
    return hessian

def gaussian_elim(UT,sig):
    """
    Function to perform the gaussian elimination required for the ranking of parameters
    """
    #pl, u = scipy.linalg.lu(UT, permute_l=True)
    #print(pl)
    #print(u)
    #print(len(UT))
    cols = len(UT)
    rows = len(UT[0])
    #print(cols, rows)
    
    for row in range(rows):
        #print(row)
        #find the maximum value
        cont = 0
        for v in range(cols):
            #print(v)
            #print("compare",abs(UT[row][v]), abs(cont))
            if abs(UT[row][v]) > abs(cont):
                cont = UT[row][v]
                piv = v
                #print(cont, piv)
            else:
                pass
        #print(piv, cont)
        # Now we have the pivot so perform guassian elim
        for r in range(rows):
            if r == row:
                pass
            else:
                #print(rows)
                #print(r,row)
                #print("UT[r][piv]", UT[r][piv])
                #print("UT[row][piv]",UT[row][piv])
                multiplier = UT[r][piv]/UT[row][piv]
                #the only one in this column since the rest are zero
                #print("multiplier",multiplier)
                UT[r][piv] = multiplier
                for col in range(cols):
                    #print(UT[r][col])
                    #print(UT[row][col])
                    #print("the sub", multiplier*UT[row][col])
                    if UT[row][piv] < 0:
                        UT[r][col] = UT[r][col] + multiplier*UT[row][col]
                    else:
                        UT[r][col] = UT[r][col] - multiplier*UT[row][col]
                    
                    if abs(UT[r][col]) < 1e-25:
                        UT[r][col] = 0
                    #print(r,col,row)
                    #print(UT[r][col])
                #Equation solution column
                #print(sig[r])
                if UT[row][piv] < 0:
                    sig[r] = sig[r] + multiplier*sig[piv]
                else:
                    sig[r] = sig[r] - multiplier*sig[piv]
                #print(sig[r])
        #print(UT)
        #print(sig)
    
    return UT, sig