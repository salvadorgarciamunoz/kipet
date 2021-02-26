# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from pyomo.environ import *
from pyomo.dae import *
from kipet.core_methods.ParameterEstimator import *
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
                obj += sum((m.Cm[t, k] - m.Z[t, k]) ** 2 / sigma_sq[k] for k in list_components)
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
        # print(idx_to_param)
        
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
        # print("idx_to_param",idx_to_param )
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
        # print("ordered_params", ordered_params)
        # set the first ranked parameter as the one with highest norm
        iter_count=0
        self.param_ranks[1] = idx_to_param[ordered_params[0]+1]
        # print("self.param_ranks",self.param_ranks)
        #The ranking strategy of Yao, where the X and Z matrices are formed
        next_est = dict()
        X= None
        kcol = None
        countdoub=0
        for i in range(nparams-1):
            if i==0:
                X = np.zeros((nvars,1))
            else:
                X = np.append(X,np.zeros([len(X),1]),1)
            #print(X)
            # Form the appropriate matrix
            for x in range(i+1):
                if x < nparams-countdoub-2:
                    #print(self.param_ranks)
                    # print(x)
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
                        # print('p,t', p,t,count2)
                        next_est[count2] = t
                        # print(next_est[count2])
                count2 += 1
            # print(sorted_magres)
            # print("next_est", next_est)
            # Add next most estimable param to the ranking list
            # print('idx_to_param[next_est[0]]', idx_to_param[next_est[0]])
            # print('self.param_ranks[1]',self.param_ranks[1][:])
            if idx_to_param[next_est[0]] not in self.param_ranks.values():
                self.param_ranks[(iter_count+2)]=idx_to_param[next_est[0]]
                iter_count += 1
            else:
                countdoub+=1
            # print("parameter ranks!", self.param_ranks)
            # print("nparams", nparams)

            #print("======================PARAMETER RANKED======================")
            if len(self.param_ranks) == nparams-countdoub-1:
                print("Parameters have been ranked")
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
        print(self.param_ranks)
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
            if count>=2:
                if hasattr(results[count-1], 'Y'):
                    cloned_pestim[count].initialize_from_trajectory('Y', results[count - 1].Y)
                    cloned_pestim[count].scale_variables_from_trajectory('Y', results[count - 1].Y)
                if hasattr(results[count-1], 'X'):
                    cloned_pestim[count].initialize_from_trajectory('X', results[count - 1].X)
                    cloned_pestim[count].scale_variables_from_trajectory('X', results[count - 1].X)
                if hasattr(results[count-1], 'C'):
                    cloned_pestim[count].initialize_from_trajectory('C', results[count - 1].C)
                    cloned_pestim[count].scale_variables_from_trajectory('C', results[count - 1].C)
                # if hasattr(results[count-1], 'S'):
                #     cloned_pestim[count].initialize_from_trajectory('S', results[count-1].S)
                #     cloned_pestim[count].scale_variables_from_trajectory('S', results[count-1].S)
                cloned_pestim[count].initialize_from_trajectory('Z', results[count-1].Z)
                cloned_pestim[count].scale_variables_from_trajectory('Z', results[count - 1].Z)
                cloned_pestim[count].initialize_from_trajectory('dZdt', results[count - 1].dZdt)
                cloned_pestim[count].scale_variables_from_trajectory('dZdt', results[count - 1].dZdt)

            results[count] = cloned_pestim[count].run_opt('ipopt',
                                        tee=True,#False,
                                        solver_opts = options,
                                        variances=sigmas, symbolic_solver_labels=True
                                        )

            # print('TC',TerminationCondition.optimal)
            # print('selfterm',self.termination_condition)

            for v,k in six.iteritems(results[count].P):
                print(v,k)
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
                a = model.Cm[c][t]
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