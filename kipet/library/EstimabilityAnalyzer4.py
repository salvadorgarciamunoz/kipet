# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from pyomo.environ import *
from pyomo.dae import *
from kipet.library.ParameterEstimator import *
from pyomo.core.base.expr import Expr_if
import six
import copy
import re
import os

__author__ = 'Michael Short'  #: October 2018

class EstimabilityAnalyzer(ParameterEstimator):
    """This class is for Estimability analyses. It's first function will be to show the parameter set that
    is suitable for estimation based on a mean squared error (MSE) based approach first described by
    Wu, McLean, Harris, and McAuley (2011). And will contain a number of functions that will perform the 
    estimability analysis. This should eventually be expanded to include a host of functions and methods.

    Parameters
    ----------
    model : ParameterEstimator
        The full model ParameterEstimator problem needs to be fed into the Estimability Analyzer as this is needed
        in order to build the sensitivities for ranking parameters as well as for constructing the simplified models
    """

    def __init__(self, model):
        super(EstimabilityAnalyzer, self).__init__(model)
        #self.full_model = ParameterEstimator(model)
        self.param_ranks = dict()
        
    def run_sim(self, solver, **kdws):
        raise NotImplementedError("EstimabilityAnalyzer object does not have run_sim method. Call run_analyzer")

    def rank_params_yao(self, param_scaling = None, meas_scaling = None):
        """This function ranks parameters in the method described in Yao (2003) by obtaining the sensitivities related
        to the parameters in the model through solving the original NLP model for concentrations, getting the sensitivities
        relating to each paramater, and then using them to predict the next sensitivity. User must provide scaling factors
        as defined in the paper. These are in the form of dictionaries, relating to the confidences relating to the initial
        guesses for the parameters as well as for the confidence inthe measurements.


        Args:
        ----------
        param_scaling: dictionary
        dictionary including each parameter and their relative uncertainty. e.g. a value of 0.5 means that the value
        for the real parameter is within 50% of the guessed value
    
        meas_scaling: scalar
        scalar value showing the certainty of the measurement obtained from the device manufacturer or general 
        knowledge of process
        
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
            meas_scaling = 0.01
            print("WARNING: No scaling for measurments provided by user, so uncertainties based on measurements will be set to 0.01")
        elif meas_scaling != None:
            if isinstance(meas_scaling, int) or isinstance(meas_scaling, float):
                print("meas_scaling", meas_scaling)
            else:
                raise RuntimeError('The meas_scaling must be type int')
        # In order to get the sensitivites the problem is solved using the parameter estimator for concentrations.
        # While this may not be the most efficient way to get the sensitivities for large difficult models,
        # this is the current chosen strategy.
        p_estimator = ParameterEstimator(self.model)
        p_estimator.apply_discretization('dae.collocation',nfe=60,ncp=3,scheme='LAGRANGE-RADAU')
        sigmas = {'A':1e-10,'B':1e-10,'C':1e-11, 'D':1e-11,'E':1e-11,'device':3e-9}
        hessian, results_pyomo = p_estimator.run_opt('k_aug',
                                            variances=sigmas,
                                            tee=True,
                                            #solver_opts = options,
                                            with_d_vars = True,
                                            covariance=True,
                                            estimability=True)
        #Get the appropriate columns with the appropriate parameters
        results_pyomo.C.plot.line(legend=True)
    #    plt.xlabel("time (s)")
    #    plt.ylabel("Concentration (mol/L)")
    #    plt.title("Concentration Profile")
        
        results_pyomo.Z.plot.line(legend=True)
    #    plt.xlabel("time (s)")
    #    plt.ylabel("Concentration (mol/L)")
    #    plt.title("Concentration Profile")
        print(hessian.size)
        nvars = np.size(hessian,0)
        print(hessian)
        nparams = 0
        idx_to_param = {}
        for v in six.itervalues(self.model.P):
            if v.is_fixed():
                print(v, end='\t')
                print("is fixed")
                continue
            print(v)
            idx_to_param[nparams]=v
            nparams += 1
            
        all_H = hessian
        H = all_H[-nparams:, :]
        print(H)
        #euclidean norm for each column of Hessian relating parameters to outputs
        eucnorm = dict()
        #paramdict = dict()
        count=0
        for i in range(nparams):
            print(i)
            total = 0
            for row in range(len(H)):
                total += H[row][count]**2
            print(idx_to_param[i])            
            float(total)
            total = np.asscalar(total)
            print(total)
            sqr = (total)**(0.5)
            eucnorm[count]=sqr
            #paramdict[count]=idx_to_param[i]
            count+=1
           
        print("Euclidean Norms: ", eucnorm)
        
        sorted_euc = sorted(eucnorm.values(), reverse=True)
        print("Sorted Norms: ",sorted_euc)

        count=0
        ordered_params = dict()
        for p in idx_to_param:
            for t in idx_to_param:
                if sorted_euc[p]==eucnorm[t]:
                    ordered_params[count] = t
            count +=1
        print("Euclidean Norms, sorted: ",sorted_euc)
        print("params: ", idx_to_param)
        #print("ordered param dict: ", paramdict)
        print("ordered params:", ordered_params)
        for i in idx_to_param:
            print(i)
            print("idx_to_param[i]:", idx_to_param[i])
            
        for i in ordered_params:
            print(i)
            print("orderedparams[i]:", ordered_params[i])
            
        iter_count=0
        self.param_ranks[1] = idx_to_param[ordered_params[0]]
        for i in self.param_ranks:
            print(i)
            print("parameter ranked first:", self.param_ranks[i])
            
        #The ranking strategy of Yao, where the X and Z matrices are formed
        next_est = dict()
        X= None
        kcol = None
        for i in range(nparams-1):
            print("i", i)
            print(iter_count)
            print("nvars:",nvars)
            if i==0:
                print("hi there")
                X = np.zeros((nvars,1))
                #X = X.reshape((nvars,1))
            print(X)
    
            for k in range(i+1):
                print("iter_count",iter_count)
                
            for x in range(i+1):
                paramhere = ordered_params[x]
                print("paramhere:", paramhere)
                print(x)
                print("Hcol", H[:][ordered_params[x]])
                print(H[:][ordered_params[x]].shape)
                
                kcol = H[:][ordered_params[x]].T
                print("X size: ", X.shape)
                print("kcol size: ", kcol.shape)
                print(kcol)
                recol= kcol.reshape((nvars,1))
                print("recol",recol)
                print("recolshape: ", recol.shape)
                if x >= 1:
                    X = np.append(X,np.zeros([len(X),1]),1)
                print("X",X)
                print(X.shape)
                for n in range(nvars):
                    print("x",x)
                    print("ordered param x",ordered_params[x])
                    print("n",n)
                    print(X[n][x])
                    print(recol[n][0])
                    X[n][x] = recol[n][0]
                print(X)
                print(X.shape)
                #Use Ordinary Least Squares to use X to predict Z
            A = X.T.dot(X)
            print("A",A)
            print("Ashape:", A.shape)
            B= np.linalg.inv(A)
            print("B",B)
            print("B shape: ", B.shape)
            C = X.dot(B)
            print(C)
            print(C.shape)
            D=C.dot(X.T)
            print("D",D)
            print("D shape",D.shape)
            Z = H
            Zbar=D.dot(Z.T)
            print(H)
            print(H.shape)
            print("Zbar:", Zbar)
            print("Zbar shape: ", Zbar.shape)
            #Get residuals of prediction
            Res = Z.T - Zbar
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
            print("magres: ", magres)
            print(ordered_params)
            for i in ordered_params:
                print(i)
                print("ordered_params[i]:", ordered_params[i])
            sorted_magres = sorted(magres.values(), reverse=True)
            print("sorted_magres",sorted_magres)
            count2=0
            for p in idx_to_param:
                for t in idx_to_param:
                    if sorted_magres[p]==magres[t]:
                        next_est[count2] = t
                count2 += 1
            print("next_est",next_est)  
            self.param_ranks[(iter_count+2)]=idx_to_param[next_est[0]]
            iter_count += 1
            for i in self.param_ranks:
                print(i)
                print("self.param_ranks:", self.param_ranks[i])
            print("======================PARAMETER RANKED======================")
            print("len(self.param_ranks)", len(self.param_ranks))
            print("nparam-1", nparams - 1)
            if len(self.param_ranks) == nparams - 1:
                print(len(self.param_ranks))
                print(nparams-1)
                print("All parameters have been ranked")
                break
        
        print("The parameters are ranked in the following order from most estimable to least estimable:")
        for i in self.param_ranks:
            print("Number ", i, "is " self.param_ranks[i])
        
    def run_analyzer(self, method = None, parameter_rankings = None):
        """This function performs the estimability analysis. The user selects the method to be used. The default will
        be selected based on the type of data selected. For now, only the method of Wu, McLean, Harris, and McAuley 
        (2011) using the means squared error is used. Other estimability analysis tools will be added in time. 
        The parameter rankings need to be included as well and this can be done using various methods, however for now, 
        only the Yao (2003) method is used.


        Args:
        ----------
        method: function
            The estimability method to be used. Default is Wu, et al (2011) for concentrations. Others to be added
    
        parameter_rankings: list
            A list containing the parameter rankings in order from most estimable to least estimable. Can be obtained using
            one of Kipet's parameter ranking functions.
        
        returns:
            list of parameters that should remain in the parameter estimation, while all other parameters should be fixed.
        """