"""Multiple Experiment Estimator"""

# Standard library imports
import copy
import math
import os
import scipy.stats as st

# Third party imports
import numpy as np
import pandas as pd
from pyomo import *
from pyomo.dae import *
from pyomo.environ import *
from scipy.sparse import coo_matrix

# KIPET library imports
from kipet.common.parameter_handling import initialize_parameters
from kipet.common.objectives import conc_objective, comp_objective, absorption_objective
from kipet.common.read_hessian import split_sipopt_string
from kipet.mixins.PEMixins import PEMixins
from kipet.core_methods.fe_factory import *
from kipet.core_methods.FESimulator import *
from kipet.core_methods.Optimizer import *
from kipet.core_methods.ParameterEstimator import *
from kipet.core_methods.PyomoSimulator import *
from kipet.core_methods.VarianceEstimator import *

from kipet.top_level.variable_names import VariableNames

__author__ = 'Michael Short, Kevin McBride'  #: February 2019 - October 2020

class MultipleExperimentsEstimator(PEMixins, object):
    """This class is for Estimation of Variances and parameters when we have multiple experimental datasets.
    This class relies heavily on the Pyomo block class as we put each experimental class into its own block.
    This blocks are first run individually in order to find good initializations and then they are linked and
    run together in a large optimization problem.

    Parameters
    ----------
    model : TemplateBuilder
        The full model TemplateBuilder problem needs to be fed into the MultipleExperimentsEstimator. This
        pyomo model will form the basis for all the optimization tasks
    """

    def __init__(self, reaction_models):
        
        #super(MultipleExperimentsEstimator, self).__init__()
        self.reaction_models = reaction_models
        self.experiments = list(self.reaction_models.keys())
        self._idx_to_variable = dict()
        
        self.variances = {name: model.variances for name, model in self.reaction_models.items()}
        self.make_sublist()
        
        self._n_meas_times = 0
        self._n_meas_lambdas = 0
        self._n_actual = 0
        self._n_params = 0
        
        self._spectra_given = True
        self._concentration_given = False
        
        self.global_params = None
        self.parameter_means = False
        
        self.__var = VariableNames()
  
    
    def make_sublist(self):

        self._sublist_components = {}        
        for name, model in self.reaction_models.items():
            self._sublist_components[name] = [comp.name for comp in model.components if comp.state == 'concentration']
        return None
        
    def _define_reduce_hess_order_mult(self):
        """This function is used to link the variables to the columns in the reduced
            hessian for multiple experiments.   
           
            Not meant to be used directly by users
        """
        self.model.red_hessian = Suffix(direction=Suffix.IMPORT_EXPORT)
        count_vars = 1
        model_obj = self.model.experiment

        for i in self.experiments:
            if hasattr(self.reaction_models[i].model, 'C'):
            #if self._spectra_given:
                if hasattr(self, 'model_variance') and self.model_variance or not hasattr(self, 'model_variance'):
                    count_vars = self._set_up_reduced_hessian(model_obj[i], self.model.experiment[i].meas_times, self._sublist_components[i], 'C', count_vars)
        
        for i in self.experiments:
            if hasattr(self.reaction_models[i].model, 'S'):
            #if self._spectra_given:
                if hasattr(self, 'model_variance') and self.model_variance or not hasattr(self, 'model_variance'):       
                    count_vars = self._set_up_reduced_hessian(model_obj[i], self.model.experiment[i].meas_lambdas, self._sublist_components[i], 'S', count_vars)
                
        for i in self.experiments:
            for v in model_obj[i].P.values():
                if v.is_fixed():
                    print(v, end='\t')
                    print("is fixed")
                    continue
                self._idx_to_variable[count_vars] = v
                self.model.red_hessian[v] = count_vars
                count_vars += 1
        return None
           
    def _set_up_marks(self, conc_only=True):
        """Set up the data based on the number of species and measurements
        
        """    
        nt = np.cumsum(np.array([len(self.model.experiment[exp].meas_times) for i, exp in enumerate(self.experiments)]))
        self.t_mark = {i: n for i, n in enumerate(nt)}
        nt = nt[-1]
        self._n_meas_times = nt
        
        #nc = np.cumsum(np.array([len(self._sublist_components[exp]) for i, exp in enumerate(self.experiments)]))
        nc = np.cumsum([len(self.reaction_models[k].components) for k in self.reaction_models.keys()])
        
        self.n_mark = {i: n for i, n in enumerate(nc)}
        nc = nc[-1]
        self._n_actual = nc
        
        #nparams = np.cumsum(np.array([self._get_nparams(self.model.experiment[exp]) for i, exp in enumerate(self.experiments)]))
        nparams = np.cumsum([len(self.reaction_models[k].parameters) for k in self.reaction_models.keys()])
         
        self.p_mark = {i: n for i, n in enumerate(nparams)}
        nparams = nparams[-1]
        self._n_params = nparams
        
        if not conc_only:
            
            nw = np.cumsum(np.array([len(self.model.experiment[exp].meas_lambdas) for i, exp in enumerate(self.experiments)]))
            self.l_mark = {i: n for i, n in enumerate(nw)}
            nw = nw[-1]
            self._n_meas_lambdas = nw
            
        else:
            self.l_mark = {}
            self._n_meas_lambdas = 0
        
        print(self.t_mark, self.n_mark, self.p_mark, self.l_mark)
        print(self._n_meas_times, self._n_actual, self._n_params, self._n_meas_lambdas)
        
        return None
    
    def _display_covariance(self, variances_p):
        """Displays the covariance results to the console
        """
        #print(self.confidence_interval)
        number_of_stds = 1#st.norm.ppf(1-(1-self.confidence_interval)/2)
        
        print(number_of_stds)
        
        print('\nParameters:')
        for exp in self.experiments:
            for k, p in self.model.experiment[exp].P.items():
                if p.is_fixed():
                    continue
                print('{}, {}'.format(k, p.value))
        print('\nConfidence intervals:')
        for exp in self.experiments:
            for i, (k, p) in enumerate(self.model.experiment[exp].P.items()):
                if p.is_fixed():
                    continue
                std = (self.variance_scale*variances_p[i])** 0.5
                print('{} ({},{})'.format(k, p.value - number_of_stds*std, p.value + number_of_stds*std))
                
        if hasattr(self.model.experiment[exp], 'Pinit'):
            print('\nLocal Parameters:')
            for exp in self.experiments:
                for k in self.model.experiment[exp].Pinit.keys():
                    self.model.experiment[exp].Pinit[k] = self.model.experiment[exp].init_conditions[k].value
                    print('{}, {}'.format(k, self.model.experiment[exp].Pinit[k].value))
            print('\nConfidence intervals:')
            for exp in self.experiments:
                for i, k in enumerate(self.model.experiment[exp].Pinit.keys()):
                    self.model.experiment[exp].Pinit[k] = self.model.experiment[exp].init_conditions[k].value
                    #std = (self.variance_scale*variances_p[i])** 0.5
                    print('{} ({},{})'.format(k, 
                                              self.model.experiment[exp].Pinit[k].value - (number_of_stds*variances_p[i]**0.5), 
                                              self.model.experiment[exp].Pinit[k].value + (number_of_stds*variances_p[i]**0.5)))
                            
        return None
        
    def _compute_covariance(self, hessian, variances):
        """
        Computes the covariance matrix for the paramaters taking in the Hessian
        matrix and the variances.
        Outputs the parameter confidence intervals.

        This function is not intended to be used by the users directly

        """
        if not self.spectra_problem:
            self._set_up_marks()
            res = {}
            for exp in self.experiments:
                res.update(self._compute_residuals(self.model.experiment[exp], exp_index=exp))

            H = hessian[-self._n_params:, :]
            variances_p = np.diag(H)

        else:         
            self._set_up_marks(conc_only=False) 
            variances_p, covariances_p = self._variances_p_calc(hessian, variances)   
        
        print(f'VP: {variances_p}')
        self._display_covariance(variances_p)
        
        return None
        
    def _compute_B_matrix(self, variances, **kwds):
        """Builds B matrix for calculation of covariances

            This method is not intended to be used by users directly

        Args:
            variances (dict): variances

        Returns:
            None
        """
        nt = self._n_meas_times
        nw = self._n_meas_lambdas
        nc = self._n_actual
        nparams = self._n_params

        npn = np.r_[self.n_mark[0], np.diff(list(self.n_mark.values()))]
        npt = np.r_[self.t_mark[0], np.diff(list(self.t_mark.values()))]
        npl = np.r_[self.l_mark[0], np.diff(list(self.l_mark.values()))]
        npp = np.r_[self.p_mark[0], np.diff(list(self.p_mark.values()))]
        exp_lookup = {i: exp for i, exp in enumerate(self.experiments)}
        ntheta = sum(npn*(npt + npl) + npp)
        
        exp_count = 0
        timeshift = 0 
        waveshift = 0
        
        rows = []
        cols = []
        data = []
        
        meas_times = {exp : {indx: time for indx, time in enumerate(self.model.experiment[exp].meas_times)} for exp in self.experiments}
        meas_lambdas = {exp : {indx: wave for indx, wave in enumerate(self.model.experiment[exp].meas_lambdas)} for exp in self.experiments}
        
        for i in range(nt):
            for j in range(nw):
            
                nc = npn[exp_count]
                if i == self.t_mark[exp_count] and j == self.l_mark[exp_count]:
                    exp_count += 1
                    timeshift = i
                    waveshift = j    
           
                exp = exp_lookup[exp_count]
                
                for comp_num, comp in enumerate(self._sublist_components[exp]):
                    
                    if i - timeshift in list(range(npt[exp_count])):
                        time = meas_times[exp][i - timeshift]
                    
                    if j - waveshift in list(range(npl[exp_count])):
                        wave = meas_lambdas[exp][j - waveshift]
   
                    r_idx1 = i*nc + comp_num
                    r_idx2 = j*nc + comp_num + nc*nt
                    c_idx =  i*nw + j
                    
                    rows.append(r_idx1)
                    cols.append(c_idx)
                    data.append(-2 * self.model.experiment[exp].S[wave, comp].value / (self.variances[exp]['device']))
  
                    rows.append(r_idx2)
                    cols.append(c_idx)
                    data.append(-2 * self.model.experiment[exp].C[time, comp].value / (self.variances[exp]['device']))
                         
        B_matrix = coo_matrix((data, (rows, cols)), shape=(ntheta, nw * nt)).tocsr()
        self.B_matrix = B_matrix
        
        return B_matrix
        
    def _compute_Vd_matrix(self, variances, **kwds):
        """Builds d covariance matrix

            This method is not intended to be used by users directly

        Args:
            variances (dict): variances

        Returns:
            None
        """
        row = []
        col = []
        data = []
        nt = self._n_meas_times
        nw = self._n_meas_lambdas
        nc = self._n_actual

        v_array = np.zeros(nc)
        s_array = np.zeros(nw * nc)
        
        count = 0
        for x in self.experiments:
            for k, c in enumerate(self._sublist_components[x]):
                v_array[count] = variances[x][c]
                count += 1
        
        kshift = 0
        jshift = 0
        knum = 0
        jnum=0
        count=0
        exp_count = 0
        
        for x in self.experiments:
            kshift += knum
            jshift += jnum
            if exp_count != 0:
                kshift+=1

            for j, l in enumerate(self.model.experiment[x].meas_lambdas):
                for k, c in enumerate(self._sublist_components[x]):
                    
                    if exp_count == 0:
                        nc = self.n_mark[exp_count]
                        nt = self.t_mark[exp_count]
                        nw = self.l_mark[exp_count]
                    else: 
                        nt = (self.t_mark[exp_count] - self.t_mark[exp_count - 1] )
                        nw = (self.l_mark[exp_count] - self.l_mark[exp_count - 1])

                    s_array[(j+jshift) * nc + (k+kshift)] = self.model.experiment[x].S[l, c].value
                    idx = (j+jshift) * nc + (k+kshift)
                    knum = max(knum,k)
                    count += 1
                
                jnum = max(jnum,j)
            exp_count += 1

        row = []
        col = []
        data = []
        nt = self._n_meas_times
        nw = self._n_meas_lambdas
        nd = nt * nw
        v_device = list()
        
        for x in self.experiments:
            if 'device' in variances[x]:
                v_device.append(variances[x]['device']) 
        
        exp_count = 0
        v_device_exp = v_device[exp_count]
        for i in range(nt):
            for j in range(nw):
                if i == self.t_mark[exp_count] and j == self.l_mark[exp_count]:
                    exp_count += 1
                    v_device_exp = v_device[exp_count]
                    
                val = sum(v_array[k] * s_array[j * nc + k] ** 2 for k in range(nc)) + v_device_exp
                row.append(i * nw + j)
                col.append(i * nw + j)
                data.append(val)
                for p in range(nw):
                    if j != p:
                        val = sum(v_array[k] * s_array[j * nc + k] * s_array[p * nc + k] for k in range(nc))
                        row.append(i * nw + j)
                        col.append(i * nw + p)
                        data.append(val)

        Vd_matrix = coo_matrix((data, (row, col)), shape=(nd, nd)).tocsr()
        
        self.Vd_matrix = Vd_matrix
    
        return Vd_matrix
    
    """
    Save the sim results in self.sim_results as a dict: self._sim_solved set to True
    """
    
    def solve_hessian(self, solver, **kwargs):
        """Solves for the Hessian regardless of data source - only uses ipopt_sens
        
        Args:
            sigma_sq (dict): variances

            optimizer (SolverFactory): Pyomo Solver factory object

            tee (bool,optional): flag to tell the optimizer whether to stream output
            to the terminal or not

        Returns:
            hessian (np.ndarray): The hessian matrix for covariance calculations
            
        """
        tee = kwargs.pop('tee', True)
        solver_opts = kwargs.pop('solver_opts', dict())
        
        if solver == 'k_aug':
            solver = 'ipopt_sens'
        
        if solver == 'ipopt_sens':
            if not 'compute_red_hessian' in solver_opts.keys():
                solver_opts['compute_red_hessian'] = 'yes'
                
        # Create the optimizer
        optimizer = SolverFactory(solver)
        for key, val in solver_opts.items():
            optimizer.options[key] = val
            
        # Declare the model
        m = self.model
        self._define_reduce_hess_order_mult()
        
        # Solve
        self._tmpfile = "ipopt_hess"
        solver_results = optimizer.solve(m, 
                                          tee=tee,
                                          logfile=self._tmpfile,
                                          report_timing=True)
        
        with open(self._tmpfile, 'r') as f:
            output_string = f.read()
        if os.path.exists(self._tmpfile):
            os.remove(self._tmpfile)

        ipopt_output, hessian_output = split_sipopt_string(output_string)
        hessian = read_reduce_hessian(hessian_output, len(self._idx_to_variable))
        
        return hessian
        
    def _scale_variances(self,):
        """Option to scale the variances for MEE"""
        var_scaled = dict()
        for s,t in self.variances.items():
            maxx = max(list(t.values()))
            ind_var = dict()
            for i,j in t.items():
                ind_var[i] = j/maxx
            var_scaled[s] = ind_var
        self.variances = var_scaled
        self.variance_scale = maxx
        
        return None
    
    # def calculate_parameter_averages(self):
        
    #     p_dict = {}
    #     c_dict = {}
        
    #     for key, model in self.reaction_models.items():
    #         for param, obj in getattr(model, self.__var.model_parameter).items():
    #             if param not in p_dict:
    #                 p_dict[param] = obj.value
    #                 c_dict[param] = 1
    #             else:
    #                 p_dict[param] += obj.value
    #                 c_dict[param] += 1
                    
    #     self.avg_param = {param: p_dict[param]/c_dict[param] for param in p_dict.keys()}
        
    #     return None
        
    # def initialize_parameters(self):
        
    #     self.calculate_parameter_averages()
        
    #     for key, model in self.reaction_models.items():
    #         for param, obj in getattr(model, self.__var.model_parameter).items():
    #             obj.value = self.avg_param[param] 
                
    #     return None
            
    def solve_consolidated_model(self, 
                                 global_params=None,
                                 **kwargs):
        """This function consolidates the individual models into a single
        optimization problem that links the parameters and spectra (if able)
        from each experiment
        
        """
        solver_opts = kwargs.get('solver_opts', {'linear_solver': 'ma57'})
        tee = kwargs.get('tee', False)
        scaled_variance = kwargs.get('scaled_variance', False)
        shared_spectra = kwargs.get('shared_spectra', True)
        solver = kwargs.get('solver', 'ipopt')
        parameter_means = kwargs.get('mean_start', True)
        
        print("\nSOLVING PARAMETER ESTIMATION FOR MULTIPLE DATASETS\n")
       
        combined_model = ConcreteModel()
        
        self.variance_scale = 1
        if scaled_variance == True:
            self._scale_variances()
            
        # if parameter_means:
        #     initialize_parameters(self.reaction_models)
        
        if global_params is None:
            global_params = self.all_params
        
        def build_individual_blocks(m, exp):
            """This function forms the rule for the construction of the individual blocks 
            for multiple experiments, referenced in run_parameter_estimation. This function 
            is not meant to be used by users directly.
            
            Args:
                m (pyomo Concrete model): the concrete model that we are adding the block to
                exp (list): a list containing the experiments
                
            Returns:
                Pyomo model: Pyomo model inside the block
                
            """
            list_components = self.reaction_models[exp].components.names
            #WITH_D_VARS as far as I can tell should always be True, unless we ignore the device noise
            with_d_vars= True
            m = copy.copy(self.reaction_models[exp].p_model)
            
            # Quick fix - I don't know what is causing this
            if hasattr(m, 'alltime_domain'):
                m.del_component('alltime_domain')
            
            if with_d_vars and hasattr(m, 'D'): #self._spectra_given:
              
                m.D_bar = Var(m.meas_times,
                              m.meas_lambdas)
    
                def rule_D_bar(m, t, l):   
                    return m.D_bar[t, l] == sum(getattr(m, self.__var.concentration_spectra)[t, k] * getattr(m, self.__var.spectra_species)[l, k] for k in self.reaction_models[exp].p_estimator._sublist_components)
    
                m.D_bar_constraint = Constraint(m.meas_times,
                                                m.meas_lambdas,
                                                rule=rule_D_bar)
            
            m.error = Var(bounds = (0, None))
                
            def rule_objective(m):
                
                expr = 0
                spectral_term = 0
                concentration_term = 0
                measured_concentration_term = 0
                complementary_state_term = 0
                weights = [1, 1, 1, 1]
                obj_variances = self.variances
                
                if hasattr(m, self.__var.spectra_data):
                    spectral_term = absorption_objective(m, 
                                                 device_variance=obj_variances[exp]['device'],
                                                 g_option=self.reaction_models[exp]._G_data['G_contribution'],
                                                 with_d_vars=with_d_vars,
                                                 shared_spectra=shared_spectra,
                                                 species_list=list_components)
                    
                    concentration_term = conc_objective(m, variance=obj_variances[exp], source='spectra')
                
                if hasattr(m, self.__var.concentration_measured):
                    measured_concentration_term = conc_objective(m, variance=obj_variances[exp])
                
                if hasattr(m, self.__var.state):
                    complementary_state_term = comp_objective(m, variance=obj_variances[exp])
                    
                expr = weights[0]*spectral_term + \
                    weights[1]*concentration_term + \
                    weights[2]*measured_concentration_term + \
                    weights[3]*complementary_state_term
    
                return m.error == expr
    
            m.obj_const = Constraint(rule=rule_objective)
                
            return m  
        
        combined_model.experiment = Block(self.experiments, rule=build_individual_blocks)
        combined_model.map_exp_to_count = dict(enumerate(self.experiments))
        
        def param_linking_rule(m, exp, param):
            prev_exp = None
            key = next(key for key, value in combined_model.map_exp_to_count.items() if value == exp)
            if key == 0:
                return Constraint.Skip
            else:
                for key, val in combined_model.map_exp_to_count.items():
                    if val == exp:
                        prev_exp = combined_model.map_exp_to_count[key-1]
                if param in global_params and prev_exp != None:
                    return getattr(combined_model.experiment[exp], self.__var.model_parameter)[param] == getattr(combined_model.experiment[prev_exp], self.__var.model_parameter)[param]
                else:
                    return Constraint.Skip
        
        set_fixed_params=set()
        
        for exp in self.experiments:
            for param, param_obj in getattr(combined_model.experiment[exp], self.__var.model_parameter).items():
                if param_obj.is_fixed():
                    set_fixed_params.add(param)
                    
        print("Fixed parameters are: ", set_fixed_params)
        
        set_params_across_blocks = self.all_params.difference(set_fixed_params)
        combined_model.parameter_linking = Constraint(self.experiments, set_params_across_blocks, rule = param_linking_rule)
        
        def wavelength_linking_rule(m, exp, wave, comp):
            prev_exp = None
            key = next(key for key, value in combined_model.map_exp_to_count.items() if value == exp)
            if key == 0:
                return Constraint.Skip
            else:
                for key, val in combined_model.map_exp_to_count.items():
                    if val == exp:
                        prev_exp = combined_model.map_exp_to_count[key-1]
                if wave in self.all_wavelengths and prev_exp != None:
                    if comp in combined_model.experiment[prev_exp].mixture_components and comp in combined_model.experiment[exp].mixture_components:
                        return getattr(combined_model.experiment[exp], self.__var.spectra_species)[wave,comp] == getattr(combined_model.experiment[prev_exp], self.__var.spectra_species)[wave,comp]
                    else:
                        return Constraint.Skip
                else:
                    return Constraint.Skip

        if shared_spectra == True:
            combined_model.spectra_linking = Constraint(self.experiments, self.all_wavelengths, self.all_species, rule = wavelength_linking_rule)
        
        # Add in experimental weights
        combined_model.objective = Objective(sense=minimize, expr=sum(b.error for b in combined_model.experiment[:]))
        
        self.model = combined_model
        
        if solver in ['k_aug', 'ipopt_sens']:
            covariance = True
            hessian = self.solve_hessian(solver, tee=tee, solver_opts=solver_opts)
            self._compute_covariance(hessian, self.variances)
        else:
            optimizer = SolverFactory('ipopt')
            optimizer.solve(combined_model, options=solver_opts, tee=True)            
        
        solver_results = {}
        
        for i in combined_model.experiment:
            solver_results[i] = ResultsObject()
            solver_results[i].load_from_pyomo_model(combined_model.experiment[i])
                                            
        return solver_results
    
    @property
    def all_params(self):
        set_of_all_model_params = set()
        for name, model in self.reaction_models.items():
            set_of_all_model_params = set_of_all_model_params.union(model.parameters.names)
        return set_of_all_model_params
    
    
    @property
    def all_wavelengths(self):
        set_of_all_wavelengths = set()
        for name, model in self.reaction_models.items():
            set_of_all_wavelengths = set_of_all_wavelengths.union(list(model.model.meas_lambdas))
        return set_of_all_wavelengths
    
    @property
    def all_species(self):
        set_of_all_species = set()
        for name, model in self.reaction_models.items():
            set_of_all_species = set_of_all_species.union(model.components.names)
        return set_of_all_species
    
#%%

# combined_model = kipet_model.mee.model

# from kipet.core_methods.ResultsObject import ResultsObject

# for i in combined_model.experiment:
#       solver_results[i] = ResultsObject()
#       solver_results[i].load_from_pyomo_model(combined_model.experiment[i])
                                            
# print(solver_results)
