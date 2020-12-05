"""

This is a new NSD module for use with IPOPT

Goal: To be clean as possible with the implementation and to use as many of
      the existing code as possible - it is all in the reduced_hessian already.
"""
# Standard library imports
import copy

# Third party imports
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from pyomo.environ import (
    Objective,
    SolverFactory,
    Suffix,
    )
from scipy.optimize import (
    Bounds,
    minimize,
    )

# KIPET library imports
from kipet.library.common.parameter_handling import (
    check_initial_parameter_values,
    set_scaled_parameter_bounds,
    )
from kipet.library.common.parameter_ranking import (
    parameter_ratios,
    rank_parameters,
    )
from kipet.library.common.reduced_hessian import (
    add_global_constraints,
    calculate_reduced_hessian,
    optimize_model,
    calculate_duals,
    )
from kipet.library.post_model_build.scaling import (
    remove_scaling,
    scale_parameters,
    update_expression,
    )
from kipet.library.common.objectives import (
    conc_objective,
    comp_objective,
    )
from kipet.library.core_methods.ParameterEstimator import ParameterEstimator

class NSD():
    
    def __init__(self, model_list, init=None):
        
        self.kipet_models = model_list
        self.model_list = [r.model for r in model_list]
        
        self.d_init = {p: [0.5*(v.lb + v.ub), v.lb, v.ub]  for p, v in model_list[0].model.P.items()}
        if init is not None:
            for k, v in init.items():
                self.d_init[k][0] = v
        
        self.method = 'trust-constr'
        
        all_parameters = []
        for model in self.model_list:
            for param in model.P.keys():
                if param not in all_parameters:
                    all_parameters.append(param)
        
        self.parameter_names = all_parameters
        
    def __str__(self):
        
        return 'Nested Schur Decomposition Object'
    
    def set_initial_value(self, init):
        """Add custom initial values for the parameters
        
        Args:
            init (dict): keys are parameters, values are floats
            
        Returns:
            None
            
        """
        for k, v in init.items():
            if k in self.d_init:
                self.d_init[k][0] = v
    
        return None
    
    def _rule_objective(self, model):
        """This function defines the objective function for the given model
        
        Args:
            model (pyomo.core.base.PyomoModel.ConcreteModel): This is the pyomo
            model instance for the estimability problem.
                
        Returns:
            obj (pyomo.environ.Objective): This returns the objective function
            for the estimability optimization.
        
        """
        obj = 0
        obj += 0.5*conc_objective(model)*1e6 
        obj += 0.5*comp_objective(model)*1e6
    
        return Objective(expr=obj)
    
    def _model_preparation(self, scaled=True, use_duals=True):
        """Helper function that should prepare the models when called from the
        main function. Includes the experimental data, sets the objectives,
        simulates to warm start the models if no data is provided, sets up the
        reduced hessian model with "fake data", and discretizes all models

        """
        for model in self.model_list:
            
            if not hasattr(model, 'objective'):
                model.objective = self._rule_objective(model)
            
            # The model needs to be discretized
            model_pe = ParameterEstimator(model)
            model_pe.apply_discretization('dae.collocation',
                                          ncp=3,  #self.ncp,
                                          nfe=50, #self.nfe,
                                          scheme='LAGRANGE-RADAU')
            
            # Here is where the parameters are scaled
            if scaled:
                scale_parameters(model)
                set_scaled_parameter_bounds(model, rho=10) #self.rho)
            
            else:
                check_initial_parameter_values(model)
            
            if use_duals:
                model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
                # model.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
                # model.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
                # model.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
                # model.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)
                # model.red_hessian = Suffix(direction=Suffix.EXPORT)
                # model.dof_v = Suffix(direction=Suffix.EXPORT)
                # model.rh_name = Suffix(direction=Suffix.IMPORT)
                
                # count_vars = 1
                # for k, v in model.P.items():
                #     model.dof_v[k] = count_vars
                #     count_vars += 1
                
                # model.npdp = Suffix(direction=Suffix.EXPORT)
            
        return None
    
    def _generate_bounds_object(self):
        """Creates the Bounds object needed by SciPy for minimization
        
        Returns:
            bounds (scipy Bounds object): returns the parameter bounds for the
                trust-region method
        
        """
        lower_bounds = []
        upper_bounds = []
        
        for k, v in self.d_init.items():
            lower_bounds.append(v[1])
            upper_bounds.append(v[2])
        
        lower_bounds = np.array(lower_bounds, dtype=float)
        upper_bounds = np.array(upper_bounds, dtype=float) 
        bounds = Bounds(lower_bounds, upper_bounds, True)
        return bounds
    
    @staticmethod
    def inner_problem(x, scenarios, parameter_names):
        """Inner problem calculation for the NSD
        
        Args:
            x (np.array): array of parameter values
            
            scenarios (list): list of reaction models
            
            parameter_names (list): list of global parameters
            
        Returns:
            
            objective_value (float): sum of sub-problem objectives
        """
        kwargs = {
            'calc_method': 'global',
            'method': 'k_aug',
            'set_param_bounds': False,
            }
        
        for i, p in enumerate(parameter_names):
            print(f'{p} = {x[i]:0.6f}')
        
        objective_value = 0
        for model in scenarios:
            optimize_model(model, d=x, parameter_set=parameter_names, **kwargs)
            objective_value += model.objective.expr()
            
        return objective_value
    
    @staticmethod
    def calculate_m(x, scenarios, parameter_names):
        """Calculate the vector of duals for the NSD
        
        Args:
            x (np.array): array of parameter values
            
            scenarios (list): list of reaction models
            
            parameter_names (list): list of global parameters
            
        Returns:
            
            m (np.array): vector of duals
            
        """
        m = pd.DataFrame(np.zeros((len(parameter_names), 1)), index=parameter_names, columns=['dual'])
        
        for model_opt in scenarios:
            duals = calculate_duals(model_opt) 
            for param in m.index:
                if param in duals.keys():
                    m.loc[param] = m.loc[param] + duals[param]

        m = m.values.flatten()
        return m
    
    @staticmethod
    def calculate_M(x, scenarios, parameter_names):
        """Calculate the sum of reduced Hessians for the NSD
        
        Args:
            x (np.array): array of parameter values
            
            scenarios (list): list of reaction models
            
            parameter_names (list): list of global parameters
            
        Returns:
            
            M (np.array): sum of reduced Hessians
        """
        M_size = len(parameter_names)
        M = pd.DataFrame(np.zeros((M_size, M_size)), index=parameter_names, columns=parameter_names)
            
        for model in scenarios:
            
            kwargs = {'calc_method': 'global',
                      'scaled': False}
            reduced_hessian = calculate_reduced_hessian(model, parameter_set=parameter_names, **kwargs)
            
            M = M.add(reduced_hessian).combine_first(M)
            M = M[parameter_names]
            M = M.reindex(parameter_names)
        
        M = M.values
      
        return M
    
    def trust_region(self, debug=False, scaled=False):
        """This is the outer problem controlled by a trust region solver 
        running on scipy. This is the only method that the user needs to 
        call after the NSD instance is initialized.
        
        Returns:
            results (scipy.optimize.optimize.OptimizeResult): The results from the 
                trust region optimation (outer problem)
                
            opt_dict (dict): Information obtained in each iteration (use for
                debugging)
                
        """
        # Prepare the models for NSD
        self._model_preparation(scaled=scaled, use_duals=True)
        
        # Set up the initial parameter values
        d_init = self.d_init
        d_vals =  [d[0] for k, d in self.d_init.items()]
        if scaled:
            d_init_unscaled = d_vals
            d_vals = [1 for p in d_vals]

        # Record the parameter values in each iteration
        self.d_iter = []
        def callback(x, *args):
            self.d_iter.append(x)
    
        # Start TR Routine
        if self.method in ['trust-exact', 'trust-constr']:

            tr_options={
                'xtol': 1e-6,
                }
            
            results = minimize(self.inner_problem, 
                               d_vals,
                               args=(self.model_list, self.parameter_names), 
                               method=self.method,
                               jac=self.calculate_m,
                               hess=self.calculate_M,
                               callback=callback,
                               bounds=self._generate_bounds_object(),
                               options=tr_options,
                           )
            
            # Prepare parameter results
            if scaled:
                s_factor = {k: d_init_unscaled[k] for k in self.d_init.keys()}
            else:
                s_factor = {k: 1 for k in self.d_init.keys()}
            
            self.parameters_opt = {k: results.x[i]*s_factor[k] for i, k in enumerate(self.d_init.keys())}
       
        return results
    
    def plot_paths(self, filename=''):
        """Plot the parameter paths through parameter space during the NSD
        algorithm. For diagnostic purposes.
        
        """
        x_data = list(range(1, len(self.d_iter) + 1))
        y_data = np.r_[self.d_iter]
        
        fig = go.Figure()    
        
        for i, params in enumerate(self.d_init.keys()):
            
            fig.add_trace(
                go.Scatter(x = x_data,
                           y = y_data[:, i],
                           name = params,
                      # line=dict(color=colors[i], width=4),
                   )
                )
        
        fig.update_layout(
            title='Parameter Paths in NSD',
            xaxis_title='Iterations',
            yaxis_title='Parameter Values',
            )
    
        plot(fig)
    
        return None

    # def run_simple_newton_step(self, alpha=1, iterations=15, opt_tol=1e-8):
        
    #     # global parameter_names
        
    #     scaled = False
    #     use_duals = True
    #     self._model_preparation(scaled=scaled, use_duals=use_duals)
    #     parameter_set = self.parameter_names
        
    #     ADD_CONSTRAINTS = True
    
    #     options = dict(calc_method='global',
    #                    method='k_aug', 
    #                    scaled=scaled,
    #                    use_duals=use_duals,
    #                    set_up_constraints=ADD_CONSTRAINTS,
    #                    )
    
    #     for i in range(iterations):
            
    #         obj_val = inner_problem(self.model_list, 
    #                                 parameter_set=parameter_set,
    #                                 **options,
    #                                 )
           
    #         # Get the M matrices to determine search direction
    #         M = cM(self.model_list, parameter_set, **options)
    #         m = cm(self.model_list, parameter_set)
            
    #         ADD_CONSTRAINTS = False 
            
    #         # Calculate the search direction
    #         d = np.linalg.inv(M) @ (m)
            
    #         # Update model parameters
    #         for model in self.model_list:
    #             for j, param in enumerate(parameter_set):
    #                 model.P[param].set_value(d[j,0]*alpha + model.P[param].value)
                
    #         model.P .display()
    #         if max(abs(d)) <= opt_tol:
    #             print('Tolerance reached')
    #             break
        
    #     # Update ReactionModel objects with the final parameter values
    #     for m, reaction in enumerate(self.kipet_models):
    #         for k, v in reaction.model.P.items():
    #             if scaled:
    #                 reaction.model.P[k].set_value(self.model_list[m].K[k].value*self.model_list[m].P[k].value)
    #             else:
    #                 reaction.model.P[k].set_value(self.model_list[m].P[k].value)
        
    #     # Plot the results of the fitting
    #     self.plot_results()
        
    #     return None
                
    def plot_results(self):
        for model in self.kipet_models:
            model.simulate()
            model.results.plot(extra_data={'data': model.datasets['C_data'].data, 'label': 'Meas.'})
            
        return None
            
    
if __name__ == '__main__':
   
    from kipet.new_examples.Ex_18_NSD import generate_models
    # returns a list of ReactionModels
    reaction_models = generate_models() 
    
    # Creates the NSD object - takes a list of ReactionModel objects
    nsd = NSD(reaction_models)
    
    # Set the initial values
    nsd.set_initial_value({'k1' : 0.4,
                           'k2' : 1.1}
                          )
    
    # Runs the NSD using simple newton steps
    #nsd.run_simple_newton_step(iterations=1)  
    
    # Runs the TR Method
    results = nsd.trust_region(scaled=False)
    nsd.plot_results()
    nsd.plot_paths()
