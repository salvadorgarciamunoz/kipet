"""
This holds the class PyomoSimulator, which simply modifies a Pyomo model using various methods
"""

# Third library import
import numpy as np
import pandas as pd
from pyomo.environ import Suffix
from pyomo.core.base import TransformationFactory
from pyomo.opt import SolverFactory 

# KIPET library imports
from kipet.calculation_tools.interpolation import interpolate_trajectory
from kipet.model_tools.visitor_classes import ScalingVisitor
from kipet.estimator_tools.results_object import ResultsObject
from kipet.model_tools.pyomo_model_tools import (get_index_sets,
                                                      index_set_info,
                                                      model_info)
from kipet.general_settings.variable_names import VariableNames


class PyomoSimulator:
    """Simulator based on pyomo.dae discretization strategies.

    """

    def __init__(self, model):
        """Simulator constructor.

        :param ConcreteModel model: The Pyomo model of the ReactionModel

        """
        self.__var = VariableNames()
        self.model = model

        # Most of the attributes are model attributes
        self.attrs = model_info(self.model)
        for key, value in self.attrs.items():
            setattr(self, f'_{key}', value)

        # Creates a scaling factor suffix
        if not hasattr(self.model, 'scaling_factor'):
            self.model.scaling_factor = Suffix(direction=Suffix.EXPORT)
            
        self.comps = {}
        if hasattr(self.model, 'abs_components'):
            self.comps['absorbing'] = [k for k in self.model.abs_components] # all that absorb
        else:
            self.comps['absorbing'] = []
        if hasattr(self.model, 'known_absorbance'):
            self.comps['known_absorbance'] = [k for k in self.model.known_absorbance] # all that are known
        else:
            self.comps['known_absorbance'] = []
        
        self.comps['all'] = [k for k in self.model.mixture_components] # all species
        self.comps['unknown_absorbance'] = [k for k in self.comps['absorbing'] if k not in self.comps['known_absorbance']]

    def apply_discretization(self, transformation, **kwargs):
        """Discretizes the model.

        :param str transformation: The type of transformation (only dae.collocation...)
        :param dict kwargs: The options for the discretization

        :return: None

        """
        fixed_times = kwargs.pop('fixed_times', None)
        
        if not self.model.alltime.get_discretization_info():
            
            discretizer = TransformationFactory(transformation)

            if fixed_times == None:
                discretizer.apply_to(self.model, wrt=self.model.alltime, **kwargs)
            else:
                discretizer.apply_to(self.model, wrt=fixed_times, **kwargs)
                
            self._alltimes = sorted(self.model.alltime)
            self._n_alltimes = len(self._alltimes)

            # This needs to be looked at in more detail to see if it is still needed.

            #added for optional smoothing parameter with reading values from file CS:
            # if self._smoothparam_given:
            #     dfps = pd.DataFrame(index=self.model.alltime, columns=self.model.smoothparameter_names)
            #     for t in self.model.alltime:
            #         if t not in self.model.allsmooth_times:  # for points that are the same in original meas times and feed times
            #             dfps.loc[t] = float(22.) #something that is not between 0 and 1
            #         else:
            #             ps_dict_help = dict()
            #             for p in self.model.smoothparameter_names:
            #                 ps_dict_help[t, p] = value(self.model.smooth_param_data[t, p])
            #             dfps.loc[t] = [ps_dict_help[t, p] for p in self.model.smoothparameter_names]
            #     dfallps = dfps
            #     dfallps.sort_index(inplace=True)
            #     dfallps.index = dfallps.index.to_series().apply(
            #         lambda x: np.round(x, 6))  # time from data rounded to 6 digits

            #     dfallpsall = pd.DataFrame(index=self.model.alltime, columns=self.model.smoothparameter_names)
            #     dfsmoothdata = pd.DataFrame(index=sorted(self.model.smooth_param_datatimes), columns=self.model.smoothparameter_names)

            #     for t in self.model.smooth_param_datatimes:
            #         dfsmoothdata.loc[t] = [value(self.model.smooth_param_data[t, p]) for p in self.model.smoothparameter_names]

            #     for p in self.model.smoothparameter_names:
            #         values = interpolate_trajectory(self.model.alltime, dfsmoothdata[p])
            #         for i, ti in enumerate(self.model.alltime):
            #             if float(dfallps[p][ti]) > 1:
            #                 valueinterp=values[i]
            #                 dfallpsall[p][ti] = float(valueinterp)
            #             else:
            #                 dfallpsall.loc[ti] = float(dfallps[p][ti])

            self._default_initialization()
            
            # if hasattr(self.model, 'K'):
            #     print('Scaling the parameters')
            #     self.scale_parameters()
            
        else:
            print('***WARNING: Model already discretized. Ignoring second discretization')
            
    def scale_model(self):
        """Method to scale the model parameters

        :return: None

        """
        if hasattr(self.model, self.__var.model_parameter_scaled):
            print('Scaling the parameters')
            self.scale_parameters()
            
        return None
    
    def scale_parameters(self):
        """If scaling, this multiplies the constants in model.K to each
        parameter in model.P.
        
        I am not sure if this is necessary and will look into its importance.

        :return: None

        """
        self.scale = {}
        for i in self.model.P:
            self.scale[id(self.model.P[i])] = self.model.K[i]

        for var in self.__var.modeled_states:
            for i in getattr(self.model, var):
                self.scale[id(getattr(self.model, var)[i])] == 1
    
        for k, v in getattr(self.model, self.__var.ode_constraints).items():
            scaled_expr = self.scale_expression(v.body, self.scale)
            getattr(self.model, self.__var.ode_constraints)[k] = scaled_expr == 0
    
    @staticmethod
    def scale_expression(expr, scale):
        """Replace variables in an expression with scaled variables

        :param expression expr: The target expression
        :param dict scale: The mapping of scale factors to the variables

        :return: The expression after the scaled variables have been updated

        """
        visitor = ScalingVisitor(scale)
        return visitor.dfs_postorder_stack(expr)

    def fix_from_trajectory(self, variable_name, variable_index, trajectories):
        """Takes in data and fixes a trajectory to the data (interpolates if necessary)

        :param str variable_name: The name of the variable type (Y, Z, etc.)
        :param str variable_index: The name of the variable (A, B, etc.)
        :param pandas.DataFrame trajectories: The dataset containing the target trajectory

        :return: The list of fixed values
        :rtype: list

        """
        if variable_name in  self.__var.modeled_states:
            pass
            # raise NotImplementedError("Fixing state variables is not allowd. Only algebraics can be fixed")

        single_traj = trajectories[variable_index]
        sim_alltimes = sorted(self._alltimes)
        var = getattr(self.model, variable_name)
        values = interpolate_trajectory(sim_alltimes, single_traj)
        r_values = []
        for i, t in enumerate(sim_alltimes):
            var[t, variable_index].fix(values[i])
            r_values.append((t, values[i]))

        return r_values

    def unfix_time_dependent_variable(self, variable_name, variable_index):
        var = getattr(self.model, variable_name)
        sim_times = sorted(self._alltimes)
        for i, t in enumerate(sim_times):
            var[t, variable_index].fixed = False

    # initializes the trajectories to the initial conditions
    def _default_initialization(self):
        """Initializes discreted variables model with initial condition values.

        This method is not intended to be used by users directly

        :return: None

        """
        tol = 1e-4
        
        if hasattr(self.model, self.__var.concentration_model):
            z_init = []
            for t in self._alltimes:
                for k in self._mixture_components:
                    if abs(self.model.init_conditions[k].value) > tol:
                        z_init.append(self.model.init_conditions[k].value)
                    else:
                        z_init.append(1.0)
    
            z_array = np.array(z_init).reshape((self._n_alltimes, self._n_components))
            z_init_panel = pd.DataFrame(data=z_array,
                                        columns=self._mixture_components,
                                        index=self._alltimes)
            
            self.initialize_from_trajectory(self.__var.concentration_model, z_init_panel)

        c_init = []
        if self._concentration_given:
            pass
        else:
            for t in self.model.allmeas_times:
                for k in self._mixture_components:
                    #if t in self._meas_times:
                    if abs(self.model.init_conditions[k].value) > tol:
                        c_init.append(self.model.init_conditions[k].value)
                    else:
                        c_init.append(1.0)
                    #else: c_init.append(float('nan')) #added for new huplc structure!

        if self.model.allmeas_times:
            if self._concentration_given:
                pass
            else:
                c_array = np.array(c_init).reshape((len(self.model.allmeas_times), self._n_components))
                c_init_panel = pd.DataFrame(data=c_array,
                                        columns=self._mixture_components,
                                        index=self.model.allmeas_times)

                if hasattr(self.model, self.__var.concentration_measured):
                    self.initialize_from_trajectory(self.__var.concentration_measured, c_init_panel)
                    print("self._n_meas_times is true in _default_init in PyomoSim")

        if hasattr(self.model, self.__var.state_model):
            x_init = []
            for t in self._alltimes:
                for k in self._complementary_states:
                    if abs(self.model.init_conditions[k].value) > tol:
                        x_init.append(self.model.init_conditions[k].value)
                    else:
                        x_init.append(1.0)
    
            x_array = np.array(x_init).reshape((self._n_alltimes, self._n_complementary_states))
            x_init_panel = pd.DataFrame(data=x_array,
                                        columns=self._complementary_states,
                                        index=self._alltimes)

            self.initialize_from_trajectory(self.__var.state_model, x_init_panel)

    def initialize_parameters(self, params):
        """Initialize the parameters given a dict of parameter values

        :param dict params: A dictionary with parameter keys and initial values

        :return: None

        """
        for k, v in params.items():
            getattr(self.model, self.__var.model_parameter).value = v

    def build_sets_new(self, variable_name, trajectories):
        """Checks trajectories to see if they have the correct format

        :param str variable_name: The model variable
        :param pandas.DataFrame trajectories: The trajectory data

        :return: the inner and component sets as a tuple
        :rtype: tuple

        """
        var = getattr(self.model, variable_name)
        index_sets = get_index_sets(var)
        
        if isinstance(trajectories, pd.DataFrame):
            if variable_name not in self.__var.__dict__.values():
                inner_set = list(trajectories.index)
                component_set = list(trajectories.columns)
            else:
                
                if var.dim() > 1 and var.index_set().dim() == 0:
                    inner_set = list(set([t[0] for t in var.index_set()]))
                    component_set = list(set([t[1] for t in var.index_set()]))
                    inner_set.sort()
                else:
                    index_sets = get_index_sets(var)
                    index_dict = index_set_info(index_sets)
                    
                    inner_set = getattr(self.model, index_sets[index_dict['cont_set'][0]].name)
                    component_set = getattr(self.model, index_sets[index_dict['other_set'][0]].name)
                
            return inner_set, component_set
                
        else:
            return None, None
        
    def initialize_from_trajectory(self, variable_name, trajectories):
        """Initializes discretized points with values from trajectories.

        :param str variable_name : Name of the variable in pyomo model
        :param pandas.DataFrame trajectories: Indexed in in the same way the pyomo
            variable is indexed. If the variable is by two sets then the first set is
            the indices of the data frame, the second set is the columns

        :return: None

        """
        if not self.model.alltime.get_discretization_info():
            raise RuntimeError('apply discretization first before initializing')
            
        var = getattr(self.model, variable_name)
        inner_set, component_set = self.build_sets_new(variable_name, trajectories)
       
        if inner_set is None and component_set is None:
            return None

        for component in component_set:
            if component in trajectories.columns:
                single_trajectory = trajectories[component]
                values = interpolate_trajectory(inner_set, single_trajectory)
                for i, t in enumerate(inner_set):
                    if not np.isnan(values[i]):
                        var[t, component].value = values[i]

        return None

    def scale_variables_from_trajectory(self, variable_name, trajectories):
        """Scales discretized variables with maximum value of the trajectory.

        .. note::

            This method only works with ipopt

        :param str variable_name : Name of the variable in pyomo model
        :param pandas.DataFrame trajectories: Indexed in in the same way the pyomo
            variable is indexed. If the variable is by two sets then the first set is
            the indices of the data frame, the second set is the columns

        :return: None

        """
        tol = 1e-5
        
        var = getattr(self.model, variable_name)
        inner_set, component_set = self.build_sets_new(variable_name, trajectories)
       
        if inner_set is None and component_set is None:
            return None
        
        for component in component_set:
            
            if component not in component_set:
                raise RuntimeError(f'Component {component} is not used in the model')
                
            nominal_vals = abs(trajectories[component].max())
            if nominal_vals >= tol:
                scale = 1.0 / nominal_vals
                for t in inner_set:
                    self.model.scaling_factor.set_value(var[t, component], scale)

        self._ipopt_scaled = True
        return None

    def run_sim(self, solver, **kwds):
        """ Runs simulation by solving nonlinear system with IPOPT

        :param str solver: The name of the nonlinear solver to used
        :param dict kwds: A dict of options passed to the solver

        :Keyword Args:

            - solver_opts (dict, optional): Options passed to the nonlinear solver
            - variances (dict, optional): Map of component name to noise variance. The
              map also contains the device noise variance
            - tee (bool,optional): flag to tell the simulator whether to stream output
              to the terminal or not

        :return: None

        """
        solver_opts = kwds.pop('solver_opts', dict())
        tee = kwds.pop('tee', False)
        seed = kwds.pop('seed', None)

        if not self.model.alltime.get_discretization_info():
            raise RuntimeError('apply discretization first before runing simulation')

        np.random.seed(seed)
        opt = SolverFactory(solver)
        for key, val in solver_opts.items():
            opt.options[key] = val
            
        solver_results = opt.solve(self.model, tee=tee, symbolic_solver_labels=True)
        results = ResultsObject()
        results.load_from_pyomo_model(self.model)

        return results
   
    @staticmethod
    def add_warm_start_suffixes(model, use_k_aug=False):
        """Adds suffixed variables to problem

        :param ConcreteModel model: A Pyomo model
        :param bool use_k_aug: Indicates if k_aug solver is being used

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

        :param ConcreteModel model: A Pyomo model

        :return: None

        """
        model.ipopt_zL_in.update(model.ipopt_zL_out)
        model.ipopt_zU_in.update(model.ipopt_zU_out)
        
        return None
