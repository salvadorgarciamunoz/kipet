"""
Effort to remove the PyomoSimulator class
"""
# Standard library imports

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

__var = VariableNames()

"""Simulator based on pyomo.dae discretization strategies.

:Methods:
    
    - :func:`apply_discretization`
    - :func:`scale_model`
    - :func:`scale_parameters`
    - :func:`scale_expression`
    - :func:`fix_from_trajectory`
    - :func:`unfix_time_dependent_variable`
    - :func:`initialize_parameters`
    - :func:`build_set_new`
    - :func:`initialize_from_trajectory`
    - :fuch:`scale_variables_from_trajectory`
    - :func:`run_sim`
    - :func:`add_warm_start_suffixes`
    - :func:`update_warm_start`

"""


# def __init__(self, model):
#     """Simulator constructor.

#     :param ConcreteModel model: The Pyomo model of the ReactionModel

#     """
#     self.__var = VariableNames()
#     self.model = model

#     # Most of the attributes are model attributes
#     self.attrs = model_info(self.model)
#     for key, value in self.attrs.items():
#         setattr(self, f'_{key}', value)

#     # Creates a scaling factor suffix
#     if not hasattr(self.model, 'scaling_factor'):
#         self.model.scaling_factor = Suffix(direction=Suffix.EXPORT)

def apply_discretization(model, transformation, **kwargs):
    """Discretizes the model.

    :param str transformation: The type of transformation (only dae.collocation...)
    :param dict kwargs: The options for the discretization

    :return: None

    """
    fixed_times = kwargs.pop('fixed_times', None)

    if not model.alltime.get_discretization_info():

        discretizer = TransformationFactory(transformation)

        if fixed_times == None:
            discretizer.apply_to(model, wrt=model.alltime, **kwargs)
        else:
            discretizer.apply_to(model, wrt=fixed_times, **kwargs)

        alltimes = sorted(model.alltime)
        n_alltimes = len(alltimes)

        # This needs to be looked at in more detail to see if it is still needed.

        # added for optional smoothing parameter with reading values from file CS:
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

        _default_initialization(model)

        # if hasattr(self.model, 'K'):
        #     print('Scaling the parameters')
        #     self.scale_parameters()

    else:
        print('***WARNING: Model already discretized. Ignoring second discretization')


def scale_model(model):
    """Method to scale the model parameters

    :return: None

    """
    if hasattr(model, __var.model_parameter_scaled):
        print('Scaling the parameters')
        scale_parameters()

    return None


def scale_parameters(model):
    """If scaling, this multiplies the constants in model.K to each
    parameter in model.P.
    
    I am not sure if this is necessary and will look into its importance.

    :return: None

    """
    scale = {}
    for i in model.P:
        scale[id(model.P[i])] = model.K[i]

    for var in __var.modeled_states:
        for i in getattr(model, var):
            scale[id(getattr(model, var)[i])] == 1

    for k, v in getattr(model, __var.ode_constraints).items():
        scaled_expr = scale_expression(v.body, scale)
        # self.model.odes[k] = scaled_expr == 0
        getattr(model, __var.ode_constraints)[k] = scaled_expr == 0


def scale_expression(expr, scale):
    """Replace variables in an expression with scaled variables

    :param expression expr: The target expression
    :param dict scale: The mapping of scale factors to the variables

    :return: The expression after the scaled variables have been updated

    """
    visitor = ScalingVisitor(scale)
    return visitor.dfs_postorder_stack(expr)


def fix_from_trajectory(model, variable_name, variable_index, trajectories):
    """Takes in data and fixes a trajectory to the data (interpolates if necessary)

    :param str variable_name: The name of the variable type (Y, Z, etc.)
    :param str variable_index: The name of the variable (A, B, etc.)
    :param pandas.DataFrame trajectories: The dataset containing the target trajectory

    :return: The list of fixed values
    :rtype: list

    """
    if variable_name in __var.modeled_states:
        pass
        # raise NotImplementedError("Fixing state variables is not allowd. Only algebraics can be fixed")

    single_traj = trajectories[variable_index]
    sim_alltimes = sorted(model.alltimes)
    var = getattr(model, variable_name)
    values = interpolate_trajectory(sim_alltimes, single_traj)
    r_values = []
    for i, t in enumerate(sim_alltimes):
        var[t, variable_index].fix(values[i])
        r_values.append((t, values[i]))

    return r_values


def unfix_time_dependent_variable(model, variable_name, variable_index):
    """Sets the fixed attribute of a model variable to False
    
    :param ConcreteModel model: A Pyomo model
    :param str variable_name: The name of the variable to unfix
    :param str variable_index: The specific component of the variable
    
    :return: None
    
    """
    var = getattr(model, variable_name)
    sim_times = sorted(model.alltimes)
    for i, t in enumerate(sim_times):
        var[t, variable_index].fixed = False


def _default_initialization(model):
    """Initializes discreted variables model with initial condition values.

    This method is not intended to be used by users directly

    :return: None

    """
    tol = 1e-4
    attrs = model_info(model)

    if hasattr(model, __var.concentration_model):
        z_init = []
        for t in attrs['alltimes']:
            for k in attrs['mixture_components']:
                if abs(model.init_conditions[k].value) > tol:
                    z_init.append(model.init_conditions[k].value)
                else:
                    z_init.append(1.0)

        z_array = np.array(z_init).reshape((attrs['n_alltimes'], attrs['n_components']))
        z_init_panel = pd.DataFrame(data=z_array,
                                    columns=attrs['mixture_components'],
                                    index=attrs['alltimes'])

        initialize_from_trajectory(model, __var.concentration_model, z_init_panel)

    c_init = []
    if attrs['concentration_given']:
        pass
    else:
        for t in attrs['allmeas_times']:
            for k in attrs['mixture_components']:
                if t in attrs['meas_times']:
                    if abs(model.init_conditions[k].value) > tol:
                        c_init.append(model.init_conditions[k].value)
                    else:
                        c_init.append(1.0)
                else:
                    c_init.append(float('nan'))  # added for new huplc structure!

    if attrs['n_allmeas_times']:
        if attrs['concentration_given']:
            pass
        else:
            c_array = np.array(c_init).reshape((attrs['n_allmeas_times'], attrs['n_components']))
            c_init_panel = pd.DataFrame(data=c_array,
                                        columns=attrs['mixture_components'],
                                        index=attrs['allmeas_times'])

            if hasattr(model, __var.concentration_measured):
                initialize_from_trajectory(model, __var.concentration_measured, c_init_panel)
                print("self._n_meas_times is true in _default_init in PyomoSim")

    if hasattr(model, __var.state_model):
        x_init = []
        for t in attrs['alltimes']:
            for k in attrs['complementary_states']:
                if abs(model.init_conditions[k].value) > tol:
                    x_init.append(model.init_conditions[k].value)
                else:
                    x_init.append(1.0)

        x_array = np.array(x_init).reshape((attrs['n_alltimes'], attrs['n_complementary_states']))
        x_init_panel = pd.DataFrame(data=x_array,
                                    columns=attrs['complementary_states'],
                                    index=attrs['alltimes'])

        initialize_from_trajectory(model, __var.state_model, x_init_panel)


def initialize_parameters(model, params):
    """Initialize the parameters given a dict of parameter values

    :param dict params: A dictionary with parameter keys and initial values

    :return: None

    """
    for k, v in params.items():
        getattr(model, __var.model_parameter).value = v


def build_sets_new(model, variable_name, trajectories):
    """Checks trajectories to see if they have the correct format

    :param str variable_name: The model variable
    :param pandas.DataFrame trajectories: The trajectory data

    :return: the inner and component sets as a tuple
    :rtype: tuple

    """
    var = getattr(model, variable_name)
    index_sets = get_index_sets(var)

    if isinstance(trajectories, pd.DataFrame):
        if variable_name not in __var.__dict__.values():
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

                inner_set = getattr(model, index_sets[index_dict['cont_set'][0]].name)
                component_set = getattr(model, index_sets[index_dict['other_set'][0]].name)

        return inner_set, component_set

    else:
        return None, None


def initialize_from_trajectory(model, variable_name, trajectories):
    """Initializes discretized points with values from trajectories.

    :param str variable_name : Name of the variable in pyomo model
    :param pandas.DataFrame trajectories: Indexed in in the same way the pyomo
        variable is indexed. If the variable is by two sets then the first set is
        the indices of the data frame, the second set is the columns

    :return: None

    """
    if not model.alltime.get_discretization_info():
        raise RuntimeError('apply discretization first before initializing')

    var = getattr(model, variable_name)
    inner_set, component_set = build_sets_new(model, variable_name, trajectories)

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


def scale_variables_from_trajectory(model, variable_name, trajectories):
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

    var = getattr(model, variable_name)
    inner_set, component_set = build_sets_new(model, variable_name, trajectories)

    if inner_set is None and component_set is None:
        return None

    for component in component_set:

        if component not in component_set:
            raise RuntimeError(f'Component {component} is not used in the model')

        nominal_vals = abs(trajectories[component].max())
        if nominal_vals >= tol:
            scale = 1.0 / nominal_vals
            for t in inner_set:
                model.scaling_factor.set_value(var[t, component], scale)

    return None


def run_sim(model, solver, **kwds):
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
    sigmas = kwds.pop('variances', dict())
    tee = kwds.pop('tee', False)
    seed = kwds.pop('seed', None)

    if not model.alltime.get_discretization_info():
        raise RuntimeError('apply discretization first before runing simulation')

    # Adjusts the seed to reproduce results with noise
    np.random.seed(seed)

    # Variables
    # Z_var = self.model.Z
    # if hasattr(self.model, 'Cm'):
    #     C_var = self.model.Cm  # added for estimation with inputs and conc data CS

    # if self._huplc_given: #added for additional data CS
    #     Dhat_var = self.model.Dhat
    #     Chat_var = self.model.Chat

    # # Deactivates objective functions for simulation
    # if self.model.nobjectives():
    #     objectives_map = self.model.component_map(ctype=Objective, active=True)
    #     active_objectives_names = []
    #     for obj in objectives_map.values():
    #         name = obj.getname()
    #         active_objectives_names.append(name)
    #         str_warning = 'Deactivating objective {} for simulation'.format(name)
    #         warnings.warn(str_warning)
    #         obj.deactivate()

    opt = SolverFactory(solver)

    for key, val in solver_opts.items():
        opt.options[key] = val

    solver_results = opt.solve(model, tee=tee, symbolic_solver_labels=True)
    results = ResultsObject()

    # activates objective functions that were deactivated
    # if self.model.nobjectives():
    #     active_objectives_names = []
    #     objectives_map = self.model.component_map(ctype=Objective)
    #     for name in active_objectives_names:
    #         objectives_map[name].activate()

    # retriving solutions to results object
    results.load_from_pyomo_model(model)

    return results
    # c_noise_results = []

    # w = np.zeros((self._n_components, self._n_allmeas_times))
    # n_sig = np.zeros((self._n_components, self._n_allmeas_times))
    # # for the noise term
    # if sigmas:
    #     for i, k in enumerate(self._mixture_components):
    #         if k in sigmas.keys():
    #             sigma = sigmas[k] ** 0.5
    #             dw_k = np.random.normal(0.0, sigma, self._n_allmeas_times)
    #             n_sig[i, :] = np.random.normal(0.0, sigma, self._n_allmeas_times)
    #             w[i, :] = np.cumsum(dw_k)

    # # this addition is not efficient but it can be changed later

    # for i, t in enumerate(self._allmeas_times):
    #     for j, k in enumerate(self._mixture_components):
    #         # c_noise_results.append(Z_var[t,k].value+ w[j,i])
    #         c_noise_results.append(Z_var[t, k].value + n_sig[j, i])

    # c_noise_array = np.array(c_noise_results).reshape((self._n_allmeas_times, self._n_components))
    # results.C = pd.DataFrame(data=c_noise_array,
    #                           columns=self._mixture_components,
    #                           index=self._allmeas_times)

    # #added due to new structure for non_abs species, Cs as subset of C (CS):
    # if hasattr(self, '_abs_components'):
    #     cs_noise_results=[]
    #     for i, t in enumerate(self._allmeas_times):
    #         # if i in self._meas_times:
    #         for j, k in enumerate(self._abs_components):
    #             # c_noise_results.append(Z_var[t,k].value+ w[j,i])
    #             cs_noise_results.append(Z_var[t, k].value + n_sig[j, i])

    #     cs_noise_array = np.array(cs_noise_results).reshape((self._n_allmeas_times, self._nabs_components))
    #     results.Cs = pd.DataFrame(data=cs_noise_array,
    #                               columns=self._abs_components,
    #                               index=self._allmeas_times)


#        addition for inputs estimation with concentration data CS:
# if self._concentration_given == True and self._absorption_given == False:
#     c_noise_results = []
#     for i, t in enumerate(self._allmeas_times):
#         # if i in self._meas_times:
#         for j, k in enumerate(self._mixture_components):
#             c_noise_results.append(C_var[t, k].value)
#     c_noise_array = np.array(c_noise_results).reshape((self._n_allmeas_times, self._n_components))
#     results.C = pd.DataFrame(data=c_noise_array,
#                               columns=self._mixture_components,
#                               index=self._allmeas_times)

# if self._huplc_given == True:
#     results.load_from_pyomo_model(self.model,
#                               to_load=['Chat'])
# s_results = []
# # added due to new structure for non_abs species, non-absorbing species not included in S (CS):
# if hasattr(self, '_abs_components'):
#     for l in self._meas_lambdas:
#         for k in self._abs_components:
#             s_results.append(self.model.S[l, k].value)
# else:
#     for l in self._meas_lambdas:
#         for k in self._mixture_components:
#             s_results.append(self.model.S[l, k].value)

# d_results = []
# if sigmas:
#     sigma_d = sigmas.get('device') ** 0.5 if "device" in sigmas.keys() else 0
# else:
#     sigma_d = 0
# if s_results and c_noise_results:
#     # added due to new structure for non_abs species, Cs and S as above(CS):
#     if hasattr(self,'_abs_components'):
#         for i, t in enumerate(self._meas_times):
#             #if t in self._meas_times:
#                 # print(i, t)
#             for j, l in enumerate(self._meas_lambdas):
#                 suma = 0.0
#                 for w, k in enumerate(self._abs_components):
#                     # print(i, self._meas_times)
#                     Cs = cs_noise_results[i * self._nabs_components + w]
#                     S = s_results[j * self._nabs_components + w]
#                     suma += Cs * S
#                 if sigma_d:
#                     suma += np.random.normal(0.0, sigma_d)
#                 d_results.append(suma)
#                 # print(d_results)
#     else:
#         for i, t in enumerate(self._meas_times):
#             # # print(i, t)
#             # if t in self._meas_times:
#             for j, l in enumerate(self._meas_lambdas):
#                 suma = 0.0
#                 for w, k in enumerate(self._mixture_components):
#                     # print(i, self._meas_times)
#                     C = c_noise_results[i * self._n_components + w]
#                     S = s_results[j * self._n_components + w]
#                     suma += C * S
#                 if sigma_d:
#                     suma += np.random.normal(0.0, sigma_d)
#                 d_results.append(suma)
#                 # print(d_results)
# # added due to new structure for non_abs species, non-absorbing species not included in S (CS):
# if hasattr(self, '_abs_components'):
#     s_array = np.array(s_results).reshape((self._n_meas_lambdas, self._nabs_components))
#     results.S = pd.DataFrame(data=s_array,
#                               columns=self._abs_components,
#                               index=self._meas_lambdas)
# else:
#     s_array = np.array(s_results).reshape((self._n_meas_lambdas, self._n_components))
#     results.S = pd.DataFrame(data=s_array,
#                               columns=self._mixture_components,
#                               index=self._meas_lambdas)


# d_array = np.array(d_results).reshape((self._n_meas_times, self._n_meas_lambdas))
# results.D = pd.DataFrame(data=d_array,
#                          columns=self._meas_lambdas,
#                          index=self._meas_times)

# s_data_dict = dict()
# for t in self._meas_times:
#     for l in self._meas_lambdas:
#         s_data_dict[t, l] = float(results.D[l][t])

# # Added due to estimation with fe-factory and inputs where data already loaded to model before (CS)
# if self._spectra_given:
#     self.model.del_component(self.model.D)
#     self.model.del_component(self.model.D_index)
# #########

# self.model.D = Param(self._meas_times,
#                      self._meas_lambdas,
#                      initialize=s_data_dict)

# param_vals = dict()
# for name in self.model.parameter_names:
#     param_vals[name] = self.model.P[name].value

# results.P = param_vals

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


def update_warm_start(model):
    """Updates the suffixed variables for a warmstart

    :param ConcreteModel model: A Pyomo model

    :return: None

    """
    model.ipopt_zL_in.update(model.ipopt_zL_out)
    model.ipopt_zU_in.update(model.ipopt_zU_out)

    return None
