"""
ReactionModel class

This is a big wrapper class for most of the KIPET methods
"""
# Standard library imports
import collections
import copy
import os
import pathlib
import sys

# Third party imports
import numpy as np
import pandas as pd
from pyomo.environ import ConcreteModel, Set, Var
from pyomo.core.base.var import IndexedVar
from pyomo.dae.diffvar import DerivativeVar

# Kipet library imports
import kipet.core_methods.data_tools as data_tools
from kipet.core_methods.EstimationPotential import (
    replace_non_estimable_parameters,
    rhps_method,
    )
from kipet.core_methods.EstimabilityAnalyzer import EstimabilityAnalyzer
from kipet.core_methods.FESimulator import FESimulator
from kipet.core_methods.ParameterEstimator import ParameterEstimator
from kipet.core_methods.PyomoSimulator import PyomoSimulator
from kipet.core_methods.TemplateBuilder import TemplateBuilder
from kipet.core_methods.VarianceEstimator import VarianceEstimator
# from kipet.common.component_expression import get_unit_model
from kipet.common.model_funs import step_fun
from kipet.post_model_build.pyomo_model_tools import get_vars
from kipet.dev_tools.display import Print
from kipet.post_model_build.scaling import scale_models
from kipet.post_model_build.replacement import ParameterReplacer
from kipet.mixins.TopLevelMixins import WavelengthSelectionMixins

from kipet.top_level.data_component import (
    DataBlock, 
    DataSet,
    )
from kipet.top_level.spectral_handler import SpectralData
from kipet.top_level.element_blocks import (
    AlgebraicBlock,
    ComponentBlock,
    ConstantBlock, 
    ParameterBlock, 
    StateBlock,
    )
from kipet.top_level.expression import (
    AEExpressions,
    Expression,
    ODEExpressions,
    )
from kipet.top_level.helper import DosingPoint
from kipet.top_level.settings import (
    Settings, 
    )
from kipet.top_level.variable_names import VariableNames
# from kipet.common.VisitorClasses import FindingVisitor
from pyomo.environ import units as pyo_units

from kipet.model_components.units_handler import convert_single_dimension

__var = VariableNames()
DEBUG=__var.DEBUG
_print = Print(verbose=DEBUG)
model_components = ['parameters', 'components', 'states', 'algebraics', 'constants', 'steps']

class ReactionModel(WavelengthSelectionMixins):
    
    """This should consolidate all of the Kipet classes into a single class to
    enable a simpler framework for using the software. 
    
    """
    __var = VariableNames()
    
    def __init__(self, *args, **kwargs):
        
        
        # self.Parameter = Parameter
        # self.Component = Component
        
        self.name = kwargs.get('name', 'Model-1')
        self.model = None
        self.builder = TemplateBuilder()
        self.components = ComponentBlock()   
        self.parameters = ParameterBlock()
        self.constants = ConstantBlock()
        self.algebraics = AlgebraicBlock()
        self.states = StateBlock()
        self.datasets = DataBlock()
        self.data = {}
        self.spectra = None
        
        self.results_dict = {}
        self.settings = Settings(category='model')
        self.algebraic_variables = []
        self.variances = {}
        self.odes = ODEExpressions()
        self.algs = AEExpressions()
        self.odes_dict = {}
        self.algs_dict = {}
        self.__flag_odes_built = False
        self.__flag_algs_built = False
        self.custom_objective = None
        self.optimized = False
        self.dosing_var = None
        self.dosing_points = None
        self._has_dosing_points = False
        self._has_step_or_dosing = False
        self._has_non_absorbing_species = False
        self._var_to_fix_from_trajectory = []
        self._var_to_initialize_from_trajectory = []
        self._default_time_unit = 'seconds'
        self.unit_base = kwargs.get('unit_base', None)
        
        self._G_data = {'G_contribution': None, 'Z_in': dict(), 'St': dict()}
        self.__var = VariableNames()

        # self.components_used = set()
        self.component_map = {k: getattr(self, k) for k in ['components',
                                                            'states',
                                                            'parameters',
                                                            'algebraics',
                                                            'constants']}
        
    def __repr__(self):
        
        # m = 20
        
        # kipet_str = f'ReactionModel Object {self.name}:\n\n'
        # kipet_str += f'{"ODEs".rjust(m)} : {hasattr(self, "odes") and getattr(self, "odes") is not None}\n'
        # kipet_str += f'{"Algebraics".rjust(m)} : {hasattr(self, "odes") and getattr(self, "odes") is not None}\n'
        # kipet_str += f'{"Model".rjust(m)} : {hasattr(self, "model") and getattr(self, "model") is not None}\n'
        # kipet_str += '\n'
        
        # kipet_str += f'{self.components}\n'
        # kipet_str += f'Algebraic Variables:\n{", ".join([str(av) for av in self.algebraic_variables])}\n\n'
        # kipet_str += f'{self.parameters}\n'
        # kipet_str += f'{self.datasets}\n'
        
        return f'KipetModel {self.name}'#kipet_str
    
    def __str__(self):
        
        m = 20
        
        kipet_str = f'ReactionModel Object {self.name}:\n\n'
        kipet_str += f'{"ODEs".rjust(m)} : {hasattr(self, "odes") and getattr(self, "odes") is not None}\n'
        kipet_str += f'{"Algebraics".rjust(m)} : {hasattr(self, "odes") and getattr(self, "odes") is not None}\n'
        kipet_str += f'{"Model".rjust(m)} : {hasattr(self, "model") and getattr(self, "model") is not None}\n'
        kipet_str += '\n'
        
        kipet_str += f'{self.components}\n'
        kipet_str += f'Algebraic Variables:\n{", ".join([str(av) for av in self.algebraic_variables])}\n\n'
        kipet_str += f'{self.parameters}\n'
        kipet_str += f'{self.datasets}\n'
    
        return kipet_str
    
    """Model Components"""
    
    def _make_set_up_model(self):
        """Make the dummy model for initial pyomo vars
        
        """
        self._set_up_model = ConcreteModel()
        self._set_up_model.indx = Set(initialize=[0])
        return None
    
    def Component(self, name, index, units):
        """Creates the initial pyomo variables for model building
        
        """
        if not hasattr(self, '_set_up_model'):
            self._make_set_up_model()
        
        setattr(self._set_up_model, f'{name}_indx', Set(initialize=[0]))
        sets = [getattr(self._set_up_model, f'{name}_indx')]*index

        if units is None:
            p_units = str(1)
            comp_units = 1*self.unit_base.ur('')

        else:
            con = 1
            margin = 6
            print(f'Checking units for {name.rjust(margin)}: \t{units}')
            comp_units = convert_single_dimension(self.unit_base.ur, units, self.unit_base.TIME_BASE)
            con *= comp_units.m
            comp_units = convert_single_dimension(self.unit_base.ur, str(comp_units.units), self.unit_base.VOLUME_BASE)
            con *= comp_units.m
            p_units = con*getattr(pyo_units, str(comp_units.units))
            comp_units = con*comp_units.units
            if not self.unit_base.ur(units) == comp_units:
                print(f'    Converted to........: \t{comp_units.units}')
            
        setattr(self._set_up_model, name, Var(*sets, initialize=1, units=p_units))
        var = getattr(self._set_up_model, name)
        return var[tuple([0 for i in range(index)])], comp_units
    
    def _add_model_component(self, name, index, model_var, *args, **kwargs):
        """Generic method for adding modeling components to the ReactionModel
        
        """
        units = kwargs.get('units', None)
        value = kwargs.get('value', 1)
        kwargs['units_orig'] = units
        
        par, con = self.Component(args[0], index, units=units)
        #print(f'#### in the add model: {con}')
        
        kwargs['value'] = value*con.m
        kwargs['units'] = str(con.units)
        kwargs['conversion_factor'] = con.m
        
        if 'bounds' in kwargs:
        # if hasattr(key_comp, 'bounds') and key_comp.bounds is not None:
            
            bounds = list(kwargs.get('bounds', [0, 0]))
            if bounds[0] is not None:
                bounds[0] *= con.m
            if bounds[1] is not None:
                bounds[1] *= con.m
            kwargs['bounds'] = (bounds) 
        
        kwargs['unit_base'] = self.ub
        kwargs['pyomo_var'] = par
        kwargs['model_var'] = model_var
        getattr(self, f'{name}s').add_element(*args, **kwargs)
        return par
    
    def parameter(self, *args, **kwargs):
        """Create a parameter with a localized pyomo var
        
        """
        return self._add_model_component('parameter',
                                        1, 
                                        self.__var.model_parameter, 
                                        *args, 
                                        **kwargs)
        
    def component(self, *args, **kwargs):
        """Create a component with a localized pyomo var
        
        """
        return self._add_model_component('component', 
                                        2, 
                                        self.__var.concentration_model, 
                                        *args, 
                                        **kwargs)
    
    def state(self, *args, **kwargs):
        """Create a state with a localized pyomo var
        
        """
        return self._add_model_component('state', 
                                        2, 
                                        self.__var.state_model, 
                                        *args, 
                                        **kwargs)
    
    def constant(self, *args, **kwargs):
        """Create a constant with a localized pyomo var
        
        """
        return self._add_model_component('constant', 
                                        1, 
                                        self.__var.model_constant, 
                                        *args, 
                                        **kwargs)
        
    def algebraic(self, *args, **kwargs):
        """Create a algebraic variable with a localized pyomo var
        
        """
        print('Warning: This is deprecated and will be removed in a future version')
        
        return self._add_model_component('algebraic', 
                                        2, 
                                        self.__var.algebraic, 
                                        *args, 
                                        **kwargs)
    
    def fixed_state(self, *args, **kwargs):
        """Create a algebraic variable for fixed states
        
        """
        return self._add_model_component('algebraic', 
                                        2, 
                                        self.__var.algebraic, 
                                        *args, 
                                        **kwargs)
        
    
    # def variable(self, *args, **kwargs):
    #     """Create a generic variable with a localized pyomo var
          # This will be a generic method that others can use...
    #     """
    #     if kwargs.get('index', None) is None:
    #         raise ValueError('Custom variables require an index list')
        
    #     return self._add_model_component('custom', 
    #                                     kwargs['index'], 
    #                                     'V', 
    #                                     *args, 
    #                                     **kwargs)
    
    def step(self, name, *args, **kwargs):
        """Create a step variable with a localized pyomo var
        
        """
        self._has_step_or_dosing = True
        if not hasattr(self, '_step_list'):
            self._step_list = {}
            
        var_name = 'step'
        if name not in self._step_list:
            self._step_list[name] = [kwargs]
        else:
            self._step_list[name].append(kwargs)
            
        par, _ = self.Component(name, 2, None)
        if not hasattr(self, f'{var_name}s'):
            setattr(self, f'{var_name}s', {})
        getattr(self, f'{var_name}s')[name] = [self.__var.step_variable, par]
        return par
    
    """Dosing profiles"""
    
    def add_dosing_point(self, component, time, step):
        """Add a dosing point or several (check template for how this is handled)
        
        """
        conversion_dict = {'state': self.__var.state_model, 
                           'concentration': self.__var.concentration_model,
                           }
        if component not in self.components.names:
            raise ValueError('Invalid component name')
        dosing_point = DosingPoint(component, time, step)
        model_var = conversion_dict[self.components[component].state]
        if self.dosing_points is None:
            self.dosing_points = {}
        if model_var not in self.dosing_points.keys():
            self.dosing_points[model_var] = [dosing_point]
        else:
            self.dosing_points[model_var].append(dosing_point)
        self._has_dosing_points = True
        self._has_step_or_dosing = True
        
    def add_dosing(self, n_steps=1):
        """At the moment, this is needed to set up the dosing variable
        """
        self._has_step_or_dosing = True
        self._number_of_steps = n_steps
    
        return None
    
    def call_fe_factory(self):
        """Somewhat of a wrapper for this simulator method, but better"""

        self.simulator.call_fe_factory({
            self.__var.dosing_variable: [self.__var.dosing_component]},
            self.dosing_points)

        return None
    
    """Model data"""
    
    def add_data(self, *args, **kwargs):
        
        name = kwargs.get('name', None)
        time_scale = kwargs.get('time_scale', self.unit_base.TIME_BASE)
        
        time_conversion = 1
        if time_scale != self.unit_base.TIME_BASE:
            time_conversion = self.ub.ur(time_scale).to(self.unit_base.TIME_BASE).m
        
        if len(args) > 0:
            name = args[0]
        if name is None:
            name = f'ds{len(self.data) + 1}'
            
        filename = kwargs.get('file', None)
        data = kwargs.pop('data', None)
        category = kwargs.pop('category', None)
        remove_negatives = kwargs.get('remove_negatives', False)
    
        if filename is not None:
            calling_file_name = os.path.dirname(os.path.realpath(sys.argv[0]))
            filename = pathlib.Path(calling_file_name).joinpath(filename)
            kwargs['file'] = filename
            dataframe = data_tools.read_file(filename)
        elif filename is None and data is not None:
            dataframe = data
        else:
            raise ValueError('User must provide filename or dataframe')
        
        dataframe.index = dataframe.index*time_conversion
        
        if category == 'spectral':
            D_data = SpectralData(name, remove_negatives=remove_negatives)    
            D_data.add_data(dataframe)
            self.spectra = D_data
            return None
                    
        else:
            self.data[name] = dataframe
            dataset = DataSet(name,
                              category=category,
                              data = dataframe,
                              file = filename,
                              )
            
            if remove_negatives:
                dataset.remove_negatives()
            self.datasets.add_dataset(dataset)
        
        self._check_data_matches(name)
        self.datasets._check_duplicates()
            
        return None
    
    # Check for duplicates ==> DataBlock
    def _check_data_matches(self, name, name_is_from_model=False):
        """Easy data mapping to ElementBlocks"""
        # Reassignment
        
        # for name, dataset in self.data.items():
        #     print(name)
        #     for col in dataset.columns:
        #         print(col)
        #         for block in blocks:
        #             comp_set = getattr(self, block).names
        #             for comp in comp_set:
        #                 if col in comp_set:
        #                     print(block)
        #                     getattr(self, block)[col].data = dataset[col]
            
        blocks = ['components', 'states', 'algebraics']
        
        if not name_is_from_model:
            # matched_data_vars = set()
            # all_data_vars = set()
            dataset = self.datasets[name]
            if dataset.category in ['spectral', 'trajectory']:
                return None
            else:
                for block in blocks:
                    for col in dataset.data.columns:
                        #all_data_vars.add(col)
                        if col in getattr(self, block).names:
                            setattr(getattr(self, block)[col], 'data_link', dataset.name)
                            #matched_data_vars.add(col)
        
            #unmatched_vars = all_data_vars.difference(matched_data_vars)

        else:
            for block in blocks:
                if name in getattr(self, block).names:
                    for dataset in self.datasets:
                        for col in dataset.data.columns:
                            if col == name:
                                setattr(getattr(self, block)[name], 'data_link', dataset.name)

        return None

    """Template building"""

    def set_times(self, start_time=None, end_time=None):
        """Add times to model for simulation (overrides data-based times)"""
        
        if start_time is None or end_time is None:
            raise ValueError('Time needs to be a number')
        
        self.settings.general.simulation_times = (start_time, end_time)
        return None
    
    def _unwanted_G_initialization(self, *args, **kwargs):
        """Prepare the ParameterEstimator model for unwanted G contributions
        
        """
        self.builder.add_qr_bounds_init(bounds=(0,None),init=1.1)
        self.builder.add_g_bounds_init(bounds=(0,None))
        
        return None
    
    """ Expressions """
    
    def add_expression(self, name, expr, *args, **kwargs):
        """Add expressions (prev. algebraics) to the reaction model
        
        """        
        args_ = list(args)
        args_.insert(0, name)
        args = tuple(args_)
        
        # Adds algebraics anyways, for comparison purposes
        self._add_model_component('algebraic', 
                                  2, 
                                  self.__var.algebraic, 
                                  *args, 
                                  **kwargs)
        
        expr_ = Expression(name, expr)
        expr_.check_division()
        self.algs_dict.update(**{name: expr_})
        return expr
        
    def expression(self, label, expr):
        """
        Parameters
        ----------
        label : TYPE
            DESCRIPTION.
        expr : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        expr = Expression(label, expr)
        if not hasattr(self, 'expr_dict'):
            self.expr_dict = {}
        self.expr_dict.update(**{label: expr})
        return None
    
    def add_ode(self, ode_var, expr):
        """Method to add an ode expression to the ReactionModel
        
        Args:
            ode_var (str): state variable
            
            expr (Expression): expression for rate equation as Pyomo Expression
            
        Returns:
            None
        
        """
        expr = Expression(ode_var, expr)
        self.odes_dict.update(**{ode_var: expr})
        
        return None
    
    def add_odes(self, ode_fun):
        """Takes in a dict of ODE expressions and sends them to add_ode
        
        Args:
            ode_fun (dict): dict of ode expressions
            
        Returns:
            None
        """
        if isinstance(ode_fun, dict):
            for key, value in ode_fun.items():
                self.add_ode(key, value)
        
        return None

    def _build_odes(self):
        """Builds the ODEs by passing them to the ODEExpressions class
        
        """
        odes = ODEExpressions(self.odes_dict)
        setattr(self, 'ode_obj', odes)
        self.odes = odes.exprs
        self.__flag_odes_built = True
        
        return None
        
    def add_algebraic(self, alg_var, expr):
        """Method to add an algebraic expression to the ReactionModel
        
        Args:
            alg_var (str): state variable
            
            expr (Expression): expression for algebraic equation as Pyomo
                Expression
            
        Returns:
            None
        
        """
        expr = Expression(alg_var, expr)
        expr.check_division()
        self.algs_dict.update(**{alg_var: expr})
    
        return None
    
    def add_algebraics(self, alg_fun):
        """Takes in a dict of algebraic expressions and sends them to
        add_algebraic
        
        Args:
            algebraics (dict): dict of algebraic expressions
            
        Returns:
            None
        """
        if isinstance(alg_fun, dict):
            for key, value in alg_fun.items():
                self.add_algebraic(key, value)
        
        return None
    
    def _build_algs(self):
        """Builds the algebraics by passing them to the AEExpressions class
        
        """ 
        algs = AEExpressions(self.algs_dict)
        setattr(self, 'alg_obj', algs)
        self.algs = algs.exprs
        self.__flag_algs_built = True

        return None
    
      
    def add_objective_from_algebraic(self, algebraic_var):
        """Declare an algebraic variable that is to be used in the objective
        
        TODO: add multiple alg vars for obj
        
        Args:
            algebraic_var (str): Variable representing the formula to be added
                to the objective
                
        Returns:
            None
            
        """
        self._check_data_matches(algebraic_var, name_is_from_model=True)
        self.custom_objective = algebraic_var
        
        return None
    
    def populate_template(self, *args, **kwargs):
        """Method handling all of the preparation for the TB
        
        """
        with_data = kwargs.get('with_data', True)
        
        # for model_component, obj in self.component_map.items():
            
        #     if len(obj) > 0:
        #         self.builder.add_model_element(obj)
                
        if len(self.states) > 0:
            self.builder.add_model_element(self.states)
        
        if len(self.algebraics) > 0:
            self.builder.add_model_element(self.algebraics)
        
        if len(self.components) > 0:
            self.builder.add_model_element(self.components)
        else:
            raise ValueError('The model has no components')
            
        if len(self.parameters) > 0:
            self.builder.add_model_element(self.parameters)
        else:
            self.allow_optimization = False   
        
        if len(self.constants) > 0:
            self.builder.add_model_element(self.constants)
        
        if with_data:
            if len(self.datasets) > 0 or self.spectra is not None:
                self.builder.input_data(self.datasets, self.spectra)
                self.allow_optimization = True
            elif len(self.datasets) == 0 and self.spectra is None:
                self.allow_optimization = False
            else:
                pass
            
        # Add the ODEs
        if len(self.odes_dict) != 0:
            self._build_odes()
            self.builder.set_odes_rule(self.odes)
        elif 'odes_bypass' in kwargs and kwargs['odes_bypass']:
            pass
        else:
            raise ValueError('The model requires a set of ODEs')
            
        if len(self.algs_dict) != 0:
            self._build_algs()
            if isinstance(self.algs, dict):
                self.builder.set_algebraics_rule(self.algs, asdict=True)
            else:
                self.builder.set_algebraics_rule(self.algs)
            
        if hasattr(self, 'custom_objective') and self.custom_objective is not None:
            self.builder.set_objective_rule(self.custom_objective)
        
        self.builder.set_parameter_scaling(self.settings.general.scale_parameters)
        self.builder.add_state_variance(self.components.variances)
        
        # if self._has_step_or_dosing:
        #     self.builder.add_dosing_var(self._number_of_steps)
        
        if self._has_dosing_points:
            self._add_feed_times()
            
        if hasattr(self, '_step_list') and len(self._step_list) > 0:
            self.builder.add_step_vars(self._step_list)
            
        # It seems this is repetitive - refactor
        if hasattr(self, '_G_contribution') and self._G_contribution is not None:
        #if self.settings.parameter_estimator.G_contribution is not None:
            self._unwanted_G_initialization()
            self.builder._G_contribution = self._G_contribution
        else:
            self.builder._G_contribution = None
        
        start_time, end_time = None, None
        if self.settings.general.simulation_times is not None:
            #print(f'times are: {type(self.settings.general.simulation_times)}')
            start_time, end_time = self.settings.general.simulation_times
       
        return start_time, end_time
    
    def _make_c_dict(self):
        
        model_components = ['parameters', 'components', 'states', 'algebraics', 'constants']
        
        if not hasattr(self, 'c'):
            self.c = dict()
            for mc in model_components: 
                if hasattr(self, f'{mc}') and len(getattr(self, f'{mc}')) > 0:
                    for comp in getattr(self, f'{mc}'):
                        #print(comp)
                        #print(comp.model_var, comp.pyomo_var)
                        self.c.update({comp.name: [comp.model_var, comp.pyomo_var]})
    
            if hasattr(self, 'steps'):
                self.c.update(getattr(self, 'steps'))
    
    def create_pyomo_model(self, *args, **kwargs):
        """Adds the component, parameter, data, and odes to the TemplateBuilder
        instance and creates the model. The model is stored under self.model
        and there is nothing returned.

        Args:
            None

        Returns:
            None

        """
        if hasattr(self, 'model'):
            del self.model
            
        skip_non_abs = kwargs.pop('skip_non_abs', False)
            
        start_time, end_time = self.populate_template(*args, **kwargs)
        self._make_c_dict()
        setattr(self.builder, 'c_mod', self.c)
        
        self.model = self.builder.create_pyomo_model(start_time, end_time)
        
        non_abs_comp = self.components.get_match('absorbing', False)
        
        if not skip_non_abs and len(non_abs_comp) > 0:
            self.builder.set_non_absorbing_species(self.model, non_abs_comp, check=True)    
        
        if hasattr(self,'fixed_params') and len(self.fixed_params) > 0:
            for param in self.fixed_params:
                self.model.P[param].fix()
            
        return None
    
    def _add_feed_times(self):
        
        feed_times = set()
        
        for model_var, dp in self.dosing_points.items():
            for point in dp:
                feed_times.add(point.time)
        
        self.builder.add_feed_times(list(feed_times))
        return None
    
    def _from_trajectories(self, estimator):
        """This handles all of the fixing, initializing, and scaling from 
        trajectory data
        
        """
        self._var_to_fix_from_trajectory = self.algebraics.fixed
        
        if len(self._var_to_fix_from_trajectory) > 0:
            for fix in self._var_to_fix_from_trajectory:
                if isinstance(fix[2], str):
                    if fix[2] in self.datasets.names:
                        fix[2] = self.datasets[fix[2]].data
                getattr(self, estimator).fix_from_trajectory(*fix)
                
        if len(self._var_to_initialize_from_trajectory) > 0:
            for init in self._var_to_initialize_from_trajectory:
                if isinstance(init[1], str):
                    if init[1] in self.datasets.names:
                        init[1] = self.datasets[init[1]].data
                getattr(self, estimator).initialize_from_trajectory(*init)
                
        return None
    
    """Simulation"""
    
    def simulate(self):
        """This should try to handle all of the simulation cases"""
    
        # Create the simulator object
        _print('Making simulator')
        self.create_simulator()
        # Add any previous trajectories, if given
        _print('Adding traj')
        self._from_trajectories('simulator')
        # Run the simulation
        _print('Running sim')
        self.run_simulation()
        
        return None
    
    def create_simulator(self):
        """This should try to handle all of the simulation cases"""
        
        _print('Setting up simulator:')
        sim_set_up_options = copy.copy(self.settings.simulator)
        _print(sim_set_up_options)
        dis_method = sim_set_up_options.pop('method', 'dae.collocation')
        
        if self._has_step_or_dosing:
            dis_method = 'fe'
        
        _print(dis_method)
        
        if dis_method == 'fe':
            simulation_class = FESimulator
        else:
            simulation_class = PyomoSimulator
        
        if self.model is None:
            # with_data her
            self.create_pyomo_model(*self.settings.general.simulation_times)
        
        self.s_model = self.model.clone()
        
        if hasattr(self.s_model, self.__var.model_parameter):
            for param in getattr(self.s_model, self.__var.model_parameter).values():
                param.fix()
        
        _print(simulation_class)
        
        simulator = simulation_class(self.s_model)
        _print('After class made')
        simulator.apply_discretization(self.settings.collocation.method,
                                       ncp=self.settings.collocation.ncp,
                                       nfe=self.settings.collocation.nfe,
                                       scheme=self.settings.collocation.scheme)
        
        _print('After disc')
        
        if self._has_step_or_dosing:
            for time in simulator.model.alltime.data():
                getattr(simulator.model, self.__var.dosing_variable)[time, self.__var.dosing_component].set_value(time)
                getattr(simulator.model, self.__var.dosing_variable)[time, self.__var.dosing_component].fix()
        
            #getattr(simulator.model, 'time_step_change')[0].fix()
        
        self.simulator = simulator
        print('Finished creating simulator')
        
        return None
        
    def run_simulation(self):
        """Runs the simulations, may be combined with the above at a later date
        
        """
        if self._has_dosing_points or isinstance(self.simulator, FESimulator):
            self.call_fe_factory()
        
        simulator_options = self.settings.simulator
        simulator_options.pop('method', None)
        self.results = self.simulator.run_sim(**simulator_options)
#        self.results.file_dir = self.settings.general.charts_directory
    
        return None
    
    # def reduce_spectra_data_set(self, dropout=4):
    #     """To reduce the computational burden, this can be used to reduce 
    #     the amount of spectral data used
        
    #     """
    #     A_set = [l for i, l in enumerate(self.model.meas_lambdas) if (i % dropout == 0)]
    #     return A_set
    
    def bound_profile(self, var, bounds):
        """Wrapper for TemplateBuilder bound_profile method"""
        
        self.builder.bound_profile(var=var, bounds=bounds)
        return None
    
    """Estimators"""
    
    def create_variance_estimator(self, **kwargs):
        """This is a wrapper for creating the VarianceEstimator"""
        
        if len(kwargs) == 0:
            kwargs = self.settings.collocation
        
        if self.model is None:    
            self.create_pyomo_model()  
        
        self.create_estimator(estimator='v_estimator', **kwargs)
        self._from_trajectories('v_estimator')
        return None
        
    def create_parameter_estimator(self, **kwargs):
        """This is a wrapper for creating the ParameterEstiamtor"""
        
        if len(kwargs) == 0:
            kwargs = self.settings.collocation
            
        if self.model is None:    
            self.create_pyomo_model()  
            
        self.create_estimator(estimator='p_estimator', **kwargs)
        self._from_trajectories('p_estimator')
        return None
        
    def initialize_from_simulation(self, estimator='p_estimator'):
        
        if not hasattr(self, 's_model'):
            _print('Starting simulation for initialization')
            self.simulate()
            _print('Finished simulation, updating variables...')

        _print(f'The model has the following variables:\n{get_vars(self.s_model)}')
        vars_to_init = get_vars(self.s_model)
        
        _print(f'The vars_to_init: {vars_to_init}')
        for var in vars_to_init:
            if hasattr(self.results, var):    
                _print(f'Updating variable: {var}')
                getattr(self, estimator).initialize_from_trajectory(var, getattr(self.results, var))
            else:
                continue
        
        return None
    
    def create_estimator(self, estimator=None):
        """This function handles creating the Estimator object
        
        Args:
            estimator (str): p_estimator or v_estimator for the PE or VE
            
        Returns:
            None
            
        """        
        if estimator == 'v_estimator':
            Estimator = VarianceEstimator
            est_str = 'VarianceEstimator'
            
        elif estimator == 'p_estimator':
            Estimator = ParameterEstimator
            est_str = 'ParameterEstimator'
            
        else:
            raise ValueError('Keyword argument estimator must be p_estimator or v_estimator.')  
        
        model_to_clone = self.model
        
        setattr(self, f'{estimator[0]}_model', model_to_clone.clone())
        setattr(self, estimator, Estimator(getattr(self, f'{estimator[0]}_model')))
        getattr(self, estimator).apply_discretization(self.settings.collocation.method,
                                                      ncp=self.settings.collocation.ncp,
                                                      nfe=self.settings.collocation.nfe,
                                                      scheme=self.settings.collocation.scheme)
        _print('Starting from_traj')
        self._from_trajectories(estimator)
        
        _print('Starting sim init')
        if self.settings.parameter_estimator.sim_init and estimator == 'p_estimator':
            self.initialize_from_simulation(estimator=estimator)
        
        if self._has_step_or_dosing:
            for time in getattr(self, estimator).model.alltime.data():
                getattr(getattr(self, estimator).model, self.__var.dosing_variable)[time, self.__var.dosing_component].set_value(time)
                getattr(getattr(self, estimator).model, self.__var.dosing_variable)[time, self.__var.dosing_component].fix()
        
        
        return None
    
    # def solve_variance_given_delta(self):
    #     """Wrapper for this VarianceEstimator function"""
    #     variances = self.v_estimator.solve_sigma_given_delta(**self.settings.variance_estimator)
    #     return variances
        
    def run_ve_opt(self):
        """Wrapper for run_opt method in VarianceEstimator"""
        
        if self.settings.variance_estimator.method == 'direct_sigmas':
            worst_case_device_var = self.v_estimator.solve_max_device_variance(**self.settings.variance_estimator)
            self.settings.variance_estimator.device_range = (self.settings.variance_estimator.best_accuracy, worst_case_device_var)
            
        self._run_opt('v_estimator', **self.settings.variance_estimator)
        
        return None
    
    def run_pe_opt(self):
        """Wrapper for run_opt method in ParameterEstimator"""
        if hasattr(self, '_G_data') and self._G_data is not None:
            pe_settings = {**self.settings.parameter_estimator, **self._G_data}
        else:
            pe_settings = {**self.settings.parameter_estimator} #, **self._G_data}, 
            
        self._run_opt('p_estimator', **pe_settings)
        
        return None
    
    def _update_related_settings(self):
        """Checks if conflicting options are present and fixes them accordingly
        
        """
        # Start with what is known
        if self.settings.parameter_estimator['covariance']:
            if self.settings.parameter_estimator['solver'] not in ['k_aug', 'ipopt_sens']:
                raise ValueError('Solver must be k_aug or ipopt_sens for covariance matrix')
        
        # If using sensitivity
        # solvers switch covariance to True
        if self.settings.parameter_estimator['solver'] in ['k_aug', 'ipopt_sens']:
            self.settings.parameter_estimator['covariance'] = True
        
        #Subset of lambdas
        if self.settings.variance_estimator['freq_subset_lambdas'] is not None:
            if isinstance(self.settings.variance_estimator['freq_subset_lambdas'], int):
                self.settings.variance_estimator['subset_lambdas' ] = self.reduce_spectra_data_set(self.settings.variance_estimator['freq_subset_lambdas']) 
        
        if self.settings.general.scale_pe and not self.settings.general.no_user_scaling:
            self.settings.solver.nlp_scaling_method = 'user-scaling'
    
        if self.settings.variance_estimator.max_device_variance:
            self.settings.parameter_estimator.model_variance = False
    
        return None
    
    def fix_parameter(self, param_to_fix):
        """Fixes parameter passed in as a list
        
        Args:
            param_to_fix (list): List of parameter names to fix
        
        Returns:
            None
        
        """
        if not hasattr(self, 'fixed_params'):
            self.fixed_params = []
        
        if isinstance(param_to_fix, str):
            param_to_fix = [param_to_fix]
            
        self.fixed_params += [p for p in param_to_fix]
        
        return None
    
    def run_opt(self):
        """Run ParameterEstimator but checking for variances - this should
        remove the VarianceEstimator being required to be implemented by the user
        
        """
        _print('Starting the RunOpt method')
        
        # Make the model if not present
        if self.model is None:    
            self.create_pyomo_model()  
            _print('Generating model')
        
        # Check if all needed data for optimization available
        if not self.allow_optimization:
            raise ValueError('The model is incomplete for parameter optimization')
            
        # Some settings are required together, this method checks this
        self._update_related_settings()
        
        # Check if all component variances are given; if not run VarianceEstimator
        has_spectral_data = self.spectra is not None
        has_all_variances = self.components.has_all_variances
        variances_with_delta = None
        
        # Check for bad options
        if self.settings.variance_estimator.method == 'direct_sigmas':
            raise ValueError('This variance method is not intended for use in the manner: see Ex_13_direct_sigma_variances.py')
        
        # If not all component variances are provided and spectral data is
        # present, the VE needs to be run
        if not has_all_variances and has_spectral_data:
            """If the data is spectral and not all variances are provided, VE needs to be run"""
            
            # Create the VE
            self.create_estimator(estimator='v_estimator')
            self.settings.variance_estimator.solver_opts = self.settings.solver
            # Optional max device variance
            if self.settings.variance_estimator.max_device_variance:
                max_device_variance = self.v_estimator.solve_max_device_variance(**self.settings.variance_estimator)
            
            # elif self.settings.variance_estimator.use_delta:
            #     variances_with_delta = self.solve_variance_given_delta()

            else:
                self.run_ve_opt()
                
                #print('Finished VE Opt - moving to PE opt')
        # If not a spectral problem and not all variances are provided, they
        # set to 1
        elif not has_all_variances and not has_spectral_data:
            for comp in self.components:
                try:
                    comp.variance = self.variances[comp.name]
                except:
                    comp.variance = 1
                
        # Create ParameterEstimator
        _print('Making PEstimator')
        self.create_estimator(estimator='p_estimator')
        
        variances = self.components.variances
        self.variances = variances
        
        # The VE results can be used to initialize the PE
        if 'v_estimator' in self.results_dict:
            if self.settings.general.initialize_pe:
                # Update PE using VE results
                self.initialize_from_variance_trajectory()
                # No initialization from simulation is needed
                self.settings.parameter_estimator.sim_init = False
 
            if self.settings.general.scale_pe:
                # Scale variables from VE results
                self.scale_variables_from_variance_trajectory()
            
            # Extract vairances
            self.variances = self.results_dict['v_estimator'].sigma_sq
        
        # If using max device variance, the variances are calculated differently
        elif self.settings.variance_estimator.max_device_variance:
            self.variances = max_device_variance
        
        # Under construction
        """ TODO
        # elif variances_with_delta is not None: 
        #     variances = variances_with_delta
        """
        # Optional variance scaling
        if self.settings.general.scale_variances:
            self.variances = self._scale_variances(self.variances)
        
        # Update PE solver settings and variances
        self.settings.parameter_estimator.solver_opts = self.settings.solver
        self.settings.parameter_estimator.variances = self.variances
        
        # Run the PE
        self.run_pe_opt()
        
        # Save results in the results_dict
        self.results = self.results_dict['p_estimator']
        self.results.file_dir = pathlib.Path.cwd() #self.settings.general.charts_directory
        
        # Tells MEE that the individual model is already solved
        self.optimized = True
        
        return self.results
    
    @staticmethod
    def _scale_variances(variances):
        
        max_var = max(variances.values())
        scaled_vars = {comp: var/max_var for comp, var in variances.items()}
        return scaled_vars

    def _run_opt(self, estimator, *args, **kwargs):
        """Runs the respective optimization for the estimator"""
        
        if not hasattr(self, estimator):
            raise AttributeError(f'ReactionModel has no attribute {estimator}')
            
        self.results_dict[estimator] = getattr(self, estimator).run_opt(*args, **kwargs)
        return self.results_dict[estimator]
    
    def initialize_from_variance_trajectory(self, variable=None, obj='p_estimator'):
        """Wrapper for the initialize_from_trajectory method in
        ParameterEstimator
        
        """
        source = self.results_dict['v_estimator']
        self._from_trajectory('initialize', variable, source, obj)
        return None
    
    def initialize_from_trajectory(self, variable_name=None, source=None):
        """Wrapper for the initialize_from_trajectory method in
        ParameterEstimator or PyomoSimulator
        
        """
        self._var_to_initialize_from_trajectory.append([variable_name, source])
        return None
    
    def scale_variables_from_variance_trajectory(self, variable=None, obj='p_estimator'):
        """Wrapper for the scale_varialbes_from_trajectory method in
        ParameterEstimator
        
        """
        source = self.results_dict['v_estimator']
        self._from_trajectory('scale_variables', variable, source, obj)
        return None
        
    @staticmethod
    def _get_source_data(source, var):
        """Get the correct data from a ResultsObject or a DataFrame"""
        
        if isinstance(source, pd.DataFrame):
            return source
        else:
            return getattr(source, var)
    
    def _from_trajectory(self, category, variable, source, obj):
        """Generic initialization/scaling function"""
        
        estimator = getattr(self, obj)
        method = getattr(estimator, f'{category}_from_trajectory')
        
        if variable is None:
            vars_to_init = get_vars(estimator.model)
        
            _print(f'The vars_to_init: {vars_to_init}')
            for var in vars_to_init:
                if hasattr(source, var):    
                    _print(f'Updating variable: {var}')
                    method(var, self._get_source_data(source, var))   
        else:
            method(variable, self._get_source_data(source, variable))
        return None
                                               
    def set_known_absorbing_species(self, *args, **kwargs):
        """Wrapper for set_known_absorbing_species in TemplateBuilder
        
        """
        self.builder.set_known_absorbing_species(*args, **kwargs)    
        return None
    
    def scale(self):
        """Scale the model"""
        
        parameter_dict = self.parameters.as_dict(bounds=False)    
        scaled_parameter_dict, scaled_models_dict = scale_models(self.model,
                                                                 parameter_dict,
                                                                 name=self.name,
                                                                 )         
        return scaled_parameter_dict, scaled_models_dict
    
    def clone(self, *args, **kwargs):
        """Makes a copy of the ReactionModel and removes the data. This is done
        to reuse the model, components, and parameters in an easier manner
        
        """
        new_kipet_model = copy.deepcopy(self)
        
        name = kwargs.get('name', self.name + '_copy')
        copy_model = kwargs.get('model', True)
        copy_builder = kwargs.get('builder', True)
        copy_components = kwargs.get('components', True)   
        copy_parameters = kwargs.get('parameters', True)
        copy_datasets = kwargs.get('datasets', True)
        copy_constants = kwargs.get('constants', True)
        copy_settings = kwargs.get('settings', True)
        copy_algebraic_variables = kwargs.get('alg_vars', True)
        copy_odes = kwargs.get('odes', True)
        copy_algs = kwargs.get('algs', True)
        
        # Reset the datasets
        
        new_kipet_model.name = name
        
        if not copy_model:
            new_kipet_model.model = None
        
        if not copy_builder:
            new_kipet_model.builder = TemplateBuilder()
            
        if not copy_components:
            new_kipet_model.components = ComponentBlock()
        
        if not copy_parameters:
            new_kipet_model.parameters = ParameterBlock()
            
        if not copy_datasets:
            del new_kipet_model.datasets
            new_kipet_model.datasets = DataBlock()
            
        if not copy_constants:
            new_kipet_model.constants = None
            
        if not copy_algebraic_variables:
            new_kipet_model.algebraic_variables = []
            
        if not copy_settings:
            new_kipet_model.settings = Settings()
            
        if not copy_odes:
            new_kipet_model.odes = None
            
        if not copy_algs:
            new_kipet_model.algs = None
        
        list_of_attr_to_delete = ['p_model', 'v_model', 'p_estimator',
                                  'v_estimator', 'simulator']
        
        for attr in list_of_attr_to_delete:
            if hasattr(new_kipet_model, attr):
                setattr(new_kipet_model, attr, None)
        
        new_kipet_model.results_dict = {}
            
        return new_kipet_model
    
    def rhps_method(self,
                     method='k_aug',
                     calc_method='global',
                     scaled=True):
        """This calls the reduce_models method in the EstimationPotential
        module to reduce the model based on the reduced hessian parameter
        selection method.
        
        Args:
            kwargs:
                replace (bool): defaults to True, option to replace the
                    parameters deemed unestimable from the model with constants
                no_scaling (bool): defaults to True, removes the scaling
                    constants from the model and restores the parameter values
                    and their bounds.
                    
        Returns:
            results (ResultsObject): A standard results object with the reduced
                model results
        
        """
        if self.model is None:
            self.create_pyomo_model()
            
        kwargs = {}
        kwargs['solver_opts'] = self.settings.solver
        kwargs['method'] = method
        kwargs['calc_method'] = calc_method
        kwargs['scaled'] = scaled
        kwargs['use_bounds'] = False
        kwargs['use_duals'] = False
        kwargs['ncp'] = self.settings.collocation.ncp
        kwargs['nfe'] = self.settings.collocation.nfe
        
        # parameter_dict = self.parameters.as_dict(bounds=True)
        results, reduced_model = rhps_method(self.model, **kwargs)
        
#        results.file_dir = self.settings.general.charts_directory
        
        self.reduced_model = reduced_model
        self.using_reduced_model = True
        self.reduced_model_results = results
        
        #Make a KipetModel as the result using the reduced model
        red_model = ReactionModel()
        self.red_model = ReactionModel(name=self.name+'_reduced')
                
        assign_list = ['components', 'parameters', 'constants', 'algebraics',
                       'states', 'ub', 'settings', 'c', 'odes_dict']
  
        ignore = []
        for item in assign_list:
            if item not in ignore and hasattr(self, item):
                setattr(self.red_model, item, getattr(self, item))
        
        # reduced_kipet_model = self.clone('reduced_model', **items_not_copied)
        
        #%%
        # reduced_kipet_model.add_parameter()
        # self, name=None, init=None, bounds=None
        
        # reduced_kipet_model.model = reduced_model
        # reduced_kipet_model.results = results
        
        # reduced_parameter_set = {k: [v.value, (v.lb, v.ub)] for k, v in reduced_kipet_model.model.P.items()}
        # for param, param_data in reduced_parameter_set.items():
        #     reduced_kipet_model.add_parameter(param, init=param_data[0], bounds=param_data[1])
        
        # return reduced_kipet_model
    
    def set_non_absorbing_species(self, non_abs_list):
        """Wrapper for set_non_absorbing_species in TemplateBuilder"""
        
        self._has_non_absorbing_species = True
        self.non_abs_list = non_abs_list
        return None
        
    def add_noise_to_data(self, var, noise, overwrite=False):
        """Wrapper for adding noise to data after data has been added to
        the specific ReactionModel
        
        """
        dataframe = self.datasets[var].data
        if overwrite:
            self.datasets[var].data = dataframe
        return data_tools.add_noise_to_signal(dataframe, noise)    
    
    def unwanted_contribution(self, variant, **kwargs):
        
        self._G_contribution = variant
        self._G_data = {'G_contribution': variant,
                        'Z_in': kwargs.get('Z_in', dict()),
                        'St': kwargs.get('St', dict()),
                        }
    
    def analyze_parameters(self, 
                        method=None,
                        parameter_uncertainties=None,
                        meas_uncertainty=None,
                        sigmas=None,
                        ):
        
        """This is a wrapper for the EstimabilityAnalyzer 
        """
        # Here we use the estimability analysis tools
        self.e_analyzer = EstimabilityAnalyzer(self.model)
        # Problem needs to be discretized first
        self.e_analyzer.apply_discretization('dae.collocation',
                                             nfe=60,
                                             ncp=1,
                                             scheme='LAGRANGE-RADAU')
        
        #param_uncertainties = {'k1':0.09,'k2':0.01,'k3':0.02,'k4':0.5}
        # sigmas, as before, represent the variances in regard to component
        #sigmas = {'A':1e-10,'B':1e-10,'C':1e-11, 'D':1e-11,'E':1e-11,'device':3e-9}
        # measurement scaling
        #meas_uncertainty = 0.05
        # The rank_params_yao function ranks parameters from most estimable to least estimable 
        # using the method of Yao (2003). Notice the required arguments. Returns a dictionary of rankings.
        if method == 'yao':
            
            listparams = self.e_analyzer.rank_params_yao(meas_scaling=meas_uncertainty,
                                                         param_scaling=parameter_uncertainties,
                                                         sigmas=sigmas)
            #print(listparams)
            
            # Now we can run the analyzer using the list of ranked parameters
            params_to_select = self.e_analyzer.run_analyzer(method='Wu', 
                                                            parameter_rankings=listparams,
                                                            meas_scaling=meas_uncertainty, 
                                                            variances=sigmas
                                                            )
            # We can then use this information to fix certain parameters and run the parameter estimation
            print(params_to_select)
            
            params_to_fix = list(set(self.parameters.names).difference(params_to_select))
        
        return params_to_select, params_to_fix 
    
    def fix_and_remove_parameters(self, model_name, parameters=None):
        
        if model_name not in ['s_model', 'v_model', 'p_model']:
            raise ValueError(f'ReactionModel does not have model type {model_name}')
        
        model = getattr(self, model_name)
        param_replacer = ParameterReplacer([model], fix_parameters=parameters)
        param_replacer.remove_fixed_vars()
    
        return None

        
    """MODEL FUNCTION AREA"""
    
    def make_reaction_table(self, stoich_coeff, rxns):
        
        df = pd.DataFrame()
        for k, v in stoich_coeff.items():
            df[k] = v
        
        df.index = rxns
        self.reaction_table = df
    
    def reaction_block(self, stoich_coeff, rxns):
        # make a reactions_dict in __init__ and update here
        """Method to allow for simple construction of a reaction system
        
        Args:
            stoich_coeff (dict): dict with components as keys and stoichiometric
                coeffs is lists as values:
                    example: stoich_coeff['A'] = [-1, 0 , 1] etc
                
            rxns (list): list of algebraics that represent reactions
            
        Returns:
            r_dict (dict): dict of completed reaction expressions
            
        """
        self.make_reaction_table(stoich_coeff, rxns)
        
        r_dict = {}
        
        for com in self.components.names: 
            r_dict[com] = sum(stoich_coeff[com][i] * self.ae(r) for i, r in enumerate(rxns)) 
    
        return r_dict
    
    @property
    def models(self):
        
        output = 'ReactionModel has the following:\n'
        output_dict = {}
        
        for model in [name + 'model' for name in ['', 's_', 'v_', 'p_']]:
        
            if hasattr(self, model):
                output += f'{model} True\n'
                output_dict[model] = True
            else:
                output += f'{model} False\n'
                output_dict[model] = False
            
        print(output)
        return output_dict
    
    @property
    def has_objective(self):
        """Check if p_model has an objective"""
        
        return hasattr(self.p_model, 'objective')


    def check_component_units(self):
        """Method to check whether the units provided are consisten with the
        base units
        
        """
        
        print('Checking model component units:\n')
        
        element_dict = {
           'parameters': self.parameters,
           'components': self.components,
           'constants': self.constants,
           'algebraics': self.algebraics,
           'states': self.states,
                }
        
        if not hasattr(self, 'c'):
            self._make_c_dict()
        
        for elem, obj in element_dict.items():
            for comp in obj:
                comp._check_scaling()
                
                if comp.units != comp.units_orig:
                    comp.pyomo_var.parent_component()._units = getattr(pyo_units, str(comp.units.u))
        
        self._units_checked = True
        print('')
        
        return None


    def check_component_units_base(self):
        """Method to check whether the units provided are consisten with the
        base units
        
        """
        element_dict = {
           'parameters': self.parameters,
           'components': self.components,
           'constants': self.constants,
           'algebraics': self.algebraics,
           'states': self.states,
                }
        
        from kipet.model_components.units_handler import convert_single_dimension
        
        if not hasattr(self, 'c'):
            self._make_c_dict()
        
        for elem, obj in element_dict.items():
            if elem in ['components', 'states', 'parameters', 'constants']:
                for key in obj:
                    
                    print(f'Checking units for {key.name}: {key.units}')
                    
                    key_comp = key #self.get_state(key)
                    key_comp_units = key_comp.units
                    key_comp_units = convert_single_dimension(self.unit_base.ur, key_comp_units, self.unit_base.TIME_BASE, power_fixed=False)
                    
                    print(key_comp_units)
                    print(self.unit_base.VOLUME_BASE)
                    print(f'Checking units for {key.name}: {key.units}')
                    key_comp_units = convert_single_dimension(self.unit_base.ur, key_comp_units, self.unit_base.VOLUME_BASE, power_fixed=True)
                    
                    print(key_comp_units)
                    
                    key_comp.units = key_comp_units.units
                    key_comp.value *= key_comp_units.m
                    key_comp.pyomo_var.parent_component()._units = getattr(pyo_units, str(key_comp.units))
        
                    # if self.value is not None:
                    #     self.value = quantity.m*self.value
                        
                    key_comp.conversion_factor = key_comp_units.m
                    # key_comp.units = 1*quantity.units
                    
                    if hasattr(key_comp, 'bounds') and key_comp.bounds is not None:
                        bounds = list(key_comp.bounds)
                        if bounds[0] is not None:
                            bounds[0] *= key_comp.conversion_factor
                        if bounds[1] is not None:
                            bounds[1] *= key_comp.conversion_factor
                        key_comp.bounds = (bounds) 
        
        # for elem, obj in element_dict.items():
        #     for comp in obj:
        #         comp._check_scaling()
                
        #         if comp.units != comp.units_orig:
        #             comp.pyomo_var.parent_component()._units = getattr(pyo_units, str(comp.units.u))
        
        self._units_checked = True
        print('')
        
        return None
    
    def check_model_units(self, orig_units=False, display=False):
        """Method to check the expected units of the algebraic and odes
        
        """
        from kipet.model_components.units_handler import convert_single_dimension
        
        print('Checking expected equation units:\n')
        
        if not self.__flag_odes_built:
            self._build_odes()
        if not self.__flag_algs_built:
            self._build_algs()
        
        elements = ['parameters', 'components', 'states', 'constants', 'algebraics']
        element_dict = {}
        for element in elements:
            if hasattr(self, f'{element}'):
                element_dict[element] = getattr(self, f'{element}')
            else:
                element_dict[element] = {}
                
        if not hasattr(self, 'c'):
            self._make_c_dict()
        
        for key, expr in self.odes_dict.items():
            
            key_comp = self.get_state(key)
            if orig_units:
                convert_to = ' / '.join([str(key_comp.units_orig), self.unit_base.TIME_BASE])    
                key_comp.use_orig_units = True
            else:
                convert_to = ' / '.join([str(key_comp.units), self.unit_base.TIME_BASE])
            
            expr.check_expression_units(convert_to=str(convert_to), scalar=1)
        
            
        for key, expr in self.algs_dict.items():
            
            expr_obj = self.alg_obj.exprs[key]
            
            if expr_obj.expression_orig is not None:
                expr_use = expr_obj.expression_orig
            else:
                expr_use = expr_obj.expression
            
            print(expr_use)
            key_comp = pyo_units.get_units(expr_use)
            expr.units = key_comp
            
        if display:
            self.ode_obj.display_units()
            print('')
            if len(self.alg_obj) > 0:
                self.alg_obj.display_units()
                print('')
            
            
        return None
    
    
    def plot(self, var=None, jupyter=False):
        
        """Plot results using the variable or variable class"""
        
        from kipet.visuals.plots import PlotObject
        
        self._plot_object = PlotObject(reaction_model=self, jupyter=jupyter)
        
        if var == 'Z':
            self._plot_object._plot_all_Z()
            
        if var == 'S':
            self._plot_object._plot_all_S()
            
        elif var in self.components.names:
            self._plot_object._plot_Z(var)           
            
        elif var in self.states.names:
            self._plot_object._plot_X(var)

        elif var in self.algebraics.names:
            self._plot_object._plot_Y(var)            
                    
        elif var is None:
            self._plot_object._plot_all_Z()
            
            if hasattr(self.results, 'S'):
                self._plot_object._plot_all_S()
            
            for var in self.states.names:
                self._plot_object._plot_X(var)
            
            for var in self.algebraics.names:
                self._plot_object._plot_Y(var)  
                
            if hasattr(self.results, 'step'):
                for var in self.results.step:
                    self._plot_object._plot_step(var)
                
        return None
        

    def ae(self, alg_var):
        return self.algs_dict[alg_var].expression
    
    def ode(self, ode_var):
        return self.odes_dict[ode_var].expression

    def get_state(self, comp):
        if comp in self.components:
            return self.components[comp]
        elif comp in self.states:
            return self.states[comp]
 
    def get_alg(self, comp):
        if comp in self.algebraics:
            return self.algebraics[comp]
        
    
    
def _set_directory(model_object, filename, abs_dir=False):
    """Wrapper for the set_directory method. This replaces the awkward way
    of ensuring the correct directory for the data is used.
    
    Args:
        filename (str): the file name to be formatted
        
    Returns:
        file_path (pathlib Path): The absolute path of the given file
    """
    #directory = model_object.settings.general.data_directory
    directory = pathlib.Path.cwd()
    file_path = pathlib.Path(directory).joinpath(filename)
    
    return file_path