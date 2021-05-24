# Standard library imports
#import collections
import copy
import os
import pathlib
import sys

# Third party imports
import numpy as np
import pandas as pd
#from pyomo.core.base.var import IndexedVar
#from pyomo.dae.diffvar import DerivativeVar
from pyomo.environ import ConcreteModel, Set, Var
from pyomo.environ import units as pyo_units

# Kipet library imports
import kipet.kipet_io as io
from kipet.common.interpolation import interpolate_trajectory
#from kipet.common.model_funs import step_fun
from kipet.core_methods.EstimabilityAnalyzer import EstimabilityAnalyzer
from kipet.core_methods.EstimationPotential import (
    replace_non_estimable_parameters, rhps_method)
from kipet.core_methods.FESimulator import FESimulator
from kipet.core_methods.ParameterEstimator import ParameterEstimator
from kipet.core_methods.PyomoSimulator import PyomoSimulator
from kipet.core_methods.TemplateBuilder import TemplateBuilder
from kipet.core_methods.VarianceEstimator import VarianceEstimator
from kipet.dev_tools.display import Print
from kipet.mixins.TopLevelMixins import WavelengthSelectionMixins
from kipet.model_components.units_handler import convert_single_dimension
from kipet.post_model_build.pyomo_model_tools import get_vars
from kipet.post_model_build.replacement import ParameterReplacer
from kipet.post_model_build.scaling import scale_models
from kipet.top_level.data_component import DataBlock, DataSet
from kipet.top_level.element_blocks import (AlgebraicBlock, ComponentBlock,
                                            ConstantBlock, ParameterBlock,
                                            StateBlock)
from kipet.top_level.expression import (AEExpressions, Expression,
                                        ODEExpressions)
from kipet.top_level.helper import DosingPoint
from kipet.top_level.settings import Settings
from kipet.top_level.spectral_handler import SpectralData
from kipet.top_level.variable_names import VariableNames

__var = VariableNames()
DEBUG=True #__var.DEBUG
_print = Print(verbose=DEBUG)
model_components = ['parameters', 'components', 'states', 'algebraics', 'constants', 'steps']
version_number = '0.2.2'


# This should be removed - no mixins!
class ReactionModel(WavelengthSelectionMixins):
    
    """This is the primary class used to organize the KIPET package

        KipetModel is the top-level obejct invoked when using KIPET. It contains an array
        of ReactionModels as well as several functions generally useful at the global level.
        
    
    User defined attributes
    
    :var name: The name given to the ReactionModel
    :var unit_base: The base units used in the model (provided by KipetModel)
    
    Public attributes
    
    :var spectra: SpectraData object if available, otherwise None
    :var components: ComponentBlock object
    :var parameters: ParameterBlock object
    :var constants: ConstantBlock object
    :var algebraics: AlgebraicBlock object
    :var states: StateBlock object
    :var datasets: DataBlock object
    :var results_dict: dictionary of ResultsObject instances for simulation, variance, parameter estimation models
    :var settings: Settings instance
    :var variances: dicitonary of component variances
    :var odes: ODEExpressions object
    :var algs: AEExpressions object
    :var odes_dict: dictionary of odes
    :var algs_dict: dictionary of algebraic equations
    
    Private attributes
    
    :var _model: The base Pyomo model to be created
    :var _s_model: The Pyomo model used in simulation
    :var _builder: TemplateBuilder instance
    :var _template_populated: Bool designating whether the builder object is built
    :var __flag_odes_built: Bool indicating whether the odes are built
    :var __flag_algs_built: Bool indicating whether the daes are built
    :var _custom_objective: Algebraic variable to use in the custom objective term
    :var _optimized: Bool indicating whether the ReactionModel has been optimized
    :var _dosing_points: Dictionary with optional dosing points 
    :var _has_dosing_points: Bool indicating if dosing points are used
    :var _has_step_or_dosing: Bool indicating if dosing or step variables are used
    :var _has_non_absorbing_species: Bool indicating if non-absorbing species are present
    :var _var_to_fix_from_trajectory: List of variables with fixed trajectories
    :var _var_to_initialize_from_trajectory: List of variables to initialize from trajectories
    :var _default_time_unit: Default unit of time
    :var _allow_optimization: Bool indicating if prerequisite data meets requirements for parameter fitting
    :var _G_data: Dictionary containing unwanted contribution data
    :var __var: VariableNames object containing global parameter and variable names in the Pyomo models
      
    :Methods:
    
    - :func:`add_reaction`
    - :func:`add_reaction_list`
    - :func:`remove_reaction`
    - :func:`new_reaction`
    - :func:`run_opt`
    - :func:`read_data_file`
    - :func:`write_data_file`
    - :func:`add_noise_to_data`
    
    :Properties:
    
    - :func:`all_params`
    
    :Example:
        >>> import kipet
        >>> kipet_model = kipet.KipetModel()
    
    """
    __var = VariableNames()
    
    
    def __init__(self, name=None, unit_base=None, model=None):
        
        """
        Initialization of ReactionModel instance.
        
        This is the most important object in describing, building, and solving
        a reaction model in KIPET. This object contains the information describing
        the species, the initial conditions, the ODEs and DAEs, as well as any 
        spectral data preprocessing.
        
        :var name: The name given to the ReactionModel
        :var unit_base: The base units used in the model (provided by KipetModel)
        
        """
        
        # Variables initialized by user or KipetModel
        self.name = name if name is not None else 'Model-1'
        
        if unit_base is not None:
            self.unit_base = unit_base
        else:
            from kipet.top_level.unit_base import UnitBase
            self.unit_base = UnitBase()
            
        # This is used directly by the user for modifying spectral data
        self.spectra = None
        
        # These are left as public attributes but not initialized by user
        self.components = ComponentBlock()   
        self.parameters = ParameterBlock()
        self.constants = ConstantBlock()
        self.algebraics = AlgebraicBlock()
        self.states = StateBlock()
        self.datasets = DataBlock()
        self.results_dict = {}
        self.settings = Settings(category='model')
        self.variances = {}
        self.odes = ODEExpressions()
        self.algs = AEExpressions()
        self.odes_dict = {}
        self.algs_dict = {}
        
        # Private attributes (may be changed later)
        self._model = None
        self._s_model = None
        self._builder = TemplateBuilder()
        self._template_populated = False
        self.__flag_odes_built = False
        self.__flag_algs_built = False
        self._custom_objective = None
        self._optimized = False
        self._dosing_points = None
        self._has_dosing_points = False
        self._has_step_or_dosing = False
        self._has_non_absorbing_species = False
        self._var_to_fix_from_trajectory = []
        self._var_to_initialize_from_trajectory = []
        self._default_time_unit = 'seconds'
        self._allow_optimization = False
        self._G_data = {'G_contribution': None, 'Z_in': dict(), 'St': dict()}
        self._step_list = dict()
        self.__custom_volume_state = False
        self.__volume_terms_added = False
        self.__var = VariableNames()
        
        if model is not None:
            self._copy_from_model(model)

        
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
    
    def _copy_from_model(self, model):
        """This method copies various components from an existing Reaction
        Model and sets the current model's attributes equal to these.
        
        :param ReactionModel model: The existing model from which to initialize
        
        :return: None
        
        """
        if isinstance(model, ReactionModel):
        
            assign_list = [
                "components",
                "parameters",
                "constants",
                "algebraics",
                "states",
                "unit_base",
                "settings",
                "c",
                "odes_dict",
                "algs_dict",
            ]

            for item in assign_list:
                if hasattr(model, item):
                    setattr(self, item, getattr(model, item))

        else:
            raise ValueError("KipetModel can only add ReactionModel instances.")

        return None
    
    """Model Components"""
    
    def _make_set_up_model(self):
        """Make the dummy model for initial pyomo vars
        
        """
        self._set_up_model = ConcreteModel()
        self._set_up_model.indx = Set(initialize=[0])
        return None
    
    
    def _component(self, name, index, units):
        """Creates the initial pyomo variables for model building
        
        :param str name: The name of the component
        :param int index: The number of indicies (ex. constant=1, time-series=2, ...)
        :param str units: The component units
        
        :return: The Pyomo variable representing the component and it\'s units
        :rtype: Tuple with two components (pyomo.core.base.var._GeneralVarData, pint.quantity.build_quantity_class.<locals>.Quantity)
        
        Raises:
            ValueError('A component with this name has already been defined')
        
        """
        if not hasattr(self, '_set_up_model'):
            self._make_set_up_model()

        if hasattr(self._set_up_model, name) and name not in self._step_list:        
            raise ValueError('A component with this name has already been defined')
        
        setattr(self._set_up_model, f'{name}_indx', Set(initialize=[0]))
        sets = [getattr(self._set_up_model, f'{name}_indx')]*index

        if units is None:
            p_units = str(1)
            comp_units = 1*self.unit_base.ur('')

        else:
            con = 1
            margin = 6
            # print(f'Checking units for {name.rjust(margin)}: \t{units}')
            comp_units = convert_single_dimension(self.unit_base.ur, units, self.unit_base.time)
            con *= comp_units.m
            comp_units = convert_single_dimension(self.unit_base.ur, str(comp_units.units), self.unit_base.volume)
            con *= comp_units.m
            p_units = con*getattr(pyo_units, str(comp_units.units))
            comp_units = con*comp_units.units
            if not self.unit_base.ur(units) == comp_units:
                print(f'    Converted to........: \t{comp_units.units}')
            
        setattr(self._set_up_model, name, Var(*sets, initialize=1, units=p_units))
        var = getattr(self._set_up_model, name)
        return var[tuple([0 for i in range(index)])], comp_units
    
    
    def _add_model_component(self, comp_name, index, model_var, name, **kwargs):
        """Generic method for adding modeling components to the ReactionModel
        
        :param str name: The name of the component
        :param int index: The number of indicies tied to the component
        :param str model_var: The name of the variable used in the Pyomo models
        :param dict kwargs: The dictionary of keyword arguments (different components have differing args)
            
        :return: The representative Pyomo variable that can be used in expression building
        :rtype: pyomo.core.base.var._GeneralVarData
        
        """
        units = kwargs.get('units', None)
        value = kwargs.get('value', 1)
        kwargs['units_orig'] = units
        
        par, con = self._component(name, index, units=units)
        
        kwargs['value'] = value*con.m
        kwargs['units'] = str(con.units)
        kwargs['conversion_factor'] = con.m
        
        if 'bounds' in kwargs:
            
            bounds = list(kwargs.get('bounds', [0, 0]))
            if bounds[0] is not None:
                bounds[0] *= con.m
            if bounds[1] is not None:
                bounds[1] *= con.m
            kwargs['bounds'] = (bounds) 
        
        kwargs['unit_base'] = self.unit_base
        kwargs['pyomo_var'] = par
        kwargs['model_var'] = model_var
        getattr(self, f'{comp_name}s').add_element(name, **kwargs)
        
        return par
    
    
    def parameter(self, name, **kwargs):
        """Create a parameter with a localized pyomo var
        
        Parameters are defined as those values to be fit in the kinetic models.
        
        :param str name: The name of the parameter
        
        **Keyword Arguments**
        
        :param float value: Initial value of the parameter
        :param str units: (Optional) Sets the parameter units
        :param tuple(float) bounds: (Optional) Provide parameter bounds
        :param bool fixed: (Optional) Indicates a fixed parameter or not
        :param float variance: (Optional) Provide the parameter's variance
        :param str description: (Optional) Detailed description of the parameter
        
        :return: A representative Pyomo variable that can be used in expression building
        :rtype: pyomo.core.base.var._GeneralVarData
        
        """
        if name == self.__var.volume_name:
            raise AttributeError(f'{self.__volume_name} is a protected state name')
            
        return self._add_model_component('parameter',
                                        1, 
                                        self.__var.model_parameter, 
                                        name,
                                        **kwargs)
    
    
    def component(self, name, **kwargs):
        """Create a component with a localized pyomo var
        
        KIPET considers components to be explicitly those species that can be 
        defined by concentration. Note that this can only be used for components
        that are defined by two indicies (component and time, usually).
        
        :param str name: The name of the component
        
        **Keyword Arguments**
        
        :param float value: Initial value of the parameter
        :param str units: (Optional) Sets the parameter units
        :param tuple(float) bounds: (Optional) Provide parameter bounds
        :param float variance: (Optional) Provide the parameter's variance
        :param str description: (Optional) Detailed description of the parameter
        :param bool known: Indicates whether the initial value is known
        :param bool absorbing: Indicates whether the component absorbs
        :param bool inert: Indicates whether the species reacts
        :param S: Pure component absorption spectra, if available
        :type S: pandas.DataFrame
        
        :return: A representative Pyomo variable that can be used in expression building
        :rtype: pyomo.core.base.var._GeneralVarData
        
        """
        if name == self.__var.volume_name:
            raise AttributeError(f'{self.__volume_name} is a protected state name')
            
        self.add_ode(name, 0)
        return self._add_model_component('component', 
                                        2, 
                                        self.__var.concentration_model, 
                                        name,
                                        **kwargs)
    
    
    def state(self, name, **kwargs):
        """Create a state with a localized pyomo var
        
        KIPET considers states to be complementary states that are not 
        defined by concentration. Note that this can only be used for states
        that are defined by two indicies (state and time, usually).
        
        .. note::
            
            If you are attempting to enter in a custom volume state, see the
            volume method for a more convenient method.
        
        :param str name: The name of the state
        
        **Keyword Arguments**
        
        :param float value: Initial value of the parameter
        :param str units: (Optional) Sets the parameter units
        :param tuple(float) bounds: (Optional) Provide parameter bounds
        :param float variance: (Optional) Provide the parameter's variance
        :param str description: (Optional) Detailed description of the parameter
        :param bool known: Indicates whether the initial value is known
        :param bool is_volume: Indicates if this state is the volume state
        
        :return: A representative Pyomo variable that can be used in expression building
        :rtype: pyomo.core.base.var._GeneralVarData
        
        """
        is_volume = kwargs.pop('is_volume', False)
        if is_volume:
            if name != self.__var.volume_name:
                print(f'Changing name of volume state from {name} to {self.__var.volume_name}')
                name = self.__var.volume_name
            self.__custom_volume_state = True
            
        if name == self.__var.volume_name and not is_volume:
            raise AttributeError('V is a protected state name - change the \
                                 VariableNames class if another name is needed')
            
        self.add_ode(name, 0)
        return self._add_model_component('state', 
                                        2, 
                                        self.__var.state_model, 
                                        name,
                                        **kwargs)
    
    def volume(self, **kwargs):
        """Create a volume state with a localized pyomo var
        
        This is a convenience method to change the default volume settings.
        
        KIPET considers states to be complementary states that are not 
        defined by concentration. Note that this can only be used for states
        that are defined by two indicies (state and time, usually).
        
        The name is the default volume name, usually V.
        
        **Keyword Arguments**
        
        :param float value: Initial value of the parameter
        :param str units: (Optional) Sets the parameter units
        :param tuple(float) bounds: (Optional) Provide parameter bounds
        :param float variance: (Optional) Provide the parameter's variance
        :param str description: (Optional) Detailed description of the parameter
        :param bool known: Indicates whether the initial value is known
        :param bool is_volume: Indicates if this state is the volume state
        
        :return: A representative Pyomo variable that can be used in expression building
        :rtype: pyomo.core.base.var._GeneralVarData
        
        .. note::
            
            A volume state with name 'V' will be generated automatically. If
            you want to use your own naming conventions, pass the is_volume
            argument as True for your volume state.
        
        """
        self.__custom_volume_state = True
        name = self.__var.volume_name
        self.add_ode(name, 0)
        return self._add_model_component('state', 
                                        2, 
                                        self.__var.state_model, 
                                        name,
                                        **kwargs)
    
    
    def constant(self, name, **kwargs):
        """Create a model constant with a localized pyomo var
        
        This allows the user to use model constants that have units. This helps
        ensure that the models have consistent units.
        
        :param str name: The name of the constant
        
        **Keyword Arguments**
        
        :param float value: Initial value of the parameter
        :param str units: (Optional) Sets the parameter units
        :param str description: (Optional) Detailed description of the parameter
        
        :return: A representative Pyomo variable that can be used in expression building
        :rtype: pyomo.core.base.var._GeneralVarData
        
        """
        if name == self.__var.volume_name:
            raise AttributeError(f'{self.__volume_name} is a protected state name')
            
        return self._add_model_component('constant', 
                                        1, 
                                        self.__var.model_constant, 
                                        name,
                                        **kwargs)
    
    
    def fixed_state(self, name, **kwargs):
        """Create a algebraic variable for fixed states
        
        This allows for certain states to be fixed during simulation or
        parameter fitting.
        
        .. note::
            This requires the data keyword argument!
        
        :param str name: The name of the fixed state
        :param float value: Initial value of the parameter
        :param str units: (Optional) Sets the parameter units
        :param str description: (Optional) Detailed description of the parameter        
        :param data: The data used to fix the state
        :type data: pandas.DataFrame

        :return: A representative Pyomo variable that can be used in expression building
        :rtype: pyomo.core.base.var._GeneralVarData
        
        Raises:
            ValueError: A fixed state requires data in order to fix a trajectory
        
        """
        if name == self.__var.volume_name:
            raise AttributeError(f'{self.__volume_name} is a protected state name')
            
        if 'data' not in kwargs:
            raise ValueError('A fixed state requires data in order to fix a trajectory')
        else:
            return self._add_model_component('algebraic', 
                                             2, 
                                             self.__var.algebraic, 
                                             name, 
                                             **kwargs)
        
        
    def algebraic(self, name, **kwargs):
       
        """Create a algebraic variable for fixed states
        
        :param dict kwargs: The dictionary of keyword arguments for algebraic variables representing
          fixed states (note: takes only those with two indicies)
          see ModelComponents for more information. 
        
        :return: The representative Pyomo variable that can be used in expression building
        :rtype: pyomo.core.base.var._GeneralVarData 
        """
        if name == self.__var.volume_name:
            raise AttributeError(f'{self.__volume_name} is a protected state name')
            
        return self._add_model_component('algebraic', 
                                        2, 
                                        self.__var.algebraic, 
                                        name,
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
    
    def step(self, name, **kwargs):
        """Create a step variable with a localized pyomo var
        
        This is used for modeling additions, reaction starts, and any other feature
        that may require a step function to model the behavior.
        
        :param str name: The name used for the step variable
        
        **Keyword Arguments**
        
        :param float coeff: Coefficient for the step size (leave at default of 1)
        :param float time: The time for the step change (init if not fixed)
        :param bool fixed: Choose if the time is known and fixed or variable
        :param bool switch: True if turning on, False if turning off, optionally as string ('on', 'off') args too
        :param float M: Sigmoidal tuning parameter
        :param float eta: Tuning parameter
        :param float constant: Constant added to bring step function to certain value
        :param tuple(float) bounds: Optional bounds for the timing of the step     
        
        :return: The representative Pyomo variable that can be used in expression building
        :rtype: pyomo.core.base.var._GeneralVarData
        """
        if name == self.__var.volume_name:
            raise AttributeError(f'{self.__volume_name} is a protected state name')
            
        self._has_step_or_dosing = True
        if not hasattr(self, '_step_list'):
            self._step_list = {}
            
        var_name = 'step'
        if name not in self._step_list:
            self._step_list[name] = [kwargs]
        else:
            self._step_list[name].append(kwargs)
            
        par, _ = self._component(name, 2, None)
        if not hasattr(self, f'{var_name}s'):
            setattr(self, f'{var_name}s', {})
        getattr(self, f'{var_name}s')[name] = [self.__var.step_variable, par]
        return par
    
    """Dosing profiles"""
    
    def add_dosing_point(self, component, time, conc, vol):
        """Add a dosing point for a component at a specific time with a given amount.
        
        :param str component: The name of the component being dosed
        :param float time: The time when the dosing occurs
        :param float relative: The strength (density, concentration)
        :param str absolute: The amount (g, vol)
        
        :return: None
        
        """
        conversion_dict = {'state': self.__var.state_model, 
                           'concentration': self.__var.concentration_model,
                           }
        if component not in self.components.names:
            raise ValueError('Invalid component name')
            
        # Unit changes here:
        c1 = conc[0]*self.unit_base.ur(conc[1])
        c2 = 1*self.unit_base.ur(self.unit_base.concentration)
        v1 = vol[0]*self.unit_base.ur(vol[1])
        v2 = 1*self.unit_base.ur(self.unit_base.volume)

        conc_converted = c1.to(c2)
        conc_ = (conc_converted.m, str(conc_converted.u))
        vol_converted = v1.to(v2)
        vol_ = (vol_converted.m, str(vol_converted.u))
        
        dosing_point = DosingPoint(component, time, conc_, vol_)
        model_var = conversion_dict[self.components[component].state]
        if self._dosing_points is None:
            self._dosing_points = {}
        if model_var not in self._dosing_points.keys():
            self._dosing_points[model_var] = [dosing_point]
        else:
            self._dosing_points[model_var].append(dosing_point)
        self._has_dosing_points = True
        self._has_step_or_dosing = True
        
        return None
    
    def _call_fe_factory(self):
        """A wrapper for this simulator method, but better"""

        self.simulator.call_fe_factory(
            {
                self.__var.dosing_variable: [self.__var.dosing_component],
            },
            self._dosing_points
        )

        return None
    
    """Model data"""
    
    def add_data(self, *args, **kwargs):
        """This method allows the user to add experimental data to the ReactionModel
        
        .. note::
          The data needs to be in the proper format before being used. See the examples for
          more information. KIPET will automatically identify the column names and match
          the data with the corresponding component, state, or trajectory. This does not
          need to be done by the user.
          
        Data is stored under the datasets attribute. Spectral data is handled separately under
        the spectra attribute, which is a SpectraHandler object that has various preprocessing
        methods.
          
        :param str name: The name of the model component
        
        **Keyword Arguments**
        
        :param str time_scale: The units of time of the measured data
        :param str file: The file name of the data - if given, KIPET handles data automatically
        :param data: The dataframe of the data, if not using the file directly
        :type data: pandas.DataFrame
        :param str category: Description of the data (only needed for spectral data at the moment)
        :param bool remove_negatives: Set all negative data values to zero
        
        :return: None
    
        """        
        name = kwargs.get('name', None)
        time_scale = kwargs.get('time_scale', self.unit_base.time)
        
        time_conversion = 1
        if time_scale != self.unit_base.time:
            time_conversion = self.unit_base.ur(time_scale).to(self.unit_base.time).m
        
        if len(args) > 0:
            name = args[0]
        if name is None:
            name = f'ds-{len(self.datasets) + 1}'
            
        filename = kwargs.get('file', None)
        data = kwargs.pop('data', None)
        category = kwargs.pop('category', None)
        remove_negatives = kwargs.get('remove_negatives', False)
    
        if filename is not None:
            calling_file_name = os.path.dirname(os.path.realpath(sys.argv[0]))
            filename = pathlib.Path(calling_file_name).joinpath(filename)
            kwargs['file'] = filename
            dataframe = io.read_file(filename)
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
    
    
    def _check_data_matches(self, name, name_is_from_model=False):
        """Easy data mapping to ElementBlocks
        
        This method looks through the provided data sets and matches the columns
        to their respective components. This works in two ways, either before or
        after components have been defined.
        
        :param str name: the component name
        :param bool name_is_from_model: Bool to indicate whether the component has already
          been added to the model or not. Determines method for linking the data.
          
        :return: None
    
        """
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
        

        else:
            for block in blocks:
                if name in getattr(self, block).names:
                    for dataset in self.datasets:
                        for col in dataset.data.columns:
                            if col == name:
                                setattr(getattr(self, block)[name], 'data_link', dataset.name)

        return None

    """Template building"""

    def set_time(self, time_span=None):
        """Add times to model for simulation
        
        If performing simulations, the start and end times need to be provided.
        Naturally the start time should be zero and this will be hard-coded in the future.
        
        If you have provided data, the times will be determined automatically from the datasets.
        If the times are set using this methods, they will be used instead.
        
        :param float start_time: The initial time for the simulation (should be zero)
        :param float end_time: The desire time to terminate the simuation.
        
        :return: None
        
        """
        if time_span is None:
            raise ValueError('Time span needs to be a number')
        
        self.settings.general.simulation_times = (0, time_span)
        return None
    
    
    def _unwanted_G_initialization(self, *args, **kwargs):
        """Prepare the ParameterEstimator model for unwanted G contributions
        
        """
        self._builder.add_qr_bounds_init(bounds=(0,None),init=1.1)
        self._builder.add_g_bounds_init(bounds=(0,None))
        
        return None
    
    """ Expressions """
    
    def add_reaction(self, name, expr, description=None):
        """Wrapper to add reactions explicitly without using is_reaction
        
        This adds an expression to the model that is identified as being a 
        reaction (an expression used to describe changes in the components over
        time). This was done to simplify the API.
        
        :param str name: The name used to identify the reaction
        :param expr: The expression representing the reaction
        :type expr: Pyomo expression
        :param str description: An optional description of the expression
        
        :return: Returns a Pyomo variable representing the expression such that it can be used in model building
        :rtype: Pyomo Expression
        """
        return self.add_expression(name, expr, description=description, is_reaction=True)
        
    
    def add_expression(self, name, expr, **kwargs):
        """Adds an expression to the model (DAE, custtom objectives, fixed trajectories)
        
        All of the auxilliary functions such as DAEs, custom objectives, fixed trajectories, additions, etc. are
        added to the ReactionModel instance here.
        
        :param str name: The name used to identify the reaction
        :param expr: The expression representing the reaction
        :type expr: Pyomo expression
        
        **Keyword arguments**
        
        :param str description: An optional description of the expression
        :param bool is_reaction: Indicates whehter the expression is a reaction (see :func:`add_reaction`)
        
        :return: Returns a Pyomo variable representing the expression such that it can be used in model building
        :rtype: Pyomo Expression
        """        
        # Adds algebraics anyways, for comparison purposes
        self._add_model_component('algebraic', 
                                  2, 
                                  self.__var.algebraic, 
                                  name, 
                                  **kwargs)
        
        expr_ = Expression(name, expr)
        expr_.check_division()
        self.algs_dict.update(**{name: expr_})
        
        return expr
        
    
    def add_ode(self, ode_var, expr):
        """Method to add an ode expression to the ReactionModel
        
        :param str ode_var: Component or state variable
        :param expr: Expression for rate equation as a Pyomo expression
        :type expr: Pyomo expression    
        
        :return: None
        
        """
        expr = Expression(ode_var, expr)
        self.odes_dict.update(**{ode_var: expr})
        
        return expr.expression
    
    
    def add_odes(self, ode_fun):
        """Takes in a dict of ODE expressions and sends them to add_ode
        
        Use this if you compose your ODEs as a dictionary before adding them
        to the model.
        
        :param dict ode_fun: Dictionay of ODE expressions

        :Example:
             >>> rates = {}
             >>> rates['A'] = -k1 * A
             >>> rates['B'] = k1 * A - k2 * B
             >>> rates['C'] = k2 * B
             >>> r1.add_odes(rates)
        
        :return: None
            
        """
        if isinstance(ode_fun, dict):
            for key, value in ode_fun.items():
                self.add_ode(key, value)
        
        return None


    def _build_odes(self):
        """Builds the ODEs by passing them to the ODEExpressions class
        
        """
        # Add ODEs to be built
        odes = ODEExpressions(self.odes_dict)
        setattr(self, 'ode_obj', odes)
        self.odes = odes.exprs
        self.__flag_odes_built = True
        
        return None
        
    
    def _build_algs(self):
        """Builds the algebraics by passing them to the AEExpressions class
        
        """ 
        algs = AEExpressions(self.algs_dict)
        setattr(self, 'alg_obj', algs)
        self.algs = algs.exprs
        self.__flag_algs_built = True

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

      
    def add_objective_from_algebraic(self, algebraic_var):
        """Declare an algebraic variable that is to be used in the objective
        
        ..note:: 
            This only handles a single custom objective!
        
        :param str algebraic_var: Variable representing the expression to be added
                to the objective (see Example 17 to see how this is used)
                
        :return: None
            
        """
        self._check_data_matches(algebraic_var, name_is_from_model=True)
        self._custom_objective = algebraic_var
        
        return None
    
    
    def make_model(self):
        """Method to generate the base model for estimability analysis.
        
        This creates a base model from the ReactionModel object.
        
        .. warning::
            This is currently a fix to make some of the estimability methods work.
            They are outdated and have not been updated to the new format completely.
            Thus, this method is not meant to be used normally.
            
        :return: A base Pyomo model
        :rtype: ConcreteModel
        
        """
        if not hasattr(self, '_model') or self._model is None:
            model_instance = self._create_pyomo_model()
            
        return model_instance
    
    
    def _populate_template(self, *args, **kwargs):
        """Method handling all of the preparation for the TB
        
        This collects all of the options and data needed to generate the model
        and passes them in the correct manner to the TemplateBuilder
        
        """
        
        if not self._template_populated:
        
            with_data = kwargs.get('with_data', True)
    
            # Add a volume state if not already done so
            if not self.__custom_volume_state:
                self.volume(value=1, units=self.unit_base.volume)
            
            # Check if adding volume terms to ODEs is True
            if self.settings.general.add_volume_terms:
                if not self.__flag_odes_built:
                    self.add_volume_terms()
                                
            if len(self.states) > 0:
                self._builder.add_model_element(self.states)
            
            if len(self.algebraics) > 0:
                self._builder.add_model_element(self.algebraics)
            
            if len(self.components) > 0:
                self._builder.add_model_element(self.components)
            else:
                raise ValueError('The model has no components')
                
            if len(self.parameters) > 0:
                self._builder.add_model_element(self.parameters)
            else:
                self._allow_optimization = False   
            
            if len(self.constants) > 0:
                self._builder.add_model_element(self.constants)
            
            if with_data:
                if len(self.datasets) > 0 or self.spectra is not None:
                    self._builder.input_data(self.datasets, self.spectra)
                    self._allow_optimization = True
                elif len(self.datasets) == 0 and self.spectra is None:
                    self._allow_optimization = False
                else:
                    pass
                
            # Add the ODEs
            if len(self.odes_dict) != 0:
                self._build_odes()
                self._builder.set_odes_rule(self.odes)
            elif 'odes_bypass' in kwargs and kwargs['odes_bypass']:
                pass
            else:
                raise ValueError('The model requires a set of ODEs')
                
            if len(self.algs_dict) != 0:
                self._build_algs()
                if isinstance(self.algs, dict):
                    self._builder.set_algebraics_rule(self.algs, asdict=True)
                else:
                    self._builder.set_algebraics_rule(self.algs)
                
            if hasattr(self, '_custom_objective') and self._custom_objective is not None:
                self._builder.set_objective_rule(self._custom_objective)
            
            self._builder.set_parameter_scaling(self.settings.general.scale_parameters)
            self._builder.add_state_variance(self.components.variances)
            
            # if self._has_step_or_dosing:
            #     self._builder.add_dosing_var(self._number_of_steps)
            
            if self._has_dosing_points:
                self._add_feed_times()
                
            if hasattr(self, '_step_list') and len(self._step_list) > 0:
                self._builder.add_step_vars(self._step_list)
                
            # It seems this is repetitive - refactor
            if hasattr(self, '_G_contribution') and self._G_contribution is not None:
            #if self.settings.parameter_estimator.G_contribution is not None:
                self._unwanted_G_initialization()
                self._builder._G_contribution = self._G_contribution
            else:
                self._builder._G_contribution = None
            
            self._template_populated = True
        
        else:
            print('Template already populated')
            
        return None
           
            
    def _get_model_times(self):
        """Gathers the model start and end times
        
        If the start and end time are not provivded, this will take the times
        from the settings object.
        
        ..warning::
            This may not be necessary and may be removed in a future version.
        
        """
        start_time, end_time = None, None
        if self.settings.general.simulation_times is not None:
            start_time, end_time = self.settings.general.simulation_times

        return start_time, end_time
    
    
    def _make_c_dict(self):
        """This makes the dummy model using the components of the model for 
        units testing
        """
        model_components = ['parameters', 'components', 'states', 'algebraics', 'constants']
        
        if not hasattr(self, 'c'):
            self.c = dict()
            for mc in model_components: 
                if hasattr(self, f'{mc}') and len(getattr(self, f'{mc}')) > 0:
                    for comp in getattr(self, f'{mc}'):
                        self.c.update({comp.name: [comp.model_var, comp.pyomo_var]})
    
            if hasattr(self, 'steps'):
                self.c.update(getattr(self, 'steps'))
    
    
    def _create_pyomo_model(self, *args, **kwargs):
        """Adds the component, parameter, data, and odes to the TemplateBuilder
        instance and creates the specified model.
        
        :returns: The finished Pyomo model

        """
        kwargs['is_simulation'] = kwargs.get('is_simulation', False)
        skip_non_abs = kwargs.pop('skip_non_abs', False)
         
        self._populate_template(*args, **kwargs)
        self._make_c_dict()
        setattr(self._builder, 'c_mod', self.c)
        
        start_time, end_time = self._get_model_times()
        if self._model is None:
            self._model = self._builder.create_pyomo_model(start_time, end_time, False)
        pyomo_model = self._builder.create_pyomo_model(start_time, end_time, kwargs['is_simulation'])
        
        if not kwargs['is_simulation']:
            non_abs_comp = self.components.get_match('absorbing', False)
            if not skip_non_abs and len(non_abs_comp) > 0:
                self._builder.set_non_absorbing_species(pyomo_model, non_abs_comp, check=True)    
            if hasattr(self,'fixed_params') and len(self.fixed_params) > 0:
                for param in self.fixed_params:
                    getattr(pyomo_model, self.__var.model_parameter)[param].fix()
        
        return pyomo_model
    
    
    def _add_feed_times(self):
        """If there are specific feed times for dosed components, these are 
        added to the builder using this wrapper method
        """
        feed_times = set()
        
        for model_var, dp in self._dosing_points.items():
            for point in dp:
                feed_times.add(point.time)
        
        self._builder.add_feed_times(list(feed_times))
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
        """This method will simulate the model using the given initial values
        and times
        
        This will create the simulation model, perform the simulation, and 
        generate the ResultsObject. This can be accessed using the 
        results_dict[\'simulator\'] attribute.
        
        :returns: None
        
        """
        # Create the simulator object
        self._create_simulator()
        # Add any previous trajectories, if given
        self._from_trajectories('simulator')
        # Run the simulation
        self._run_simulation()
        
        return None
    
    
    def _create_simulator(self):
        """This starts with creating the simualtor object, of which there are
        two. The default is the more robust FESimulator, which is required for
        any model using dosing or steps. This is also used every time a parameter
        fitting problem is run for initialization.
       
        """
        print('Setting up simulation model')
        sim_set_up_options = copy.copy(self.settings.simulator)
        
        # Initialization is now defaulted to FESimulator (settings file)
        dis_method = sim_set_up_options.pop('method', 'fe')
        
        # Override the chosen method to fe if dosing or steps are included
        if self._has_step_or_dosing:
            dis_method = 'fe'
        
        # Choose the proper simulator object
        simulation_class = PyomoSimulator
        if dis_method == 'fe':
             simulation_class = FESimulator
        
        self._s_model = self._create_pyomo_model(is_simulation=True)
        
        # Fix the optimization variables
        opt_vars = self.__var.optimization_variables
        for var in opt_vars:
        
            if hasattr(self._s_model, var):
                for param in getattr(self._s_model, var).values():
                    param.fix()
        
        # Initialize the simulator instance
        if dis_method == 'fe':
            simulator = simulation_class(self._s_model)

        else:
            simulator = simulation_class(self._s_model)

        # Discretize the model
        simulator.apply_discretization(self.settings.collocation.method,
                                       ncp=max(2, self.settings.collocation.ncp),
                                       nfe=self.settings.collocation.nfe,
                                       scheme=self.settings.collocation.scheme)
        
        # Add the dosing points to the model
        if self._has_step_or_dosing:
            for time in simulator.model.alltime.data():
                getattr(simulator.model, self.__var.dosing_variable)[time, self.__var.dosing_component].set_value(time)
                getattr(simulator.model, self.__var.dosing_variable)[time, self.__var.dosing_component].fix()
        
        # Add this simulator to the class attributes
        self.simulator = simulator
        print('Finished creating simulator')
        
        return None
    
    
    def _run_simulation(self):
        """Runs the simulations, may be combined with the above at a later date
        
        """
        if isinstance(self.simulator, FESimulator):
            self._call_fe_factory()
        
        simulator_options = self.settings.simulator
        simulator_options.pop('method', None)
        results = self.simulator.run_sim(**simulator_options)
        self.results_dict['simulator'] = results 
        self.results = results    
        return None
    
    
    def bound_profile(self, var, bounds):
        """Wrapper for TemplateBuilder bound_profile method
        
        The user can define a variable and its bounds using this method.
        
        :param str var: The model variable to be constrained
        :param tuple bounds: The lower and upper bounds of the variable
        
        :return: None
        
        """
        
        self._builder.bound_profile(var=var, bounds=bounds)
        return None
    
    
    """Estimators"""
    
    # def create_variance_estimator(self, **kwargs):
    #     """This is a wrapper for creating the VarianceEstimator"""
        
    #     if len(kwargs) == 0:
    #         kwargs = self.settings.collocation
        
    #     if self._model is None:    
    #         self._create_pyomo_model()  
        
    #     self._create_estimator(estimator='v_estimator', **kwargs)
    #     self._from_trajectories('v_estimator')
    #     return None
        
    # def create_parameter_estimator(self, **kwargs):
    #     """This is a wrapper for creating the ParameterEstiamtor"""
        
    #     if len(kwargs) == 0:
    #         kwargs = self.settings.collocation
            
    #     if self._model is None:    
    #         self._create_pyomo_model()  
            
    #     self._create_estimator(estimator='p_estimator', **kwargs)
    #     self._from_trajectories('p_estimator')
    #     return None
    
    
    def _calculate_S_from_Z_data(self):
        """Calculates the indivdual S profiles from simulated results
        
        :return: The predicted S profiles
        :rtype: pandas.DataFrame
        
        """
        C_orig = self.results_dict['simulator'].Z
        D = self.spectra.data
    
        C = interpolate_trajectory(list(D.index), C_orig)
    
        non_abs_species = self.components.get_match('absorbing', False)
        C = C.drop(columns=non_abs_species)
    
        indx_list = list(D.index)
        for i, ind in enumerate(indx_list):
            indx_list[i] = round(ind, 6)
        
        D.index = indx_list
        
        assert C.shape[0] == D.values.shape[0]
        
        M1 = np.linalg.inv(C.T @ C)
        M2 = C.T @ D.values
        S = (M1 @ M2).T
        S.columns = C.columns
        S = S.set_index(D.columns)
        
        for comp in non_abs_species:
            S[comp] = 0
        
        return S
    
    
    # def _check_S_singularity(self, threshold=1e-5):
    #     """This is still in development and may not actually be too useful
    #     """
    #     C_orig = self.results_dict['simulator'].Z
    #     D = self.spectra.data
    
    #     C = interpolate_trajectory(list(D.index), C_orig)
    
    #     indx_list = list(D.index)
    #     for i, ind in enumerate(indx_list):
    #         indx_list[i] = round(ind, 6)
        
    #     D.index = indx_list
        
    #     M = C.T @ C
    #     print('This is the determinant of C.T @ C')
    #     print(np.linalg.det(M))
        
    #     eigs, U = np.linalg.eigh(M)
        
    #     print('The singular values are:')
    #     print(eigs)
    #     d = eigs
    #     d = np.where(d > 1e-16, d, 1e-16)
    #     d = 1/d
    #     d = np.where(d > threshold, 0, d)
        
    #     print(d)
    #     M1 = np.diag(d)
        
    #     M2 = C.T @ D.values
        
    #     S = (M1 @ M2).T
    #     S.columns = C.columns
    #     S = S.set_index(D.columns)
        
    #     num_comp = sum(np.where(d > 0, 1, 0))

    #     return S, num_comp
        
    # def _initialize_S_from_simulation(self, model):
    #     """This method takes simulated concentration profiles and the given
    #     spectra data to recreate the single species absorbance profiles to be
    #     used as initial values in the variance estimation
        
    #     This directly sets the S values in the TemplateBuilder instance. If
    #     the S profile is provided, it is fixed in the model.
    #     """
        
    #     S = self._calculate_S_from_Z_data()
    #     S[S < 0] = 1e-8
        
    #     print('Adding initialized S to the model')
        
    #     for k in S.columns:
    #         if self.components[k].absorbing:
    #             if self.components[k].S is None:
    #                 # Absorbance is taken from simulated values
    #                 for l in S.index:        
    #                     getattr(model, 'S')[l, k].set_value(float(S[k][l]))
    #             else:
    #                 # The single species absorbance is provided - fit to the collocation points
    #                 for l in S.index:        
    #                     S_comp = interpolate_trajectory(list(S.index), self.components[k].S)
    #                     S_comp = S_comp.set_index(S.index)
    #                     #print(S_comp)
    #                     getattr(model, 'S')[l, k].set_value(float(S_comp[k][l]))
    #                     getattr(model, 'S')[l, k].fix()
    #         else:
    #             # Species does not absorb - set to zero
    #             for l in S.index:        
    #                 getattr(model, 'S')[l, k].set_value(0)
                    
    #     print('Finished initializing the S profiles')        
        
    #     print('Initializing both the C and Z profiles using simulated Z profiles')
    #     getattr(self, 'v_estimator').initialize_from_trajectory('C', self.results_dict['simulator'].Z)
    #     getattr(self, 'v_estimator').initialize_from_trajectory('Z', self.results_dict['simulator'].Z)
        
    #     return None
        
    
    def _initialize_from_simulation(self, estimator='p_estimator'):
        """This method initializes the model using simulated data
        
        :param str estimator: The name of the estimator to be initialized (v_estimator or p_estimator)
        
        :return: None
        
        """
        if not hasattr(self, 's_model') or self._s_model is None:
            self.simulate()

        vars_to_init = get_vars(self._s_model)
        
        #_print(f'The vars_to_init: {vars_to_init}')
        for var in vars_to_init:
            if hasattr(self.results, var) and var != 'S':
                #_print(f'Updating variable: {var}')
                getattr(self, estimator).initialize_from_trajectory(var, getattr(self.results, var))
            elif var == 'S' and hasattr(self.results, 'S'):
                getattr(self, estimator).initialize_from_trajectory(var, getattr(self.results, var))
            else:
                continue
        
        return None
    
    # def initialize_from_reduced_spectral_data(self, estimator='p_estimator'):
        
    #     if not hasattr(self, 's_model'):
    #         _print('Starting simulation for initialization')
    #         self.simulate()
    #         _print('Finished simulation, updating variables...')

    #     _print(f'The model has the following variables:\n{get_vars(self._s_model)}')
    #     vars_to_init = get_vars(self._s_model)
        
    #     _print(f'The vars_to_init: {vars_to_init}')
    #     for var in vars_to_init:
    #         if hasattr(self.results, var):    
    #             _print(f'Updating variable: {var}')
    #             getattr(self, estimator).initialize_from_trajectory(var, getattr(self.results, var))
    #         else:
    #             continue
        
    #     return None
    
    
    def _create_estimator(self, estimator=None):
        """This function handles creating the Estimator object
        
        :param str estimator: p_estimator or v_estimator for the PE or VE
            
        :returns: None
            
        """        
        if estimator == 'v_estimator':
            Estimator = VarianceEstimator
            est_str = 'VarianceEstimator'
            
        elif estimator == 'p_estimator':
            Estimator = ParameterEstimator
            est_str = 'ParameterEstimator'
            
        else:
            raise ValueError('Keyword argument estimator must be p_estimator or v_estimator.')  
        
        model_to_clone = self._create_pyomo_model()
        
        setattr(self, f'{estimator[0]}_model', model_to_clone.clone())
        setattr(self, estimator, Estimator(getattr(self, f'{estimator[0]}_model')))
        getattr(self, estimator).apply_discretization(self.settings.collocation.method,
                                                      ncp=self.settings.collocation.ncp,
                                                      nfe=self.settings.collocation.nfe,
                                                      scheme=self.settings.collocation.scheme)
     
        self._from_trajectories(estimator)
        
        if self.settings.parameter_estimator.sim_init and estimator == 'v_estimator':
            self._initialize_from_simulation(estimator=estimator)
            # This really doesn't seem to help as much - use the simpler implementation above
            #self.initialize_S_from_simulation(getattr(self, 'v_model'))
            
        # What is S between VE and PE?
        if self.settings.parameter_estimator.sim_init and estimator == 'p_estimator':
            if hasattr(self, 'v_estimator'):
                self._initialize_from_variance_trajectory()
            else:
                self._initialize_from_simulation(estimator=estimator)
        
        if self._has_step_or_dosing:
            for time in getattr(self, estimator).model.alltime.data():
                getattr(getattr(self, estimator).model, self.__var.dosing_variable)[time, self.__var.dosing_component].set_value(time)
                getattr(getattr(self, estimator).model, self.__var.dosing_variable)[time, self.__var.dosing_component].fix()
        
        
        return None
    
    # def solve_variance_given_delta(self):
    #     """Wrapper for this VarianceEstimator function"""
    #     variances = self.v_estimator.solve_sigma_given_delta(**self.settings.variance_estimator)
    #     return variances
        
    def _run_ve_opt(self):
        """Wrapper for run_opt method in VarianceEstimator"""
        
        if self.settings.variance_estimator.method == 'direct_sigmas':
            worst_case_device_var = self.v_estimator.solve_max_device_variance(**self.settings.variance_estimator)
            self.settings.variance_estimator.device_range = (self.settings.variance_estimator.best_accuracy, worst_case_device_var)
            
        self._run_opt('v_estimator', **self.settings.variance_estimator)
        
        return None
    
    
    def _run_pe_opt(self):
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
        
        :param list param_to_fix: List of parameter names to fix
        
        :returns: None
        
        """
        if not hasattr(self, 'fixed_params'):
            self.fixed_params = []
        
        if isinstance(param_to_fix, str):
            param_to_fix = [param_to_fix]
            
        self.fixed_params += [p for p in param_to_fix]
        
        return None
    
    
    def run_opt(self):
        """This runs the parameter fitting optimization problem. It will automatically
        perform the variance estimation step performed by the VarianceEstimator. The user
        can define the options for this using the settings attribute.
        
        This method adds the results of the optimization as a ResultsObject to the attribute results.
        This is also conveniently returned from this method as well.
        
        :return: The results of the optimization
        :rtype: ResultsObject
        
        """
        print(f'*** KIPET version {version_number}')
        print(f'*** Starting the parameter estimation procedure')
        
        print(f'\n*** Simulating the model with initial values')
        self.simulate()
        print(f'*** Simulation completed succesfully')
        
        # Check if all needed data for optimization available
        if not self._allow_optimization:
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
            print('*** Generating VarianceEsitmator Instance')
            self._create_estimator(estimator='v_estimator')
            self.settings.variance_estimator.solver_opts = self.settings.solver
            # Optional max device variance
            if self.settings.variance_estimator.max_device_variance:
                max_device_variance = self.v_estimator.solve_max_device_variance(**self.settings.variance_estimator)
            
            # elif self.settings.variance_estimator.use_delta:
            #     variances_with_delta = self.solve_variance_given_delta()

            else:
                print('*** Starting the variance estimator')
                self._run_ve_opt()
                
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
        print('*** Generating ParameterEsitmator Instance')
        self._create_estimator(estimator='p_estimator')
        
        variances = self.components.variances
        self.variances = variances
        
        # The VE results can be used to initialize the PE
        if 'v_estimator' in self.results_dict:
            if self.settings.general.initialize_pe:
                # Update PE using VE results
                self._initialize_from_variance_trajectory()
                # No initialization from simulation is needed
                self.settings.parameter_estimator.sim_init = False
 
            if self.settings.general.scale_pe:
                # Scale variables from VE results
                self._scale_variables_from_variance_trajectory()
            
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
        print('*** Solving the parameter fitting problem...\n')
        self._run_pe_opt()
        
        # Save results in the results_dict
        self.results = self.results_dict['p_estimator']
        self.results.file_dir = pathlib.Path.cwd() #self.settings.general.charts_directory
        
        # Tells MEE that the individual model is already solved
        self._optimized = True
        
        return self.results
    
    
    @staticmethod
    def _scale_variances(variances):
        """If the option to scale variances is True, this will scale the variances
        
        :return: scaled variances
        :rtype: dict
        
        """        
        max_var = max(variances.values())
        scaled_vars = {comp: var/max_var for comp, var in variances.items()}
        return scaled_vars


    def _run_opt(self, estimator, *args, **kwargs):
        """Runs the respective optimization for the given estimator while
        passing all of the arguments along
        
        :return: The results of the respective estimator problem
        :rtype: ResultsObject
        
        """
        if not hasattr(self, estimator):
            raise AttributeError(f'ReactionModel has no attribute {estimator}')
            
        self.results_dict[estimator] = getattr(self, estimator).run_opt(*args, **kwargs)
        return self.results_dict[estimator]
    
    
    def _initialize_from_variance_trajectory(self, variable=None, obj='p_estimator'):
        """Wrapper for the initialize_from_trajectory method in
        ParameterEstimator
        
        """
        source = self.results_dict['v_estimator']
        self._from_trajectory('initialize', variable, source, obj)
        return None
    
    
    def initialize_from_trajectory(self, variable_name=None, source=None):
        """Wrapper for the initialize_from_trajectory method in
        ParameterEstimator or PyomoSimulator
        
        This take a component or state variable already defined and initializes it
        using data already entered into the model. The dataset containing this data is the 
        source argument. See Example 4 to see this in use.
        
        :param str variable_name: The name of the component or state to be initialized
        :param str source: The name of the dataset where the trajectory data is found
        
        :return: None
        
        """
        self._var_to_initialize_from_trajectory.append([variable_name, source])
        return None
    
    
    def _scale_variables_from_variance_trajectory(self, variable=None):
        """Wrapper for the scale_varialbes_from_trajectory method in
        ParameterEstimator
        
        :param str variable: The name of the variable to initialize
        
        :return: None
        
        """
        source = self.results_dict['v_estimator']
        self._from_trajectory('scale_variables', variable, source, 'p_estimator')
        return None
        
    
    @staticmethod
    def _get_source_data(source, var):
        """Get the correct data from a ResultsObject or a DataFrame
        
        :param source: The dataframe or the ResultsObject with the data
        :type source: pandas.DataFrame/ResultsObject
        
        :return: The source data
        :rtype: pandas.DataFrame
        
        """
        if isinstance(source, pd.DataFrame):
            return source
        else:
            return getattr(source, var)
    
    
    def _from_trajectory(self, category, variable, source, obj):
        """Generic initialization/scaling function
        
        :param str category: The estimator to be initialized from
        :param str variable: The variable to be initialized
        :param str source: The dataset name containing the trajectory
        :param str obj: The attribute name of the estimator
        
        :return: None
        
        """
        estimator = getattr(self, obj)
        method = getattr(estimator, f'{category}_from_trajectory')
        
        if variable is None:
            vars_to_init = get_vars(estimator.model)
            for var in vars_to_init:
                if hasattr(source, var):
                    method(var, self._get_source_data(source, var))   
        else:
            method(variable, self._get_source_data(source, variable))
        return None
                   
                            
    def set_known_absorbing_species(self, *args, **kwargs):
        """Wrapper for set_known_absorbing_species in TemplateBuilder
        
        .. note::
            This may be removed. It is not clear if this is still needed.
        
        :return: None
        
        """
        self._builder.set_known_absorbing_species(*args, **kwargs)    
        return None
    
    
    # def scale(self):
    #     """Scale the model"""
        
    #     parameter_dict = self.parameters.as_dict(bounds=False)    
    #     scaled_parameter_dict, scaled_models_dict = scale_models(self._model,
    #                                                              parameter_dict,
    #                                                              name=self.name,
    #                                                              )         
    #     return scaled_parameter_dict, scaled_models_dict
    
    # def clone(self, *args, **kwargs):
    #     """Makes a copy of the ReactionModel and removes the data. This is done
    #     to reuse the model, components, and parameters in an easier manner
        
    #     """
    #     new_kipet_model = copy.deepcopy(self)
        
    #     name = kwargs.get('name', self.name + '_copy')
    #     copy_model = kwargs.get('model', True)
    #     copy_builder = kwargs.get('builder', True)
    #     copy_components = kwargs.get('components', True)   
    #     copy_parameters = kwargs.get('parameters', True)
    #     copy_datasets = kwargs.get('datasets', True)
    #     copy_constants = kwargs.get('constants', True)
    #     copy_settings = kwargs.get('settings', True)
    #     copy_algebraic_variables = kwargs.get('alg_vars', True)
    #     copy_odes = kwargs.get('odes', True)
    #     copy_algs = kwargs.get('algs', True)
        
    #     # Reset the datasets
        
    #     new_kipet_model.name = name
        
    #     if not copy_model:
    #         new_kipet_model.model = None
        
    #     if not copy_builder:
    #         new_kipet_model.builder = TemplateBuilder()
            
    #     if not copy_components:
    #         new_kipet_model.components = ComponentBlock()
        
    #     if not copy_parameters:
    #         new_kipet_model.parameters = ParameterBlock()
            
    #     if not copy_datasets:
    #         del new_kipet_model.datasets
    #         new_kipet_model.datasets = DataBlock()
            
    #     if not copy_constants:
    #         new_kipet_model.constants = None
            
    #     if not copy_algebraic_variables:
    #         new_kipet_model.algebraic_variables = []
            
    #     if not copy_settings:
    #         new_kipet_model.settings = Settings()
            
    #     if not copy_odes:
    #         new_kipet_model.odes = None
            
    #     if not copy_algs:
    #         new_kipet_model.algs = None
        
    #     list_of_attr_to_delete = ['p_model', 'v_model', 'p_estimator',
    #                               'v_estimator', 'simulator']
        
    #     for attr in list_of_attr_to_delete:
    #         if hasattr(new_kipet_model, attr):
    #             setattr(new_kipet_model, attr, None)
        
    #     new_kipet_model.results_dict = {}
            
    #     return new_kipet_model
    
    def rhps_method(self,
                     method='k_aug',
                     calc_method='global',
                     scaled=True):
        """This calls the reduce_models method in the EstimationPotential
        module to reduce the model based on the reduced hessian parameter
        selection method.
        
        This calls the EstimationPotential class and performs the reduced 
        Hessian parameter selection method outlined
        in Chen and Biegler, 2020, AIChE. 
        
        This function returns None, but the resulting reduced model is stored
        under the red_model attribute.
        
        :param str method: The method to be used in calculating the reduced Hessian from the KKT matrix: k_aug or pynumero
        :param str calc_method: The method used to determine the sensitivity of the parameters (global or fixed)
        :param bool scaled: Indicates whether the parameters are scaled
        
        :return: None
        
        """
        if self._model is None:
            self._model = self._create_pyomo_model()
            
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
        results, reduced_model = rhps_method(self._model, **kwargs)
        
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
        """Wrapper for set_non_absorbing_species in TemplateBuilder
        
        :param list non_abs_list: The list of components that do not absorb
        
        :return: None
        """
        
        self._has_non_absorbing_species = True
        self.non_abs_list = non_abs_list
        return None
        
    # def add_noise_to_data(self, var, noise, overwrite=False):
    #     """Wrapper for adding noise to data after data has been added to
    #     the specific ReactionModel
        
    #     """
    #     dataframe = self.datasets[var].data
    #     if overwrite:
    #         self.datasets[var].data = dataframe
    #     return data_tools.add_noise_to_signal(dataframe, noise)    
    
    
    def unwanted_contribution(self, variant, St=None, Z_in=None):
        """This method lets the user define whether the system has unwanted
        contributions in the spectral data.
        
        This is based on the work in () paper goes here
        
        :param str variant: The type of unwanted contribution\: time_variant or time_invariant
        :param dict St: The stoichiometric matrix of the reaction network
        :param dict Z_in: The dosing points, if any
        
        :return: None
            
        """
        self._G_contribution = variant
        
        if St is None:
            St = self.stoich_from_reactions(as_dict=True)
        
        if Z_in is None:
            Z_in = {}
        
        self._G_data = {'G_contribution': variant,
                        'Z_in': Z_in,
                        'St': St,
                        }
        return None
    
    
    def analyze_parameters(self, 
                        method=None,
                        parameter_uncertainties=None,
                        meas_uncertainty=None,
                        sigmas=None,
                        ):
        
        """This is a wrapper for the EstimabilityAnalyzer
        
        :param str method: The estimability method to be used (yao)
        :param dict parameter_uncertainties: The uncertainty in the parameters {parameter: uncertainty}
        :param float meas_uncertainty: Measurement scaling
        :param dict sigmas: The variances of the components {component: variance}
        
        :return: The list of parameters to fit and the list to fix
        :rtype: tuple(list, list)
        
        """
        if not hasattr(self, '_model') or self._model is None:
            self._create_pyomo_model()
        
        # Here we use the estimability analysis tools
        self.e_analyzer = EstimabilityAnalyzer(self._model)
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
    
    # def fix_and_remove_parameters(self, model_name, parameters=None):
        
        
        
    #     if model_name not in ['s_model', 'v_model', 'p_model']:
    #         raise ValueError(f'ReactionModel does not have model type {model_name}')
        
    #     model = getattr(self, model_name)
    #     param_replacer = ParameterReplacer([model], fix_parameters=parameters)
    #     param_replacer.remove_fixed_vars()
    
    #     return None

        
    """MODEL FUNCTION AREA"""
    
    # @staticmethod
    # def _set_up_stoich_mat(rm, St, reaction, component, value):
    #     """This method generates the stoichiometric matrix based on the provided
    #     ODEs.
        
    #     :param ReactionModel rm: The ReactionModel object
    #     :param dict St: The stoichiometric matrix
        
        
    #     """
    #     if component in rm.components.names and reaction in rm.algs_dict:
    #         St[reaction][rm.components.names.index(component)] = value
    #     else:
    #         pass
        

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
    
    def _create_stoich_dataframe(self):
        """Builds the dataframe used to hold the stoichiometric matrix
        
        :return: The empty dataframe for the stoichiometric matrix
        :rtype: pandas.DataFrame
        
        """
        if self.algebraics.get_match('is_reaction', True) is None:
            raise ValueError('You need to declare reaction expressions')
        
        odes = self.odes_dict
        reaction_exprs = self.algebraics.get_match('is_reaction', True)
        dr = pd.DataFrame(np.zeros((len(self.components), len(reaction_exprs))), columns=reaction_exprs, index=self.components.names)    
           
        return dr
    
    
    def reactions_from_stoich(self, St, add_odes=True):
        """The user can provide a stoichiometric matrix and have the reaction
        ODEs generated from it
        
        :param dict St: A dictionary with lists of stoichiometric coefficients
        :param bool add_odes: Indicates if the ODEs can be added to the ReactionModel,
          if not, they will be returned in a dictionary.
        
        .. note::
            If you need to add something else to the ODEs, such as feeds or step
            functions, then the ODEs need to be returned as a dict, modified, and
            added to ReactionModel using the add_odes method.
            
        :Example:
            >>> rA = r1.add_reaction('rA', k1*A, description='Reaction A')
            >>> rB = r1.add_reaction('rB', k2*B, description='Reaction B')
            >>> stoich_data = {'rA': [-1, 1, 0], 'rB': [0, -1, 1]}
            >>> reaction_model.reactions_from_stoich(stoich_data, 'reaction')
            
        :return: dict of reaction expressions (for further additions)
        :rtype: dict
        
        """
        dr = self._create_stoich_dataframe()
        comps = self.components.names
        
        # Check the input
        _is_comp = True
        _is_reaction = True
        
        for key in St.keys():
            if not key in self.components.names:
                _is_comp = False
                break
        for key in St.keys():
            if not key in self.algebraics.get_match('is_reaction', True):
                _is_reaction = False
                break
        
        if not _is_comp and not _is_reaction:
            raise ValueError('There is something wrong with the stoichiometric matrix provided')
        
        if _is_reaction:
            for rxn, s_list in St.items():
                dr.loc[:, rxn] = s_list

        else:
            for comp, s_list in St.items():
                dr.loc[comp, :] = s_list

        # Now we have the dr dataframe, put together the reactions
        odes_dict = {}
        for comp in comps:
            ode = 0
            for rxn in dr.columns:
                ode += dr.loc[comp, rxn]*self.algs_dict[rxn].expression
                
            odes_dict[comp] = ode
            if add_odes:
                self.add_ode(comp, ode)

        return odes_dict if not add_odes else None

    def stoich_from_reactions(self, as_dict=False):
        """Generate a dictionary or dataframe representing the stoichiometric
        reaction matrix given the reaction expressions and the ODES are
        defined
        
        .. note::
          Debugging code still left here because this may still lead to errors
        
        :param bool as_dict: Indicates whether a dictionary (True) or dataframe
          should be returned
          
        :return: The dictionary or dataframe of the stoichiometic matrix
        :rtype: dictionary or dataframe
        
        """
        from pyomo.core.expr.numeric_expr import (DivisionExpression,
                                                  NegationExpression,
                                                  ProductExpression,
                                                  SumExpression)
      
        if self.odes_dict is None or len(self.odes_dict) == 0:
            raise ValueError('You need to input the reaction ODEs')
          
        dr = self._create_stoich_dataframe()
        all_rxns = {key: self.algs_dict[key].expression for key in self.algebraics.get_match('is_reaction', True)}
        
        def check_expr_type(expr):
        
            if isinstance(expr, NegationExpression):
                expr_use = expr.args[0]
                scalar = -1       
                return expr_use, -1
            else:
                return expr, 1
    
        for comp, ode in self.odes_dict.items():
    
        
            expr = ode.expression
            #print('')
            #print(f'Looking at {comp} and {expr}')
            #print(comp, expr)
            #expr_new = 0
            
            #print(f'The ODE expression being handled is:\n     {expr}\n')
            #print(f'The type of expression being handled is:\n     {type(expr)}\n')
        
            #print(type(expr))
        
            expr_use = expr
        
            scalar = 1
        
            if isinstance(expr, NegationExpression):
                expr_use = expr.args[0]
                scalar = -1        
                
            if isinstance(ode.expression, (DivisionExpression, ProductExpression)):
            #    _print(f'The number of terms in this expression is: 1\n')
                expr_use = expr 
            
            if isinstance(ode.expression, SumExpression):
                expr_use = expr.args
                
            #    term = self.check_term(expr, convert_to)
            #    expr_new = scalar*term
            coeff = 1
            
            #else:
            #print(f'The number of terms in this expression is: {len(expr.args)}\n')
                
            expr_use = [expr_use]
 
 #           print('\nStarting For LOOP')
            for i, term in enumerate(expr_use):
  #              print(i, term)
   #             print('')
    #            print(type(term))
                
                if not isinstance(term, list):
                    term = [term]
                for t in term:
     #               print(t)
                    t, scalar_update = check_expr_type(t)
                    
                    for name, rxn in all_rxns.items():
      #                  print(name, rxn)
                        if t is rxn:
       #                     print(f'{name}: {t} == {rxn}')
                            dr.loc[comp, name] = scalar*scalar_update
                        #print(f'Equal to {name}: {term == rxn}')
                
            #term = self.check_term(term, convert_to)
            #expr_new += scalar*term
        
        #_print('The new expression is:\n')
        #_print(f'     {expr_new}')
            
        #self.expression = expr_new
        #self.units = getattr(pyo_units, convert_to)
        self.St = dr
        
        if as_dict:
            _dict = dict(dr)
            St = {k: list(v.values) for k, v in _dict.items()}
            return St
            
        else:
            return dr
        
        
    def add_volume_terms(self):
        """This method will automatically update all component ODEs with 
        terms that account for volume changes (i.e. batch reactors with
        component feeds)
        
        This method will check if the provided volume_state is found within
        the state block of the ReactionModel. If not, an error will be raised.
        
        If the state is found, the state and its ODE will be used to develop
        terms to take the volume changes into account. For example:
        ::
            
            dVdt/V * C
            
        where dVdt is the volume ODE, V is the current volume state, and C is
        the component concentration.
        
        :param str volume_state: The name of the volume state
        
        :return: None
        """
        volume_state = self.__var.volume_name
        
        if volume_state not in self.states.names:
            raise AttributeError(f"State {volume_state} not found in the model.")
            
        if volume_state not in self.odes_dict:
            raise ValueError(f'The state {volume_state} does not have an ODE.')
        
        dVdt = self.odes_dict[volume_state].expression
        V = self.states[volume_state].pyomo_var
        
        # Add the volume change to each component ODE
        for com in self.components.names:
            self.odes_dict[com].expression -= dVdt/V*self.components[com].pyomo_var
        
        self.__volume_terms_added = True
        
        return None
    
    
    @property
    def models(self):
        """Returns a list showing which models have been created for the ReactionModel
        instance. This is more of a debugging method.
        
        .. note::
            This is slated for removal.
        
        :return: Dictionary containing the models as keys and bools indicating their existence
        :rtype: dict

        """
        output = 'ReactionModel has the following:\n'
        output_dict = {}
        
        for model in [name + 'model' for name in ['', 's_', 'v_', 'p_']]:
        
            if hasattr(self, model):
                output += f'{model} True\n'
                output_dict[model] = True
            else:
                output += f'{model} False\n'
                output_dict[model] = False
            
        #print(output)
        return output_dict
    
    @property
    def has_objective(self):
        """Check if p_model has an objective
        
        .. note::
            This is slated for removal.
        
        :return: Boolean showing if the parameter estimator has an objective
        
        """
        return hasattr(self.p_model, 'objective')


    # def check_component_units(self):
    #     """Method to check whether the units provided are consistent with the
    #     base units.
        
    #     """
        
    #     print('Checking model component units:\n')
        
    #     element_dict = {
    #        'parameters': self.parameters,
    #        'components': self.components,
    #        'constants': self.constants,
    #        'algebraics': self.algebraics,
    #        'states': self.states,
    #             }
        
    #     if not hasattr(self, 'c'):
    #         self._make_c_dict()
        
    #     for elem, obj in element_dict.items():
    #         for comp in obj:
    #             comp._check_scaling()
                
    #             if comp.units != comp.units_orig:
    #                 comp.pyomo_var.parent_component()._units = getattr(pyo_units, str(comp.units.u))
        
    #     self._units_checked = True
    #     print('')
        
    #     return None


    # def check_component_units_base(self):
    #     """Method to check whether the units provided are consistent with the
    #     base units
        
    #     """
    #     element_dict = {
    #        'parameters': self.parameters,
    #        'components': self.components,
    #        'constants': self.constants,
    #        'algebraics': self.algebraics,
    #        'states': self.states,
    #             }
        
    #     from kipet.model_components.units_handler import \
    #         convert_single_dimension
        
    #     if not hasattr(self, 'c'):
    #         self._make_c_dict()
        
    #     for elem, obj in element_dict.items():
    #         if elem in ['components', 'states', 'parameters', 'constants']:
    #             for key in obj:
                    
    #                 print(f'Checking units for {key.name}: {key.units}')
                    
    #                 key_comp = key #self.get_state(key)
    #                 key_comp_units = key_comp.units
    #                 key_comp_units = convert_single_dimension(self.unit_base.ur, key_comp_units, self.unit_base.TIME_BASE, power_fixed=False)
                    
    #                 print(key_comp_units)
    #                 print(self.unit_base.VOLUME_BASE)
    #                 print(f'Checking units for {key.name}: {key.units}')
    #                 key_comp_units = convert_single_dimension(self.unit_base.ur, key_comp_units, self.unit_base.VOLUME_BASE, power_fixed=True)
                    
    #                 print(key_comp_units)
                    
    #                 key_comp.units = key_comp_units.units
    #                 key_comp.value *= key_comp_units.m
    #                 key_comp.pyomo_var.parent_component()._units = getattr(pyo_units, str(key_comp.units))
        
    #                 # if self.value is not None:
    #                 #     self.value = quantity.m*self.value
                        
    #                 key_comp.conversion_factor = key_comp_units.m
    #                 # key_comp.units = 1*quantity.units
                    
    #                 if hasattr(key_comp, 'bounds') and key_comp.bounds is not None:
    #                     bounds = list(key_comp.bounds)
    #                     if bounds[0] is not None:
    #                         bounds[0] *= key_comp.conversion_factor
    #                     if bounds[1] is not None:
    #                         bounds[1] *= key_comp.conversion_factor
    #                     key_comp.bounds = (bounds) 
        
    #     # for elem, obj in element_dict.items():
    #     #     for comp in obj:
    #     #         comp._check_scaling()
                
    #     #         if comp.units != comp.units_orig:
    #     #             comp.pyomo_var.parent_component()._units = getattr(pyo_units, str(comp.units.u))
        
    #     self._units_checked = True
    #     print('')
        
    #     return None
    
    def check_model_units(self, orig_units=False, display=False):
        """Method to check the expected units of the algebraic expressions and the odes
        based on the given components and states.
        
        This method goes through the ODEs and algebraics term by term and checks whether the
        provided units (constants, etc.) match with the expected ODE given the base units.
        
        This is shown in Example 16 where one of the constants does not match.
        
        :param bool orig_units: Option to use the original units
        :param bool display: Show the original units
        
        :return: None
        
        """
        from kipet.model_components.units_handler import \
            convert_single_dimension
        
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
                convert_to = ' / '.join([str(key_comp.units_orig), self.unit_base.time])    
                key_comp.use_orig_units = True
            else:
                convert_to = ' / '.join([str(key_comp.units), self.unit_base.time])
            
            expr.check_expression_units(convert_to=str(convert_to), scalar=1)
        
            
        for key, expr in self.algs_dict.items():
            
            expr_obj = self.alg_obj.exprs[key]
            
            if expr_obj.expression_orig is not None:
                expr_use = expr_obj.expression_orig
            else:
                expr_use = expr_obj.expression
            
            key_comp = pyo_units.get_units(expr_use)
            expr.units = key_comp
            
        if display:
            self.ode_obj.display_units()
            print('')
            if len(self.alg_obj) > 0:
                self.alg_obj.display_units()
                print('')
            
            
        return None
    
    
    def plot(self, var=None, jupyter=False, filename=None):
        """Plotting method for the ReactionModel results.
        
        This provides a simple platform for generating figures using Plotly to
        show the results of the simulation or parameter fitting. The plots are
        given a name and timestamp automatically and stored in the charts 
        directory within the working directory.
        
        You can plot individual components as var. If you pass 'Z' as var, all
        components are plotted together. If you pass 'S' as var, all individual
        species absorption profiles are shown. Each state must be individually
        plotted owing to their different scales. Algebraics are also plotted in
        this manner. If var is None, all plots associated with the model will
        be plotted.
        
        :param str var: The variable to be plotted
        :param bool jupyter: Option for accessing plots using Jupyter Notebooks
        :param str filename: The optional filename for the plots
        
        :return: None
        
        """
        from kipet.visuals.plots import PlotObject
        
        self._plot_object = PlotObject(reaction_model=self, jupyter=jupyter, filename=filename)
        
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
        """Quick method to return an algebraic expression
        
        This is useful in some cases where more complex expressions need
        to be constructed.
        
        :param str alg_var: The name of the expression
        
        :return: The Pyomo expression designated by alg_var
        :rtype: Pyomo expression
        
        """
        return self.algs_dict[alg_var].expression
    

    def ode(self, ode_var):
        """Quick method to return an ODE expression
        
        This is useful in some cases where more complex expressions need
        to be constructed.
        
        :param str ode_var: The name of the expression
        
        :return: The Pyomo expression designated by ode_var
        :rtype: Pyomo expression
        
        """
        return self.odes_dict[ode_var].expression

    def odes_expr(self):
        """This method returns a dict of the reactions such that they can be
        augmented or otherwise edited
        
        :return: dictionary of odes expressions
        :rtype: dict
        
        """
        ode_dict = {}
        
        for key, value in self.odes_dict.items():
            ode_dict[key] = value.expression
            
        return ode_dict


    def get_state(self, comp):
        """Generic method to get the component or state variable object
        
        :param str comp: The component or state variable name
        
        :return: The component object
        :rtype: ModelComponent
        
        """
        if comp in self.components:
            return self.components[comp]
        elif comp in self.states:
            return self.states[comp]
 
    def get_alg(self, comp):
        """Quick method to get the algebraic variable object
        
        :param str comp: The name of the algebraic variable
        
        :return: The algebraic variable object
        :rtype: ModelAlgebraic
        
        """
        if comp in self.algebraics:
            return self.algebraics[comp]
        
    def diagnostics(self, model):
        """Tool to show the variables and constraints in a specific model
        
        :param str model: The ReactionModel model
        
        :return: None
        
        """
        from pyomo.environ import Var, Constraint
        print('Model Variables:')
        for element in getattr(self, model).component_objects(Var):
            print(f'{element}: {len(element)}')
            
        print('\nModel Constraints:')
        for element in getattr(self, model).component_objects(Constraint):
            print(f'{element}: {len(element)}')
    
        return None
    