"""
ModelComponent Classes
"""
# Kipet library imports
from kipet.model_components.units_handler import convert_single_dimension


class ModelElement:
    
    """Abstract Class to handle the modeling components in KIPET"""
    
    def __init__(self, 
                 name=None,
                 class_=None, 
                 value=None,
                 units=None,
                 unit_base=None,
                 description=None,
                 pyomo_var=None,
                 model_var=None,
                 units_orig=None,
                 conversion_factor=1,
                 ): 
        """The initialization of a general model component (higher level)
        
        :param str name: The name of the model parameter
        :param str class: The subclass of model component
        :param float value: The initial value
        :param str units: The units used to describe the parameter
        :param UnitRegistry unit_base: Inherited unit registry object
        :param str description: Optional description of the component (used in plots)
        :param GeneralVar pyomo_var: The dummy pyomo variable representing this component
        :param str model_var: The model variable where the component is indexed
        :param str units_orig: The originally provided units
        :param float conversion_factor: An attribute for converting scalars in unit changes
        
        """
        self.name = self._check_name(name)
        self.class_ = class_
        self.value = value
        self.description = description
    
        self.pyomo_var = pyomo_var
        self.model_var = model_var
        self.unit_base = unit_base
        self.ur = unit_base.ur
        self.units = 1*self.ur('') if units is None else 1*self.ur(units)
        self.units_orig = self.ur(units_orig)
        self.use_orig_units = False
        self.conversion_factor = conversion_factor
        #self._check_scaling()

    def _check_scaling(self):

        quantity = 1 * self.units
        quantity.ito_base_units()
        
        quantity = convert_single_dimension(self.ur, quantity, self.unit_base.TIME_BASE, power_fixed=False)
        quantity = convert_single_dimension(self.ur, quantity, self.unit_base.VOLUME_BASE, power_fixed=True)
        
        if not self.units.u == quantity.u:
            
            margin = 8
            
            print(f'Converting {self.name}:')
            print('From'.rjust(margin) + f' : {self.value:e} {self.units.u}')
            print('To'.rjust(margin) + f' : {self.value*quantity.m:e} {quantity.u}')
            
            self.conversion_factor = quantity.m
            self.units = 1*quantity.units
            
            if self.value is not None:
                self.value = quantity.m*self.value
                
            if hasattr(self, 'bounds') and self.bounds is not None:
                bounds = list(self.bounds)
                if bounds[0] is not None:
                    bounds[0] *= self.conversion_factor
                if bounds[1] is not None:
                    bounds[1] *= self.conversion_factor
                self.bounds = (bounds) 

    # def convert_unit(self, u_orig, u_goal, scalar=1, power=1, both_powers=False):
   
    #     c1 = 1*self.ur(u_orig)
    #     c2 = 1*self.ur(u_goal)
        
    #     power2 = 1
    #     if both_powers:
    #         power2 = power
            
    #     con = (c1**power).to((c2**power2).u)/c2**power2 * (c2**power2).u/(c1**power).u
        
    #     return scalar * con
    
    # def convert_single_dimension(self, unit_to_convert, u_goal, power_fixed=False):
        
    #     u_g = self.ur(u_goal)
    #     orig_dim = {k.replace('[', '').replace(']', ''): v for k, v in dict(u_g.dimensionality).items()}
    #     units = {k: v for k, v in unit_to_convert._units.items()}
    #     dim_to_find = list(orig_dim.keys())[0]
    #     power = orig_dim[dim_to_find]
        
    #     # print(f'unit_to_convert: {unit_to_convert}')
    #     # print(f'dim_to_find: {dim_to_find}')
    #     # print(f'power {power}')
        
    #     for dims in units:
            
    #         s = self.ur(dims)
    #         s = list({k.replace('[', '').replace(']', ''): v for k, v in dict(s.dimensionality).items()})[0]
    #         # print(f's: {s}')
            
    #         if s == dim_to_find:
    #             if power_fixed and abs(units[dims]) == power:                                       
    #                 power = units[dims]
    #                 u_orig = dims
    #                 # print(dims, power)
    #                 con = self.convert_unit(u_orig, u_goal, power=abs(power))
    #                 # print(con)
    #                 con = con ** (power/abs(power))
    #                 new_unit = unit_to_convert * con
    #                 return new_unit
    #             elif not power_fixed:
    #                 power = units[dims]
    #                 u_orig = dims
    #                 # print(dims, power)
    #                 con = self.convert_unit(u_orig, u_goal, power=abs(power), both_powers=True)
    #                 # print(con)
    #                 con = con ** (power/abs(power))
    #                 new_unit = unit_to_convert * con
    #                 return new_unit
        
    #     return unit_to_convert
        
    def _check_name(self, name):
        """Check for valid attr names in the given string
        
        Args:
            name (str): given name for a python attribute
        
        Returns:
            checked_name (str): valid attribute name
            
        """
        string_replace_dict = {
            ' ': '_',
            '-': 'n',
            '+': 'p',
            '.': '_',
            }
        
        name = str(name)
        
        if name[0].isnumeric():
            name = 'y' + name
        
        for k, v in string_replace_dict.items():
            name = name.replace(k, v)
        
        return name
    
    @property
    def in_orig_units(self):
        """Convert the value of the element back into its original unit value
        
        """
        current = self.value*self.units
        original = current.to(self.units_orig.units)
        return original.m
    
class ModelAlgebraic(ModelElement):
    
    class_ = 'model_algebraic'
    
    def __init__(self,
                 name=None,
                 value=None,
                 bounds=(None, None),
                 units=None,
                 unit_base=None,
                 description=None,
                 data=None,
                 step=None,
                 pyomo_var=None,
                 model_var=None,
                 units_orig=None,
                 conversion_factor=1,
                 is_reaction=False,
                 active=True,
                 ):
    
        """The initialization of a model algebraic (dosing, steps, custom, etc.)
        
        :param str name: The name of the model component
        :param float value: The initial value 
        :param tuple bounds: A tuple of two floats to place bounds on the component trajectories
        :param str units: The units used to describe the component
        :param UnitRegistry unit_base: Inherited unit registry object
        :param bool known: Indicates if the value attribute is fixed
        :param str description: Optional description of the component (used in plots)
        :param pandas.DataFrame data: Optional data - if used this will fix the trajectory
        :param dict step: If the algebraic is a step variable, the data is passed here
        :param GeneralVar pyomo_var: The dummy pyomo variable representing this component
        :param str model_var: The model variable where the component is indexed
        :param str units_orig: The originally provided units
        :param float conversion_factor: An attribute for converting scalars in unit changes
        :param bool is_reaction: Indicates if the algebraic is a reaction equation
        :param bool active: Turns the algebraic equation on/off
        
        """
        super().__init__(name, ModelComponent.class_, value, units,
                         unit_base, description, pyomo_var, model_var, 
                         units_orig, conversion_factor,
                         )
   
        self.bounds = bounds
        self.data = data
        self.step = step
        self.is_reaction = is_reaction
        self.active = active
    
    def __str__(self):
        
        margin = 25
        settings = f'ModelAlgebraic\n'
        
        for key in self.__dict__: #['name', 'class_', 'value', 'units']:
            if key == 'class_':
                continue
            settings += f'{str(key).rjust(margin)} : {getattr(self, key)}\n'
            
        return settings
        
    def __repr__(self):
        
        return f'ModelAlgebraic({self.name})'
    
    @property
    def lb(self):
        """Lower bound property"""
        return self.bounds[0]

    @property
    def ub(self):
        """Upper bound property"""
        return self.bounds[1]

    
class ModelComponent(ModelElement):
    """A simple class for holding component information"""
    
    class_ = 'model_component'
    
    
    def __init__(self,
                 name=None,
                 state=None,
                 value=None,
                 variance=None,
                 units=None,
                 unit_base=None,
                 known=True,
                 bounds=(None,None),
                 description=None,
                 absorbing=True,
                 pyomo_var=None,
                 model_var=None,   
                 units_orig=None,
                 conversion_factor=1,
                 inert=False,
                 S=None
                 ):
        """The initialization of a model component (concentration based)
        
        :param str name: The name of the model component
        :param str state: Fixed to concentration
        :param float value: The initial value (usually concentration)
        :param float variance: The component variance if known
        :param str units: The units used to describe the component
        :param UnitRegistry unit_base: Inherited unit registry object
        :param bool known: Indicates if the value attribute is fixed
        :param tuple bounds: A tuple of two floats to place bounds on the component trajectories
        :param str description: Optional description of the component (used in plots)
        :param bool absorbing: Indicates if the component is seen in the absorbtion data
        :param GeneralVar pyomo_var: The dummy pyomo variable representing this component
        :param str model_var: The model variable where the component is indexed
        :param str units_orig: The originally provided units
        :param float conversion_factor: An attribute for converting scalars in unit changes
        :param bool inert: Indicates if the component participates in reactions
        :param pandas.DataFrame S: The absorbtion spectra, if known
        
        """
    
        super().__init__(name, ModelComponent.class_, value, units,
                         unit_base, description, pyomo_var, model_var,
                         units_orig, conversion_factor)
        # component attributes
        self.variance = variance
        self.state = 'concentration'
        self.known = known
        self.bounds = bounds
        self.absorbing = absorbing
        self.inert = inert
        self.S = S
        
        #self._check_units()
        
    def __str__(self):
        
        margin = 25
        settings = f'ModelComponent\n'
        
        for key in self.__dict__: #['name', 'class_', 'value', 'units']:
            if key == 'class_':
                continue
            settings += f'{str(key).rjust(margin)} : {getattr(self, key)}\n'
            
        return settings
        
    def _check_units(self):
      
        if self.state == 'concentration':
            check_quantity = 1 * self.units
            if not check_quantity.check('[concentration]'):
                raise AttributeError(f'Concentration units incorrect for species {self.name}')
    
    
    def __repr__(self):
        
        return f'ModelComponent({self.name})'
    
    @property
    def lb(self):
        """Lower bound property"""
        return self.bounds[0]

    @property
    def ub(self):
        """Upper bound property"""
        return self.bounds[1]

    
    
class ModelState(ModelElement):
    """A simple class for holding non-component state information"""
    
    class_ = 'model_state'
    
    def __init__(self,
                 name=None,
                 state='state',
                 value=None,
                 variance=None,
                 units=None,
                 unit_base=None,
                 known=True,
                 data=None,
                 bounds=(None, None),
                 description=None,
                 pyomo_var=None,
                 model_var=None,
                 units_orig=None,
                 conversion_factor=1,
                 ):
        """The initialization of a model state
        
        :param str name: The name of the model state
        :param str state: Fixed to state
        :param float value: The initial value (usually concentration)
        :param float variance: The state variance if known
        :param str units: The units used to describe the component
        :param UnitRegistry unit_base: Inherited unit registry object
        :param bool known: Indicates if the value attribute is fixed
        :param tuple bounds: A tuple of two floats to place bounds on the component trajectories
        :param str description: Optional description of the component (used in plots)
        :param GeneralVar pyomo_var: The dummy pyomo variable representing this component
        :param str model_var: The model variable where the component is indexed
        :param str units_orig: The originally provided units
        :param float conversion_factor: An attribute for converting scalars in unit changes
        
        
        """
        super().__init__(name, ModelComponent.class_, value, units,
                         unit_base, description, pyomo_var, model_var, 
                         units_orig, conversion_factor)
        # component attributes
        self.variance = variance
        self.state = state
        self.known = known
        self.bounds = bounds
        self.data = data
        
        #self._check_units()
        
    def __str__(self):
        
        margin = 25
        settings = f'ModelState\n'
        
        for key in self.__dict__: #['name', 'class_', 'value', 'units']:
            if key == 'class_':
                continue
            settings += f'{str(key).rjust(margin)} : {getattr(self, key)}\n'
            
        return settings
        
    def _check_units(self):
      
        if self.state == 'state':
            check_quantity = 1 * self.units
            if not check_quantity.check('[concentration]'):
                raise AttributeError(f'Concentration units incorrect for species {self.name}')
    
    
    def __repr__(self):
        
        return f'ModelState({self.name})'

    @property
    def lb(self):
        """Lower bound property"""
        return self.bounds[0]

    @property
    def ub(self):
        """Upper bound property"""
        return self.bounds[1]        


class ModelParameter(ModelElement):
    """A simple class for holding kinetic parameter data
    
    TODO: change init to value
    """

    class_ = 'model_parameter'

    def __init__(self,
                 name,
                 value=None,
                 units=None,
                 unit_base=None,
                 bounds=(None, None),
                 fixed=False,
                 variance=None,
                 description=None,
                 pyomo_var=None,
                 model_var=None,
                 units_orig=None,
                 conversion_factor=1,
                 ):
        """The initialization of a model parameter
        
        :param str name: The name of the model parameter
        :param float value: The initial value
        :param str units: The units used to describe the parameter
        :param UnitRegistry unit_base: Inherited unit registry object
        :param tuple bounds: A tuple of two floats to place bounds on the component trajectories
        :param bool fixed: Indicates if the parameter is fixed
        :param float variance: The parameter variance if known
        :param str description: Optional description of the component (used in plots)
        :param GeneralVar pyomo_var: The dummy pyomo variable representing this component
        :param str model_var: The model variable where the component is indexed
        :param str units_orig: The originally provided units
        :param float conversion_factor: An attribute for converting scalars in unit changes
        
        """
        super().__init__(name, ModelComponent.class_, value, units,
                         unit_base, description, pyomo_var, model_var, 
                         units_orig, conversion_factor)
        self.bounds = bounds
        self.fixed = fixed
        self.variance = variance
        
        if self.value is None:
            self.value = sum(self.bounds)/2
    
    def __str__(self):
        
        margin = 25
        settings = 'ModelParameter\n'
        
        for key in self.__dict__: #['name', 'class_', 'value', 'units']:
            if key == 'class_':
                continue
            settings += f'{str(key).rjust(margin)} : {getattr(self, key)}\n'
            
        return settings
        
    def __repr__(self):
        
        return f'ModelParameter({self.name})'
    
    @property
    def lb(self):
        """Lower bound property"""
        return self.bounds[0]

    @property
    def ub(self):
        """Upper bound property"""
        return self.bounds[1]
    
    
class ModelConstant(ModelElement):
    """A simple class for holding kinetic parameter data
    
    TODO: change init to value
    """

    class_ = 'model_constant'

    def __init__(self,
                 name,
                 value=None,
                 units=None,
                 unit_base=None,
                 description=None,
                 pyomo_var=None,
                 model_var=None,
                 units_orig=None,
                 conversion_factor=1,
                 ):
        """The initialization of a model constant
        
        :param str name: The name of the model parameter
        :param float value: The initial value
        :param str units: The units used to describe the parameter
        :param UnitRegistry unit_base: Inherited unit registry object
        :param str description: Optional description of the component (used in plots)
        :param GeneralVar pyomo_var: The dummy pyomo variable representing this component
        :param str model_var: The model variable where the component is indexed
        :param str units_orig: The originally provided units
        :param float conversion_factor: An attribute for converting scalars in unit changes
        
        """
        super().__init__(name, ModelComponent.class_, value, units,
                         unit_base, description, pyomo_var, model_var, 
                         units_orig, conversion_factor)
        self._class_ = type(self)
    
    def __str__(self):
        
        margin = 25
        settings = 'ModelConstant\n'
        
        for key in self.__dict__:
            if key == 'class_':
                continue
            settings += f'{str(key).rjust(margin)} : {getattr(self, key)}\n'
            
        return settings
        
    def __repr__(self):
        
        return f'ModelConstant({self.name})'
