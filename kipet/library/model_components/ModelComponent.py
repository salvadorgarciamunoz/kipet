"""
ModelComponent Classes
"""
# Third party imports
import pint

# KIPET library imports

CONCENTRATION_BASE = 'mol/l'
TIME_BASE = 's'


class ModelElement():
    
    """Abstract Class to handle the modeling components in KIPET"""
    
    def __init__(self, 
                 name=None,
                 class_=None, 
                 value=None,
                 units=None,
                 description=None,
                 ): 
        
        self.name = self._check_name(name)
        self.class_ = class_
        self.value = value
        self.description = description
    
        self.ur = pint.UnitRegistry()
        self.units = 1*self.ur('') if units is None else 1*self.ur(units)
        #self._check_scaling()

    def _check_scaling(self):
        
        quantity = self.value * self.units
        quantity.ito_base_units()
        
        concentration_units = self.ur(CONCENTRATION_BASE)
        time_units = self.ur(TIME_BASE)
        
        # if quantity.units.dimensionality == concentration_units.dimensionality:
        #     quantity.ito(concentration_units)
            
        # else:
        # quantity.ito_base_units()
        print(f'Converting {self.name} to base units {quantity.m} {quantity.units}')
            
        self.units = quantity.units
        self.value = quantity.m
        
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
    
    
class ModelAlgebraic(ModelElement):
    
    class_ = 'model_algebraic'
    
    def __init__(self,
                 name=None,
                 value=None,
                 bounds=(None, None),
                 units=None,
                 description=None,
                 data=None,
                 step=None,
                 ):
    
        super().__init__(name, ModelComponent.class_, value, units, description)
   
        self.bounds = bounds
        self.data = data
        self.step = step
   
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
                 known=True,
                 bounds=(None,None),
                 description=None,
                 absorbing=True,
                 ):
    
        super().__init__(name, ModelComponent.class_, value, units, description)
   
        # component attributes
        self.variance = variance
        self.state = 'concentration'
        self.known = known
        self.bounds = bounds
        self.absorbing = absorbing
        
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
                 known=True,
                 bounds=None,
                 description=None,
                 ):
    
        super().__init__(name, ModelComponent.class_, value, units, description)
   
        # component attributes
        self.variance = variance
        self.state = state
        self.known = known
        self.bounds = bounds
        
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

        
class ModelParameter(ModelElement):
    """A simple class for holding kinetic parameter data
    
    TODO: change init to value
    """

    class_ = 'model_parameter'

    def __init__(self,
                 name,
                 value=None,
                 units=None,
                 bounds=(None, None),
                 fixed=False,
                 variance=None,
                 description=None,
                 ):

        super().__init__(name, ModelParameter.class_, value, units, description)
        
        # parameter attributes
        self.bounds = bounds
        self.fixed = fixed
        self.variance = variance
    
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
                 description=None,
                 ):

        super().__init__(name, ModelConstant.class_, value, units, description)
        self._class_ = type(self)
    
    def __str__(self):
        
        margin = 25
        settings = 'ModelConstant\n'
        
        for key in self.__dict__: #['name', 'class_', 'value', 'units']:
            if key == 'class_':
                continue
            settings += f'{str(key).rjust(margin)} : {getattr(self, key)}\n'
            
        return settings
        
    def __repr__(self):
        
        return f'ModelConstant({self.name})'