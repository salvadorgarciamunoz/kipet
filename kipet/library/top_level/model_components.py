"""
Data input handling for parameters, components, and models

@author: kevin
"""

class ParameterBlock():
    
    """Data abstraction for multiple ModelParameter instances"""
    
    def __init__(self):
        
        self.parameters = {}
        
    def __getitem__(self, value):
        
        return self.parameters[value]
         
    def __str__(self):
        
        #f'{str(k).rjust(m)} : {v}\n'
        
        #name_len_max = max(len(self.parame.name))
        
        format_string = "{:<10}{:<10}{:<10}\n"
        param_str = 'ParameterBlock:\n'
        param_str += format_string.format(*['Name', 'Init', 'Bounds'])
        
        for param in self.parameters.values():
            
            if param.init is not None:
                init_value = f'{param.init:0.2f}'
            else:
                init_value = 'None'
            
            param_str += format_string.format(param.name, f'{init_value}', f'{param.bounds}')
        
        return param_str

    def __repr__(self):
        return self.__str__()

    def __iter__(self):
        for param, data in self.parameters.items():
            yield data
            
    def __len__(self):
        return len(self.parameters)
    

    def add_parameter_list(self, param_list):
        """Handles lists of parameters or single parameters added to the model
       
        """
        #if isinstance(param_list, list):
        for param in param_list:
            self.add_parameter(*param)        
        
        # elif isinstance(param_list, dict):
        #     for param, value in param_list.items():
        #         print(param, value)
        #         self.add_component({param: value})
        
        return None
    
    def add_parameter(self, *args, **kwargs):
        
        """Should handle a series of different input methods:
          
        KP = KineticParameter('k1', init=1.0, bounds=(0.01, 10))
        builder.add_parameter_temp(KP)
        
        - or -
        
        builder.add_parameter('k1', init=1.0, bounds=(0.01, 10))
        
        - or -
        
        builder.add_parameter('k1', 1.0, (0.01, 10))
            
        """        
        bounds = kwargs.pop('bounds', None)
        init = kwargs.pop('init', None)
        
        if len(args) == 1:
            if isinstance(args[0], ModelParameter):
                self.parameters[args[0].name] = args[0]
            
            elif isinstance(args[0], (list, tuple)):
                args = [a for a in args[0]]
                self._add_parameter_with_terms(*args)
                
            elif isinstance(args[0], dict):
                args = [[k] + [*v] for k, v in args[0].items()][0]
                self._add_parameter_with_terms(*args)
                
            elif isinstance(args[0], str):
                self._add_parameter_with_terms(args[0], init, bounds)
                
            else:
                raise ValueError('For a parameter a name and initial value are required')
            
        elif len(args) >= 2:
            
            _args = [args[0], None, None]
                    
            if init is not None:
                _args[1] = init
            else:
                if not isinstance(args[1], (list, tuple)):
                    _args[1] = args[1]

            if bounds is not None:
                _args[2] = bounds
            else:
                if len(args) == 3:
                    _args[2] = args[2]
                else:
                    if _args[1] is None:
                        _args[2] = args[1]
                        
            self._add_parameter_with_terms(*_args)
    
        return None
        
    def _add_parameter_with_terms(self, name, init=None, bounds=None):
        """Adds the parameter using explicit inputs for the name, init, and 
        bounds
        
        """
        param = ModelParameter(name=name,
                               init=init,
                               bounds=bounds
                              )
            
        self.parameters[param.name] = param
            
        return None

    def as_dict(self, factor=None, bounds=False):
        """Returns the parameter data as a dict that can be used directly in
        other kipet methods
        
        """
        if bounds:
            if factor is None:
                return {p.name: (p.init, p.bounds) for p in self.parameters.values()}
    
            elif isinstance(factor, (int, float)):
                return {p.name: (factor*p.init, p.bounds) for p in self.parameters.values()}
    
            elif len(factor) == len(self.parameters):
                return {p.name: (factor[i]*p.init, p.bounds) for i, p in enumerate(self.parameters.values())}
    
            else:
                raise ValueError('Invalid factor passed: must be scalar (float, int) or an array with length equal to the number of parameters')

        else:
            if factor is None:
                return {p.name: p.init for p in self.parameters.values()}
    
            elif isinstance(factor, (int, float)):
                return {p.name: factor*p.init for p in self.parameters.values()}
    
            elif len(factor) == len(self.parameters):
                return {p.name: factor[i]*p.init for i, p in enumerate(self.parameters.values())}
    
            else:
                raise ValueError('Invalid factor passed: must be scalar (float, int) or an array with length equal to the number of parameters')

    def update(self, attr, values):
        """Update attributes using a dictionary"""
        if isinstance(values, dict):
            for key, val in values.items():
                if key in self.names: 
                    setattr(self[key], attr, val)
       
    @property 
    def names(self):
        return [param for param in self.parameters]
    
    @property
    def bounds(self):
        return {p: self.parameters[p].bounds for p in self.parameters}


class ModelParameter():
    """A simple class for holding kinetic parameter data
    
    TODO: change init to value
    """

    def __init__(self, name=None, init=None, bounds=None): #, variance=None):

        # Check if name is provided
        if name is None:
            raise ValueError('ModelParameter requires a name (k1, k2, etc.)')
        else:
            self.name = name
        
        # Check if initial value is provided - if not, then bounds is required
        if init is not None:
            self.init = init
        else:
            if bounds is not None:
                self.init = sum(bounds)/2
            else:
                raise ValueError('Both init and bounds cannot be None')
        
        # Make sure bounds has a length of 2
        if bounds is not None:
            if len(bounds) != 2:
                raise ValueError('The bounds need to be a list or tuple with a lower and upper bound')
        
        self.bounds = bounds
        
        return None
        
    def __str__(self):
        return f'ModelParameter({self.name}, init={self.init}, bounds={self.bounds})'

    def __repr__(self):
        return f'ModelParameter({self.name}, init={self.init}, bounds={self.bounds})'

    @property
    def lb(self):
        """Lower bound property"""
        return self.bounds[0]

    @property
    def ub(self):
        """Upper bound property"""
        return self.bounds[1]
    
class ComponentBlock():
    
    def __init__(self):
        
        self.components = {}
        
    def __getitem__(self, value):
        
        return self.components[value]
         
    def __str__(self):
        
        format_string = "{:<10}{:<20}{:<10}{:<10}\n"
        comp_str = 'ComponentBlock:\n'
        comp_str += format_string.format(*['Name', 'Category', 'Init', 'Variance'])
        
        for comp in self.components.values():
            
            comp_variance = 'None'
            if comp.variance is not None:
                comp_variance = f'{comp.variance:0.2e}'
                
            comp_init = 'None'
            if comp.init is not None:
                comp_init = f'{comp.init:0.2e}'
            
            comp_str += format_string.format(comp.name, comp.state, comp_init, comp_variance)
        
        return comp_str

    def __repr__(self):
        return self.__str__()
    
    def __iter__(self):
        for comp, data in self.components.items():
            yield data
    
    def __len__(self):
        return len(self.components)
    
    def add_component_list(self, comp_list):
        """Handles lists of parameters or single parameters added to the model
        """
        # if isinstance(comp_list, list):
        for comp in comp_list:
            self.add_component(*comp)         
        
        # elif isinstance(comp_list, dict):
        #     for comp, value in comp_list.items():
        #         comp_as_list = [comp]
        #         if not isinstance(value, list):
        #             value = [value]
        #         comp_as_list += [v for v in value]
        #         self.add_component(*comp_as_list)
        
        return None
    
    def add_component(self, *args, **kwargs):
        
        """Should handle a series of different input methods:
          
        C = ModelComponent('A', state='concentration', init=1.0, variance=0.002)
        """        
        # if isinstance(args[0], list):
        #     self.add_component_list(args[0])

        # if isinstance(args[0], dict):
        #     self.add_component_list(args[0])
        
        state = kwargs.pop('state', None)
        init = kwargs.pop('init', None)
        variance = kwargs.pop('variance', None)
        known = kwargs.pop('known', True)
        bounds = kwargs.pop('bounds', None)
        
        if len(args) == 1:
            if isinstance(args[0], ModelComponent):
                self.components[args[0].name] = args[0]
            
            elif isinstance(args[0], (list, tuple)):
                args = [a for a in args[0]]
                self._add_component_with_terms(*args)
                
            elif isinstance(args[0], dict):
                args = [[k] + [*v] for k, v in args[0].items()][0]
                self._add_component_with_terms(*args)
                
            elif isinstance(args[0], str):
                self._add_component_with_terms(args[0], state, init, variance, known, bounds)
                
            else:
                raise ValueError('For a component a name, state, and initial value are required')
            
        elif len(args) >= 2:
            
            _args = [args[0], None, None, None, True, None]
            print(args)
            print(len(args))
            
                    
            if state is not None:
                _args[1] = state
            else:
                _args[1] = args[1]
            
            if init is not None:
                _args[2] = init
            else:
                if len(args) >= 3:
                    _args[2] = args[2]
                    
            if variance is not None:
                _args[3] = variance
            else:
                if len(args) == 4:
                    _args[3] = args[3]
                    
            print(_args)
            self._add_component_with_terms(*_args)
    
        return None
    
    @property
    def variances(self):
        return {comp.name: comp.variance for comp in self.components.values()}
    
    @property
    def init_values(self):
        return {comp.name: comp.init for comp in self.components.values()}
    
    @property
    def known_values(self):
        return {comp.name: comp.known for comp in self.components.values()}
        
    @property
    def names(self):
        return [comp.name for comp in self.components.values()]
    
    @property
    def has_all_variances(self):
    
        all_component_variances = True
        for comp in self.components.values():
            if comp.variance is None:
                all_component_variances = False
            if not all_component_variances:
                break
        return all_component_variances
    
    def _add_component_with_terms(self, name, state, init=None, variance=None, known=True, bounds=None):
        """Adds the parameter using explicit inputs for the name, init, and 
        bounds
        
        """
        if not isinstance(name, str):
            raise ValueError('Component name must be a string')
        
        if not isinstance(state, str):
            raise ValueError('Component state must be a string')
        
#        if not isinstance(init, (int, float)):
 #           raise ValueError('Component initial value must be a number')
        
        comp = ModelComponent(name=name,
                              state=state,
                              init=init,
                              known=known,
                              bounds=bounds,
                              )
        
        if variance is not None:
            comp.variance = variance
            
        self.components[comp.name] = comp
            
        return None
    
    def update(self, attr, values):
        """Update attributes using a dictionary"""
        if isinstance(values, dict):
            for key, val in values.items():
                if key in self.names: 
                    setattr(self[key], attr, val)

class ModelComponent():
    """A simple class for holding component information"""
    
    def __init__(self, name=None, state=None, init=None, variance=None, known=True, bounds=None):
    
        if name is None:
            raise ValueError('Component requires a name (Should match provided data')
        
        # if init is None:
        #     raise ValueError('Compnent requires an initial value "init = ..."')
    
        if variance is None:
            print(f'Warning: Component {name} variance not provided')
            
        if state is None:
            raise ValueError('Component requires a state (complementary, concentration)')
        
        if not known: 
            if bounds is None:
                raise ValueError('If the component\'s initial value is unknown, bounds are required')
            elif not isinstance(bounds, (list, tuple)):
                raise ValueError('Bounds must be a list or tuple')
            else:
                if len(bounds) != 2:
                    raise ValueError('Bounds must have both a lower and an upper bound')
        
        self.name = name
        self.init = init
        self.variance = variance
        self.state = state
        self.known = known
        self.bounds = bounds

    def __str__(self):
        return f'Component: {self.name}, {self.state}, init={self.init}, variance={self.variance}, init_known={self.known}'
    
    def __repr__(self):
        return f'Component: {self.name}, {self.state}, init={self.init}, variance={self.variance}, init_known={self.known}'

