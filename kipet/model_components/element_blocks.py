"""
The components in KIPET are stored in various subclasses of ModelElementBlocks
"""
import kipet.model_components.model_components as MC
from kipet.general_settings.variable_names import VariableNames


class ModelElementBlock:
    
    """Data abstraction for multiple ModelElement instances
    
    :Methods:
        
        - :func:`add_element_list`
        - :func:`add_element`
        - :func:`as_dict`
        - :func:`update`
        - :func:`get_match`
        
    """
    
    def __init__(self, class_name):
        """Initialize the ModelElementBlock
        
        :param str class_name: The name of the element class
        
        """
        attr_name = class_name.split('_')[-1] + 's'
        self.attr_class_set_name = attr_name
        setattr(self, attr_name, {})
        
        self.element_object_name = ''.join([term.capitalize() for term in class_name.split('_')])
        self.element_object = getattr(MC, self.element_object_name)
        
        self._dict = getattr(self, self.attr_class_set_name)
        
    def __getitem__(self, value):
        
        return getattr(self, self.attr_class_set_name)[value]
    
    def __add__(self, other):
        
        return {**self._dict, **other._dict}
         
    def __str__(self):
        
        format_string = "{:<10}{:<15}{:<15}\n"
        param_str = f'{self.element_object_name}:\n'
        param_str += format_string.format(*['Name', 'Value', 'Units'])
        
        for elem in self._dict.values():
    
            elem_name = elem.name
            elem_value = 'None' if elem.value is None else elem.value
            elem_units = 'None' if elem.units is None or elem.units  else elem.units

            param_str += format_string.format(elem_name, elem_value, elem_units)
        
        return param_str

    def __repr__(self):
        return self.__str__()

    def __iter__(self):
        for param, data in getattr(self, self.attr_class_set_name).items():
            yield data
            
    def __len__(self):
        return len(getattr(self, self.attr_class_set_name))
    
    def __contains__(self, key):
        return key in self._dict

    def add_element_list(self, elem_list):
        """Handles lists of parameters or single parameters added to the model
       
        :param list elem_list: List of elements to be added to the Block
        
        :return: None
        
        """
        for elem in elem_list:
            self.add_element(*elem)        
        
        return None
    
    def add_element(self, name, **kwargs):
        """General method to add model components to the Block
          
        :param str name: The name of the element
        
        :param dict kwargs: Element specific attributes passed to the specific
          element block (subclass)
          
        :return: None
        
        """
        element = self.element_object(name,
                                      **kwargs,
                                      )
            
        getattr(self, self.attr_class_set_name)[element.name] = element
        
        return None
        
    def as_dict(self, attr):
        """Return certain element attributes as a dictionary
        
        :param str attr: The target attribute
        
        :return return_dict: The dict of target attributes
        :rtype: dict
        
        """
        return_dict = {}
        for param, obj in self._dict.items():
            return_dict[param] = getattr(obj, attr)
        return return_dict

    def update(self, attr, dict_data):
        """Update a specific attr of the element block using a dict of new
        data
        
        :param str attr: The name of the attribute to change
        :param dict dict_data: The dict of new data (comp: data)
        
        :return: None
        
        """
        for elem, new_data in dict_data.items():
            if elem in self._dict:
                setattr(self._dict[elem], attr, new_data)

        return None

    def get_match(self, attr, query):
        """Return a list of elements with an attribute matching a given query
        
        :param str attr: The target attribute
        :param str query: The query to match attributes to
        
        :return query_list: A list of matching elements
        :rtype: list
        
        """
        query_list = []
        for elem, obj in self._dict.items():
            if getattr(obj, attr) == query:
                query_list.append(elem)
                
        return query_list

    @property 
    def names(self):
        """Return the list of element names in the specific class type (this
        is determined by the subclass)

        :return: list of element names
        :rtype: list
        """
        return [elem for elem in getattr(self, self.attr_class_set_name)]
    
    
class ConstantBlock(ModelElementBlock):
    
    """Class to hold ModelElements of type constant and provide specific
    methods.
    
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(class_name='model_constant')


class AlgebraicBlock(ModelElementBlock):
    
    """Class to hold ModelElements of type algebraics and provide specific
    methods.
    
    :Methods:
        
        - :func:`fixed`
        - :func:`steps`
    
    """
    __var = VariableNames()
    
    def __init__(self, *args, **kwargs):
        super().__init__(class_name='model_algebraic')
       
    @property
    def fixed(self):
        """Return a list of those algebraics that are fixed_states
        
        This formats the list in the correct manner that fix_from_trajectory
        can use to access the data
        
        :return fix_from_traj: The list of modal var, component, and data for
          fixed trajectories/states
        :rtype: list
        
        """
        fix_from_traj = []
        for alg in self.algebraics.values():
            if alg.data is not None:
                fix_from_traj.append([self.__var.algebraic, alg.name, alg.data])
        
        return fix_from_traj
    
    @property
    def steps(self):
        """Gathers a list of which algebraics are step variables
        
        :return set steps: A set of step variables
        :rtype: set
        
        """
        steps = {}
        for alg in self.algebraics.values():
            if alg.step is not None:
                steps[alg.name] = alg
        
        return steps


class ComponentBlock(ModelElementBlock):
    
    """Class to hold ModelElements of type constant and provide specific
    methods.

    :Methods:

        - :func:`var_variances`
        - :func:`component_set`
        - :func:`variances`
        - :func:`init_values`
        - :func:`known_values`
        - :func:`has_all_variances`

    """
    def __init__(self, *args, **kwargs):
        super().__init__(class_name='model_component')

    def var_variances(self):
        """Returns a dict of component variances

         :return: dict of variances
         :rtype: dict

         """
        sigma_dict = {}
        
        for component in self.components.values():       
            if component.state == 'trajectory':
                continue       
            sigma_dict[component.name] = component.variance
            
        return sigma_dict
        
    def component_set(self, category):
        """Returns a list of component categories

        :param str category: A component category

         :return: list of states with the given category
         :rtype: list

         """
        component_set = []
        for component in self.components.values():
            if component.state == category:
                component_set.append(component.name)
                
        return component_set
    
    @property
    def variances(self):
        """Returns a dict of component variances

         :return: dict of known states
         :rtype: dict

         """
        return {comp.name: comp.variance for comp in self.components.values()}
    
    @property
    def init_values(self):
        """Returns a dict of initial component values

        :return: dict of initial values
        :rtype: dict

        """
        return {comp.name: comp.value for comp in self.components.values()}
    
    @property
    def known_values(self):
        """Returns a dict of components that have known values (fixed)

        :return: dict of known components
        :rtype: dict

        """
        return {comp.name: comp.known for comp in self.components.values()}
    
    @property
    def has_all_variances(self):
        """Returns True if all components have variances

        :return all_component_variances: Boolean indicating if all state variances are provided
        :rtype: bool

        """
        all_component_variances = True
        for comp in self.components.values():
            if comp.variance is None:
                all_component_variances = False
            if not all_component_variances:
                break
        return all_component_variances


class StateBlock(ModelElementBlock):
    
    """Class to hold ModelElements of type constant and provide specific
    methods.

    :Methods:

        - :func:`var_variances`
        - :func:`component_set`
        - :func:`variances`
        - :func:`init_values`
        - :func:`known_values`
        - :func:`has_all_variances`

    """
    __var = VariableNames()
    
    def __init__(self, *args, **kwargs):
        super().__init__(class_name='model_state')

    def var_variances(self):
        """Returns a dict of state variances

         :return: dict of variances
         :rtype: dict

         """
        sigma_dict = {}
        
        for component in self.states.values():       
            if component.state == 'trajectory':
                continue       
            sigma_dict[component.name] = component.variance
            
        return sigma_dict
        
    def component_set(self, category):
        """Returns a list of state categories

        :param str category: A state category

         :return: list of states with the given category
         :rtype: list

         """
        component_set = []
        for component in self.states.values():
            if component.state == category:
                component_set.append(component.name)
                
        return component_set
    
    @property
    def variances(self):
        """Returns a dict of state variances

         :return: dict of known states
         :rtype: dict

         """
        return {comp.name: comp.variance for comp in self.states.values()}
    
    @property
    def init_values(self):
        """Returns a dict of initial state values

        :return: dict of initial values
        :rtype: dict

        """
        return {comp.name: comp.value for comp in self.states.values()}
    
    @property
    def known_values(self):
        """Returns a dict of states that have known values (fixed)

        :return: dict of known states
        :rtype: dict

        """
        return {comp.name: comp.known for comp in self.states.values()}
    
    @property
    def has_all_variances(self):
        """Returns True if all states have variances

        :return all_component_variances: Boolean indicating if all state variances are provided
        :rtype: bool

        """
        all_component_variances = True
        for comp in self.states.values():
            if comp.variance is None:
                all_component_variances = False
            if not all_component_variances:
                break
        return all_component_variances
    
    @property
    def fixed(self):
        
        fix_from_traj = []
        for state in self.states.values():
            if state.data is not None:
                fix_from_traj.append([self.__var.state_model, state.name, state.data])
        
        return fix_from_traj
    
    
class ParameterBlock(ModelElementBlock):
    
    """Class to hold ModelElements of type constant and provide specific
    methods.
    
    """
    def __init__(self, *args, **kwargs):
        super().__init__(class_name='model_parameter')
    
    @property
    def lb(self):
        """Lower bound property

        :return: lower bound of the parameter
        :rtype: float

        """
        return self.bounds[0]

    @property
    def ub(self):
        """Upper bound property

        :return: upper bound of the parameter
        :rtype: float

        """
        return self.bounds[1]
