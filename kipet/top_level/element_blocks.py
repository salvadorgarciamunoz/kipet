"""
ModelElement Blocks
"""
import kipet.model_components.ModelComponent as model_components
from kipet.top_level.variable_names import VariableNames

class ModelElementBlock():
    
    """Data abstraction for multiple ModelElement instances"""
    
    def __init__(self, class_name):
        
        attr_name = class_name.split('_')[-1] + 's'
        self.attr_class_set_name = attr_name
        setattr(self, attr_name, {})
        
        self.element_object_name = ''.join([term.capitalize() for term in class_name.split('_')])
        self.element_object = getattr(model_components, self.element_object_name)
        
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
        # return self.element_object_name

    def __iter__(self):
        for param, data in getattr(self, self.attr_class_set_name).items():
            yield data
            
    def __len__(self):
        return len(getattr(self, self.attr_class_set_name))
    
    def __contains__(self, key):
        return key in self._dict
    

    def add_element_list(self, elem_list):
        """Handles lists of parameters or single parameters added to the model
       
        """
        for elem in elem_list:
            self.add_element(*elem)        
        
        return None
    
    def add_element(self, *args, **kwargs):
        
        """
            
        """
        name = args[0]
        element = self.element_object(name,
                                      **kwargs,
                                      )
            
        getattr(self, self.attr_class_set_name)[element.name] = element
        
    def as_dict(self, attr):
        
        return_dict = {}
        for param, obj in self._dict.items():
            return_dict[param] = getattr(obj, attr)
        return return_dict
        
    def update(self, attr, dict_data):
        
        for elem, new_data in dict_data.items():
            if elem in self._dict:
                setattr(self._dict[elem], attr, new_data)

        return None
    
    def get_match(self, attr, query):
        
        query_list = []
        for elem, obj in self._dict.items():
            if getattr(obj, attr) == query:
                query_list.append(elem)
                
        return query_list
    
    
    @property 
    def names(self):
        return [elem for elem in getattr(self, self.attr_class_set_name)]
    
    
class ConstantBlock(ModelElementBlock):
    
    def __init__(self, *args, **kwargs):
        super().__init__(class_name='model_constant')
        
class AlgebraicBlock(ModelElementBlock):
    
    __var = VariableNames()
    
    def __init__(self, *args, **kwargs):
        super().__init__(class_name='model_algebraic')
       
    @property
    def fixed(self):
        
        fix_from_traj = []
        for alg in self.algebraics.values():
            if alg.data is not None:
                fix_from_traj.append([self.__var.algebraic, alg.name, alg.data])
        
        return fix_from_traj
    
    @property
    def steps(self):
        
        steps = {}
        for alg in self.algebraics.values():
            if alg.step is not None:
                steps[alg.name] = alg
        
        return steps
        
class ComponentBlock(ModelElementBlock):
    
    def __init__(self, *args, **kwargs):
        super().__init__(class_name='model_component')
    
    
    def var_variances(self):
        
        sigma_dict = {}
        
        for component in self.components.values():       
            if component.state == 'trajectory':
                continue       
            sigma_dict[component.name] = component.variance
            
        return sigma_dict
        
    def component_set(self, category):
        
        component_set = []
        for component in self.components.values():
            if component.state == category:
                component_set.append(component.name)
                
        return component_set
    
    @property
    def variances(self):
        return {comp.name: comp.variance for comp in self.components.values()}
    
    @property
    def init_values(self):
        return {comp.name: comp.value for comp in self.components.values()}
    
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
    
class StateBlock(ModelElementBlock):
    
    def __init__(self, *args, **kwargs):
        super().__init__(class_name='model_state')
    
    
    def var_variances(self):
        
        sigma_dict = {}
        
        for component in self.states.values():       
            if component.state == 'trajectory':
                continue       
            sigma_dict[component.name] = component.variance
            
        return sigma_dict
        
    def component_set(self, category):
        
        component_set = []
        for component in self.states.values():
            if component.state == category:
                component_set.append(component.name)
                
        return component_set
    
    @property
    def variances(self):
        return {comp.name: comp.variance for comp in self.states.values()}
    
    @property
    def init_values(self):
        return {comp.name: comp.value for comp in self.states.values()}
    
    @property
    def known_values(self):
        return {comp.name: comp.known for comp in self.states.values()}
        
    @property
    def names(self):
        return [comp.name for comp in self.states.values()]
    
    @property
    def has_all_variances(self):
    
        all_component_variances = True
        for comp in self.components.values():
            if comp.variance is None:
                all_component_variances = False
            if not all_component_variances:
                break
        return all_component_variances
    
    
class ParameterBlock(ModelElementBlock):
    
    def __init__(self, *args, **kwargs):
        super().__init__(class_name='model_parameter')
    
    @property
    def lb(self):
        """Lower bound property"""
        return self.bounds[0]

    @property
    def ub(self):
        """Upper bound property"""
        return self.bounds[1]