import six

class ModelBuilderHelper(object):
    
    def __init__(self):
        self._component_names = set()
        self._paramemter_names = set()
        self._init_conditions = dict()
        
    def add_kinetic_parameter(self,name):
        if isinstance(name,six.string_types):
            self._paramemter_names.add(name)
        elif isinstance(name,list) or isinstance(name,set):
            for n in name:
                self._paramemter_names.add(name)
        else:
            raise RuntimeError('Need a string or a list of strings')
        
    def add_mixture_component(self,*args):
        if len(args) == 1:
            input = args[0]
            if isinstance(input,dict):
                for key,val in input.iteritems():
                    self._component_names.add(key)
                    self._init_conditions[key] = val
            else:
                raise RuntimeError('Need a string and an initial condition')
        elif len(args)==2:
            name = args[0]
            init_condition = args[1]
            if isinstance(name,six.string_types):
                self._component_names.add(name)
                self._init_conditions[name] = init_condition
            else:
                raise RuntimeError('Need a string and an initial condition')
        else:
            print len(args)
            raise RuntimeError('Need a string and an initial condition')
            
