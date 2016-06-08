import six
import pandas as pd

class ModelBuilderHelper(object):
    
    def __init__(self):
        self._component_names = set()
        self._parameters = dict()
        self._init_conditions = dict()
        self._spectral_data = None
        
    def add_parameter(self,*args):
        if len(args) == 1:
            name = args[0]
            if isinstance(name,six.string_types):
                self._parameters[name] = None
            elif isinstance(name,list) or isinstance(name,set):
                for n in name:
                    self._parameters[n] = None
            elif isinstance(name,dict):
                for k,v in name.iteritems():
                    self._parameters[k] = v
            else:
                raise RuntimeError('Kinetic parameter data not supported. Try str')
        elif len(args) == 2:
            first = args[0]
            second = args[1]
            if isinstance(first,six.string_types):
                self._parameters[first] = second
            else:
                raise RuntimeError('Parameter argument not supported. Try str,val')
        else:
            raise RuntimeError('Parameter argument not supported. Try str,val')
        
    def add_mixture_component(self,*args):
        if len(args) == 1:
            input = args[0]
            if isinstance(input,dict):
                for key,val in input.iteritems():
                    self._component_names.add(key)
                    self._init_conditions[key] = val
            else:
                raise RuntimeError('Mixture component data not supported. Try dict[str]=float')
        elif len(args)==2:
            name = args[0]
            init_condition = args[1]
            if isinstance(name,six.string_types):
                self._component_names.add(name)
                self._init_conditions[name] = init_condition
            else:
                raise RuntimeError('Mixture component data not supported. Try str, float')
        else:
            print len(args)
            raise RuntimeError('Mixture component data not supported. Try str, float')
            
    def add_spectral_data(self,data):
        if isinstance(data,pd.DataFrame):
            self._spectral_data = data
        else:
            raise RuntimeError('Spectral data format not supported. Try pandas.DataFrame')
        """
        if len(args)==1:
            data = args[0]
            if isinstance(data,dict):
                for key,val in data.iteritems():
                    if isinstance(key,six.string_types) and isinstance(val,pd.DataFrame):
                        self._spectral_data_sets[key] = val
                    else:
                        raise RuntimeError('Spectral data format not supported. Try dict[str] = pandas.DataFrame')
        elif len(args)==2:
            data = args[1]
            name = args[0]
            if isinstance(name,six.string_types) and isinstance(data,pd.DataFrame):
                self._spectral_data_sets[name] = data
            else:
                raise RuntimeError('Spectral data format not supported. Try srt,pandas.DataFrame')
        else:
            raise RuntimeError('Spectral data format not supported. Try srt,pandas.DataFrame')
        """         
    def create_model(self,start_time,end_time):
        raise NotImplementedError("Model bulder abstract method. Call child class")
