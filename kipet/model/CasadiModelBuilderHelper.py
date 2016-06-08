from ModelBuilderHelper import *
from casadi.tools import *
import casadi as ca
import copy

#from CasadiSimulator import *

class CasadiModel(object):
    def __init__(self):
        self.diff_exprs = dict()
        self.alg_exprs = dict()

class KinetCasadiStruct(object):
    def __init__(self,name,list_index):
        self._true_indices = [i for i in list_index]
        self._str_indices = [[str(i) for i in list_index]]
        self._symbolics = dict()
        for i in self._true_indices:
            self._symbolics[i] = ca.SX.sym("{0}[{1}]]".format(name,i))
    
    def __getitem__(self,index):
        return self._symbolics[index]
    
    def __setitem__(self,index,val):
        self._symbolics[index] = val

def fix_casadi_parameters(model,fixed_params):
    for key,val in model.diff_exprs.iteritems():
        for key1,val1 in fixed_params.iteritems():
            model.diff_exprs[key] = ca.substitute(val,key1,val1)

class CasadiModelBuilderHelper(ModelBuilderHelper):
    def __init__(self):
        super(CasadiModelBuilderHelper, self).__init__()

    def create_model(self,start_time,end_time):
        
        # Model
        casadi_model = CasadiModel()
        
        # Sets
        casadi_model.mixture_components = copy.deepcopy(self._component_names)
        casadi_model.parameter_names = self._parameters.keys()
        m_times = list()
        m_lambdas = list()
        if self._spectral_data is not None:
            m_times = self._spectral_data.index
            m_lambdas = self._spectral_data.columns
        casadi_model.measurement_times = m_times
        casadi_model.measurement_lambdas = m_lambdas
        
        # Variables        
        #casadi_model.C = struct_symSX(list(casadi_model.mixture_components))
        #casadi_model.P = struct_symSX(list(casadi_model.parameter_names))
        
        casadi_model.C = KinetCasadiStruct('C',list(casadi_model.mixture_components))
        casadi_model.P = KinetCasadiStruct('P',list(casadi_model.parameter_names))
        
        # Parameters
        casadi_model.init_conditions = self._init_conditions
        casadi_model.start_time = start_time
        casadi_model.end_time = end_time
        
        # Fixes parameters that were given numeric values
        for p,v in self._parameters.iteritems():
            if v is not None:
                casadi_model.P[p] = v
        
        return casadi_model

if __name__ == "__main__":

    builder = CasadiModelBuilderHelper()    
    builder.add_mixture_component('A',1)
    builder.add_mixture_component('B',0)
    builder.add_parameter('k',0.01)

    casadi_model = builder.create_model(0.0,200.0)
    
    casadi_model.diff_exprs = dict()
    casadi_model.diff_exprs['A'] = -casadi_model.P['k']*casadi_model.C['A']
    casadi_model.diff_exprs['B'] = casadi_model.P['k']*casadi_model.C['A']
    
    print casadi_model.diff_exprs
    
    
    
