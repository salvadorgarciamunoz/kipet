from ModelBuilderHelper import *
from casadi.tools import *
import casadi as ca
import copy

#from CasadiSimulator import *

class CasadiModel(object):
    pass

class CasadiModelBuilderHelper(ModelBuilderHelper):
    def __init__(self):
        super(CasadiModelBuilderHelper, self).__init__()

    def create_casadi_model(self,start_time,end_time,fixed_dict=None):
        
        # Model
        casadi_model = CasadiModel()
        
        # Sets
        casadi_model.mixture_component_names = copy.deepcopy(self._component_names)
        casadi_model.parameter_names = copy.deepcopy(self._paramemter_names)
        
        # Variables        
        casadi_model.C = struct_symSX(list(casadi_model.mixture_component_names))
        casadi_model.kinetic_parameter = struct_symSX(list(casadi_model.parameter_names))
            
        # Parameters
        casadi_model.init_conditions = self._init_conditions
        casadi_model.start_time = start_time
        casadi_model.end_time = end_time
        
        return casadi_model

if __name__ == "__main__":

    helper2 = CasadiModelBuilderHelper()    
    helper2.add_mixture_component('A',1)
    helper2.add_mixture_component('B',0)
    helper2.add_kinetic_parameter('k')

    casadi_model = helper2.create_casadi_model(0.0,200.0)
    
    casadi_model.diff_exprs = dict()
    casadi_model.diff_exprs['A'] = -casadi_model.kinetic_parameter['k']*casadi_model.C['A']
    casadi_model.diff_exprs['B'] = casadi_model.kinetic_parameter['k']*casadi_model.C['A']


    # fixes parameters
    fixed_params = dict()
    fixed_params[casadi_model.kinetic_parameter] = 0.01
    
    for key,val in casadi_model.diff_exprs.iteritems():
        for key1,val1 in fixed_params.iteritems():
            casadi_model.diff_exprs[key] = ca.substitute(val,key1,val1)

    print casadi_model.diff_exprs

    
    
    
    
