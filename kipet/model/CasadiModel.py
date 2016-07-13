from casadi.tools import *
import casadi as ca


class CasadiModel(object):
    """Base casadi model class.

    Attributes:

        odes (dict): map from state name to SX expression
        that represents the ordinary differential equation
        
        alg_exprs (dict): currently not used.  

    """
    def __init__(self):
        self.odes = dict()
        self.alg_exprs = dict()

class KipetCasadiStruct(object):
    """Structure used for representing kipet casadi variables.

    Attributes:

        dummy_index (bool): flag to treat time indices similarly to pyomo 
        
        _true_indices (list): list of indices to access variables in the structure

        _symbolics (dict): map of index to SX variable

    """
    def __init__(self,name,list_index,dummy_index=False):
        self._dummy_index = dummy_index
        self._true_indices = [i for i in list_index]
        self._symbolics = dict()
        for i in self._true_indices:
            self._symbolics[i] = ca.SX.sym("{0}[{1}]".format(name,i))
        
    
    def __getitem__(self,index):
        if isinstance(index,tuple):
            # ignore first index
            if self._dummy_index:
                return self._symbolics[index[1]]
            else:
                return self._symbolics[index]
        else:
            return self._symbolics[index]
    
    def __setitem__(self,index,val):
        if isinstance(index,tuple):
            if self._dummy_index:
                self._symbolics[index[1]] = val
            else:
                self._symbolics[index] = val
        else:
            self._symbolics[index] = val
