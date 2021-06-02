"""
Replacement and Scaling Mixins used in model modification
"""
from pyomo.core.base.units_container import _PyomoUnit
from pyomo.core.expr import current as EXPR
from pyomo.core.expr.numvalue import NumericConstant
from pyomo.environ import *


class ReplacementVisitor(EXPR.ExpressionReplacementVisitor):

    """Class used to replace elements within a Pyomo expression"""

    def __init__(self):
        super(ReplacementVisitor, self).__init__()
        self._replacement = None
        self._suspect = None

    def change_suspect(self, suspect_):
        self._suspect = suspect_

    def change_replacement(self, replacement_):
        self._replacement = replacement_

    def visiting_potential_leaf(self, node):
        #
        # Clone leaf nodes in the expression tree
        #
        #print(node)
        #print(type(node))
        
        
        if node.__class__ is _PyomoUnit:
            return True, node
        
        if node.__class__ in native_numeric_types:
            return True, node

        if node.__class__ is NumericConstant:
            return True, node


        if node.is_variable_type():
            if id(node) == self._suspect:
                d = self._replacement
                return True, d
            else:
                return True, node
            
        if node.is_parameter_type():
            if id(node) == self._suspect:
                d = self._replacement
                return True, d
            else:
                return True, node

        return False, None


class ScalingVisitor(EXPR.ExpressionReplacementVisitor):

    """Class used to replace elements in a Pyomo expression with scaled values"""

    def __init__(self, scale):
        super(ScalingVisitor, self).__init__()
        self.scale = scale

    def visiting_potential_leaf(self, node):
      
        if node.__class__ is _PyomoUnit:
            return True, node  
      
        if node.__class__ in native_numeric_types:
            return True, node

        if node.is_variable_type():
            #print(node)
            return True, self.scale[id(node)]*node

        if isinstance(node, EXPR.LinearExpression):
            node_ = copy.deepcopy(node)
            node_.constant = node.constant
            node_.linear_vars = copy.copy(node.linear_vars)
            node_.linear_coefs = []
            for i,v in enumerate(node.linear_vars):
                node_.linear_coefs.append( node.linear_coefs[i]*self.scale[id(v)] )
            return True, node_

        return False, None


class FindingVisitor(EXPR.ExpressionValueVisitor):

    """Class used simply to find values within Pyomo expressions"""

    def __init__(self):
        super(FindingVisitor, self).__init__()
        self._suspect = None
        self._found = False

    def find_suspect(self, suspect_):
        self._suspect = suspect_

    def visiting_potential_leaf(self, node):
        """Clone leaf in expression tree
        
        """
        if id(node) == self._suspect:
             self._found = True
  
        if node.__class__ in native_numeric_types:
            return True, node

        if node.__class__ is NumericConstant:
            return True, node

        if node.is_variable_type():
            return True, node
            
        if node.is_parameter_type():
            return True, node
            
        return False, None
