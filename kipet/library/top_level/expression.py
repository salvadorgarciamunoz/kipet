"""
Expression Classes
"""
from kipet.library.common.VisitorClasses import ReplacementVisitor
from pyomo.environ import ConcreteModel, Var, Objective
from pyomo.environ import units as u
from pyomo.util.check_units import assert_units_consistent, assert_units_equivalent, check_units_equivalent
from kipet.library.post_model_build.replacement import _update_expression

class ExpressionBlock():
    
    """Class for general expression block classes"""
    
    def __init__(self,
                 exprs=None,
                 ):
        
        self.exprs = exprs
        self._title = 'EXPR'
        
    def display(self):
        
        if self.exprs is not None:
            print(f'{self._title} expressions:')
            for key, expr in self.exprs.items():
                print(f'{key}: {expr.expression.to_string()}')


    def display_units(self):

        if self.exprs is not None:
            print(f'{self._title} units:')
            for key, expr in self.exprs.items():
                print(f'{key}: {expr.units}')
    

class ODEExpressions(ExpressionBlock):
    
    """Class for ODE expressions"""
    
    def __init__(self,
                 ode_exprs=None,
                 ):
        
        super().__init__(ode_exprs)
        self._title = 'ODE'
        
        
class AEExpressions(ExpressionBlock):
    
    """Class for AE expressions"""
    
    def __init__(self,
                 alg_exprs=None,
                 ):
        
        super().__init__(alg_exprs)
        self._title = 'ALG'

class Expression():
    
    """Class for individual expressions"""
    
    def __init__(self,
                 name,
                 expression):
        
        self.name = name
        self.expression = expression
        self.units = None
        
    def __str__(self):
        
        return self.expression.to_string()
    
    @property
    def show_units(self):
        return self.units.to_string()
    
    def _change_to_unit(self, c_mod, c_mod_new):
        """Method to remove the fixed parameters from the ConcreteModel
        TODO: move to a new expression class
        """
        var_dict = c_mod_new.var_dict
        expr_new = self.expression
        for model_var, var_obj in var_dict.items():
            old_var = getattr(c_mod, model_var)
            new_var = getattr(c_mod_new, model_var)         
            expr_new = _update_expression(expr_new, old_var, new_var)
            
        return expr_new

    def check_units(self, c_mod, c_mod_new):
        """Check the expr units by exchanging the real model with unit model
        components
        
        Args:
            key (str): component represented in ODE
            
            expr (Expression): Expression object of ODE
            
            c_mod (Comp): original Comp object used to declare the expressions
            
            c_mod_new (Comp_Check): dummy model with unit components
            
        Returns:
            pint_units (): Returns expression with unit model components
        
        """
        expr = self._change_to_unit(c_mod, c_mod_new)
        self.units = u.get_units(expr)
        return None

    
    