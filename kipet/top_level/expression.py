"""
Expression Classes
"""
# Standad library imports

# Third party imports
from pyomo.core.expr.numeric_expr import DivisionExpression, ProductExpression, NegationExpression
from pyomo.environ import (
    ConcreteModel,
    Objective,
    Var, 
    )
from pyomo.environ import units as pyo_units

# KIPET library imports
from kipet.common.VisitorClasses import ReplacementVisitor
from kipet.post_model_build.replacement import _update_expression

from kipet.dev_tools.display import Print

DEBUG = False

_print = Print(verbose=DEBUG)


class ExpressionBlock():
    
    """Class for general expression block classes"""
    
    def __init__(self,
                 exprs=None,
                 ):
        
        self.exprs = exprs
        self._title = 'EXPR'

    def __len__(self):
        
        return len(self.exprs)
        
    def display(self):
        
        if self.exprs is not None:
            print(f'{self._title} expressions:')
            for key, expr in self.exprs.items():
                print(f'{key}: {expr.expression.to_string()}')


    def display_units(self):

        margin = 8
        if self.exprs is not None:
            print(f'{self._title} units:')
            for key, expr in self.exprs.items():
                #print(f'  {key}: {expr.units}')
                print(f'{str(key).rjust(margin)} : {expr.units}')
    

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
        self.expression_orig = None
        
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
        if self.expression_orig is not None:
            expr_new = self.expression_orig    
        for model_var, var_obj in var_dict.items():
            old_var = c_mod[model_var][1]
            new_var = var_dict[model_var]
            expr_new = _update_expression(expr_new, old_var, new_var)
        return expr_new

    def check_units(self):#, c_mod, c_mod_new):
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
        self.units = pyo_units.get_units(self.expression)
        return None

    def check_division(self, eps=1e-12):
        """Add a small amount to the numerator and denominator in a
        DivisionExpression to improve the numerics
        
        """
        expr = self.expression
        
        if isinstance(expr, DivisionExpression):
        
            ex1, ex2 = expr.args
            ex1 += eps
            ex2 += eps
            expr_new = ex1/ex2
            
            self.expression = expr_new
            self.expression_orig = expr
    
        return None
    
    #Consolidate the units per expression - avoid checking the individual units
    
    def check_term(self, term, convert_to):
        
        unit_term = pyo_units.get_units(term)
        _print(f'     Units: {unit_term} ==> {convert_to}\n')
        _changed_flag = False
        term_new = term
        
        if unit_term is not None:
            _print('Starting conversion and making the new term\n')
            term_new = pyo_units.convert(term, to_units=getattr(pyo_units, convert_to))
            
        return term_new
    
    def check_expression_units(self, convert_to=None, scalar=1):
        
        if convert_to is None:
            raise ValueError('You need to supply a conversion')
        
        expr = self.expression
        expr_new = 0
        
        _print(f'The ODE expression being handled is:\n     {expr}\n')
        _print(f'The type of expression being handled is:\n     {type(expr)}\n')
    
        if isinstance(expr, NegationExpression):
            scalar *= -1
    
        if isinstance(self.expression, (DivisionExpression, ProductExpression)):
            _print(f'The number of terms in this expression is: 1\n')
            
            term = self.check_term(expr, convert_to)
            expr_new = scalar*term
        
        
        else:
            _print(f'The number of terms in this expression is: {len(expr.args)}\n')
            
            for i, term in enumerate(expr.args):
                _print(f'({i+1}) The expression term considered here is\n     {term}\n')
                term = self.check_term(term, convert_to)
                expr_new += scalar*term
        
        _print('The new expression is:\n')
        _print(f'     {expr_new}')
            
        self.expression = expr_new
        self.units = getattr(pyo_units, convert_to)
             
        return None
        
            
            
    
    
    
    
    
    
    
    
    