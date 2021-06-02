"""
Expression Classes
"""
# Third party imports
from pyomo.core.expr.numeric_expr import (DivisionExpression,
                                          NegationExpression,
                                          ProductExpression)
from pyomo.environ import units as pyo_units

# KIPET library imports
from kipet.model_tools.replacement import _update_expression


class ExpressionBlock:
    
    """Class for general expression block classes

    :Methods:

        - :func:`display`
        - :func:`display_units`
    """
    
    def __init__(self, exprs=None):
        
        self.exprs = exprs
        self._title = 'EXPR'

    def __len__(self):
        
        return len(self.exprs)
        
    def display(self):
        """This method loops through all expressions and displays them in a readable format

        :return: None

        """
        if self.exprs is not None:
            print(f'{self._title} expressions:')
            for key, expr in self.exprs.items():
                print(f'{key}: {expr.expression.to_string()}')

        return None

    def display_units(self):
        """Loops through the expressions and shows their units

        :return: None

        """
        margin = 8
        if self.exprs is not None:
            print(f'{self._title} units:')
            for key, expr in self.exprs.items():
                print(f'{str(key).rjust(margin)} : {expr.units}')

        return None


class ODEExpressions(ExpressionBlock):
    
    """Class for ODE expressions

    :param dict ode_exprs: dict of ODE expressions

    """
    
    def __init__(self, ode_exprs=None):
        
        super().__init__(ode_exprs)
        self._title = 'ODE'
        
        
class AEExpressions(ExpressionBlock):
    
    """Class for AE expressions

    :param dict alg_exprs: dict of alebraic expressions

    """
    
    def __init__(self, alg_exprs=None):
        
        super().__init__(alg_exprs)
        self._title = 'ALG'


class Expression:
    
    """Class for individual expressions

    :param str name: The name of the expression
    :param expression: Pyomo expression

    :Methods:

        - :func:`show_units`
        - :func:`check_units`
        - :func:`check_division`
        - :func:`check_expression_units`

    """
    
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
        """Display the units of the expression

        :return: String of the units
        :rtype: str

        """
        return self.units.to_string()
    
    def _change_to_unit(self, c_mod, c_mod_new):
        """Method to remove the fixed parameters from the ConcreteModel

        :param Comp c_mod: The Comp object of dummy pyomo variables
        :param Comp c_mod_new: The Comp object of new variables

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

    def check_units(self):
        """Check the expr units by exchanging the real model with unit model
        components
        
        :return: None

        """
        self.units = pyo_units.get_units(self.expression)
        return None

    def check_division(self, eps=1e-12):
        """Add a small amount to the numerator and denominator in a
        DivisionExpression to improve the numerics.

        :param float eps: The small addition to the numerator and denominator

        :return: None

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

    @staticmethod
    def _check_term(term, convert_to):
        """This loops through terms in the expression to ensure the units are valid

        :param expression term: The term to check
        :param str convert_to: The units to convert to, if necessary

        :return term_new: The updated expression term
        :rtype: expression

        """
        unit_term = pyo_units.get_units(term)
        term_new = term
        
        if unit_term is not None:
            term_new = pyo_units.convert(term, to_units=getattr(pyo_units, convert_to))
            
        return term_new
    
    def check_expression_units(self, convert_to=None):
        """Method to call in order to check an expressions units. This updates the expressions in place.

        :param str convert_to: The units to convert the expression to

        :return: None

        """
        scalar = 1
        if convert_to is None:
            raise ValueError('You need to supply a conversion')
        expr = self.expression
        expr_new = 0
        if isinstance(expr, NegationExpression):
            scalar *= -1
        if isinstance(self.expression, (DivisionExpression, ProductExpression)):
            term = self._check_term(expr, convert_to)
            expr_new = scalar*term
        else:
            if isinstance(expr, (int, float)):
                return None
            for i, term in enumerate(expr.args):
                term = self._check_term(term, convert_to)
                expr_new += scalar*term
        self.expression = expr_new
        self.units = getattr(pyo_units, convert_to)
             
        return None
