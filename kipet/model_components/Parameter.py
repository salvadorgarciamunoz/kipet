"""
initialization of dummy pyomo variables for use in model development

This should be updated to a more robust framework, eventually
"""
from pint import UnitRegistry
from pyomo.environ import ConcreteModel, Set, Var

class Component():
    """A generic pyomo variable to use in model building"""

    m = ConcreteModel()        
    m.indx = Set(initialize=[0])
    u = UnitRegistry()

    def __new__(cls,
                name,
                index=1,
                units=None):

        m = cls.m
        u = cls.u
        
        sets = [m.indx]*index
        m_units = u(units).units
        if m_units.dimensionless:
            m_units = str(1)
        else:
            m_units = getattr(u, str(m_units))
        
        setattr(m, name, Var(*sets, initialize=1, units=units))
        self = getattr(m, name)
        return self[tuple([0 for i in range(index)])]
    
class Parameter():
    """A `Symbol` representing a parameter whose value is known."""

    __qualname__ = 'Parameter'

    def __init__(self, 
                 name, 
                 value=0, 
                 index=1, 
                 units=None):
        
        self.name = name
        self.value = value
        self.index = index
        self.units = units
    
        self.Var = Component(name, index, units)

    # #def __new__(cls, name, value=0, index=1, units=None, parent=None):
    #     # self = Component.__new__(cls, name, index, units)
    #     # # self = super().__new__(cls, name)

    #     # self.value = value  # TODO: Add safeguards
    #     # self.units = units
    #     # return self

    # @property
    # def indexed(self):
    #     """Check whether the Parameter is indexed or not."""
    #     return self.expansion is not None

    # @property
    # def value(self):
    #     """Return the value or values of the Parameter."""
    #     return self._value

    # @value.setter
    # def value(self, data):
    #     """Set the value of the Parameter.

    #     In contrast to the Variable, a Parameter can be made indexed by
    #     simply providing some Mapping or a pandas.Series that imply
    #     both an index and values.
    #     """
    #     if isinstance(data, (Mapping, Series)):
    #         self.expand(data)
    #         self._value = None
    #     else:
    #         self._value = None if data is None else float(data)
    #         self.expansion = None
            
    # # @units.setter
    # # def units(self, data):
    # #     """Set the value of the Parameter.

    # #     In contrast to the Variable, a Parameter can be made indexed by
    # #     simply providing some Mapping or a pandas.Series that imply
    # #     both an index and values.
    # #     """
    # #     if data is not None:
    # #         self._units = cls.u(data)
    # #     else:
    # #         self._units = None
        
    # @property
    # def units(self):
    #     return self._units.units
        
    # @property
    # def indices(self): return self.expansion.keys()

    # @property
    # def elements(self): return self.expansion.values

    # @property
    # def items(self): return self.expansion.items()

    # @property
    # def parent(self):
    #     """Return the parent of this parameter."""
    #     return self._parent

    # def expand(self, data):
    #     """Expand the `Parameter` with indexed data."""
    #     if self._parent is not None:
    #         raise RuntimeError(f'Attempted to expand parameter {self} '
    #                            f'which is a member of {self._parent}!')
    #     self.expansion = Series((Parameter(f'{self.name}[{i}]', v,
    #                                        self)
    #                              for i, v in data.items()),
    #                             data.keys(), dtype='O')

    # def __getitem__(self, index):
    #     try:  # TODO: Custom errors still includes long stack trace...
    #         return self.expansion[index]
    #     except KeyError:
    #         if self.expansion is None:
    #             raise TypeError(f'Parameter {self.name} is not '
    #                             'indexed!')
    #         raise IndexError("Parameter index out of range")

    # def __setitem__(self, index, value):
    #     """Set the value of the element corresponding to the index."""
    #     self.expansion[index].value = value

    # def __iter__(self):
    #     """Iterate over the elements of this Parameter."""
    #     if self.expansion is None:
    #         raise TypeError(f'{self} is scalar.')
    #     for elem in self.elements:
    #         yield elem

    # @property
    # def is_Parameter(self):
    #     return True

if __name__ == '__main__':
    
    a = Parameter('k', value=1, units='1/s')
    b = Parameter('A', 10)