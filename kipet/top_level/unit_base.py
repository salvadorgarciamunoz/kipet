"""
Unit Base - Object for holding units and the unit registry
"""
import pint


class UnitBase():
    
    def __init__(self):
                
        self.ur = pint.UnitRegistry()
        self.time = 'min'
        self.volume = 'L'
        self.concentration = 'M'