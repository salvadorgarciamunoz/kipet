"""
Unit Base
"""
import pint

class UnitBase():
    
    def __init__(self):
                
        self.ur = pint.UnitRegistry()
        self.VOLUME_BASE = 'L'
        self.TIME_BASE = 'min'