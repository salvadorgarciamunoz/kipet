import imp
try:
    imp.find_module('casadi')
    from CasadiModel import *
    found_casadi=True
except ImportError:
    found_casadi=False

if found_casadi:
    __all__ = ['CasadiSimulator','PyomoSimulator','ResultsObject','Simulator']
else:
    __all__ = ['PyomoSimulator','ResultsObject','Simulator']
