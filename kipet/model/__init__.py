import imp
try:
    imp.find_module('casadi')
    from CasadiModel import *
    found_casadi=True
except ImportError:
    found_casadi=False

if found_casadi:
    __all__ = ['CasadiModel.py','TemplateBuilder']
else:
    __all__ = ['TemplateBuilder']

    
