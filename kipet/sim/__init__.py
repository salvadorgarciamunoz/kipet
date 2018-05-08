# -*- coding: utf-8 -*-
import sys
try:
    if sys.version_info.major > 3:
        import importlib
        importlib.util.find_spec("casadi")
    else:
        import imp
        imp.find_module('casadi')
    from kipet.model.CasadiModel import CasadiModel
    from kipet.model.CasadiModel import KipetCasadiStruct
    found_casadi = True
except ImportError:
    found_casadi = False


if found_casadi:
    __all__ = ['CasadiSimulator','PyomoSimulator','ResultsObject','Simulator']
else:
    __all__ = ['PyomoSimulator','ResultsObject','Simulator']
