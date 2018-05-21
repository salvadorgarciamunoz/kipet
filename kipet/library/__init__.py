# -*- coding: utf-8 -*- 
import sys 
try: 
    if sys.version_info.major > 3: 
        import importlib 
        importlib.util.find_spec("casadi") 
    else: 
        import imp 
        imp.find_module('casadi') 
    from kipet.library.CasadiModel import CasadiModel 
    from kipet.library.CasadiModel import KipetCasadiStruct 
    found_casadi = True 
except ImportError: 
    found_casadi = False 
 
 
if found_casadi: 
    __all__ = ['CasadiModel.py','TemplateBuilder','BaseAbstractModel','CasadiSimulator',
               'data_tools','fe_factory','Optimizer','ParameterEstimator','PyomoSimulator',
               'ResultsObject','Simulator','VarianceEstimator'] 
else: 
    __all__ = ['TemplateBuilder','BaseAbstractModel','CasadiSimulator',
               'data_tools','fe_factory','Optimizer','ParameterEstimator',
               'PyomoSimulator','ResultsObject','Simulator','VarianceEstimator']  
