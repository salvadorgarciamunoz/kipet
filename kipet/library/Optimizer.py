"""
Base Optimizer Class

Either move more methods from child classes or delete - go directly to PS
"""
from pyomo.environ import (
    Suffix,
    )

from kipet.library.PyomoSimulator import *


class Optimizer(PyomoSimulator):
    """Base optimizer class.

    Note:
        This class is not intended to be used directly by users

    Attributes:
        model (model): Pyomo model.

    """
    def __init__(self, model):
        """Optimizer constructor.

        Note: 
            Makes a shallow copy to the model. Changes applied to 
            the model within the simulator are applied to the original
            model passed to the simulator

        Args:
            model (Pyomo model)
        """
        super(Optimizer, self).__init__(model)
        
    def run_sim(self, solver, **kdws):
        raise NotImplementedError("Optimizer abstract method. Call child class")       

    def run_opt(self, solver, **kwds):
        raise NotImplementedError("Optimizer abstract method. Call child class")
        
    @staticmethod
    def add_warm_start_suffixes(model, use_k_aug=False):
        """Adds suffixed variables to problem"""
        
        # Ipopt bound multipliers (obtained from solution)
        model.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
        model.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
        # Ipopt bound multipliers (sent to solver)
        model.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
        model.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)
        # Obtain dual solutions from first solve and send to warm start
        model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
        
        if use_k_aug:
            m.dof_v = Suffix(direction=Suffix.EXPORT)
            m.rh_name = Suffix(direction=Suffix.IMPORT)
            
        return None
            
    @staticmethod
    def update_warm_start(model):
        """Updates the suffixed variables for a warmstart"""
        
        model.ipopt_zL_in.update(model.ipopt_zL_out)
        model.ipopt_zU_in.update(model.ipopt_zU_out)
        
        return None