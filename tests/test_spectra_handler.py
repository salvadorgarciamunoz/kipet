import unittest
from unittest import mock

import pint
from pyomo.core.base.var import _GeneralVarData

import kipet


class TestSpectraHandling(unittest.TestCase):
    
    
    """Tests the kipet_model class"""
    
    def setUp(self):
        
        self.u = pint.UnitRegistry()
    
    def make_kipet_model_instance_and_add_reaction(self):
        
        kipet_model = kipet.KipetModel()
        r1 = kipet_model.new_reaction('reaction_model')
        return kipet_model, r1
    
    def make_simple_reaction_model_with_data(self, kipet_model, name='1', subset=10):
        
        r1 = kipet_model.new_reaction(f'reaction_model_{name}')
        filename = 'example_data/Ex_1_C_data.txt'
        df_data = kipet_model.read_data_file(filename)
        
        r1.add_parameter('k1', value=2.0, bounds=(0.0, 5.0))
        r1.add_parameter('k2', value=0.2, bounds=(0.0, 2.0))
        
        r1.add_component('A', value=0.001, variance=1e-10)
        r1.add_component('B', value=0.0, variance=1e-11)
        r1.add_component('C', value=0.0, variance=1e-8)
       
        r1.add_data(data=df_data.iloc[::subset, :], remove_negatives=True)   
        
        c = r1.get_model_vars()

        r1.add_ode('A', -c.k1 * c.A )
        r1.add_ode('B', c.k1 * c.A - c.k2 * c.B )
        r1.add_ode('C', c.k2 * c.B )
        
        return r1
        
    def test_reaction_model_load_spectral_data(self):
        """
        Test adding spectra data to ReactionModel
        """
        kipet_model, r1 = self.make_kipet_model_instance_and_add_reaction()
        filename = 'example_data/Dij.txt'
        r1.add_data(category='spectral', file=filename)
        num_of_orig_wavelengths = r1.spectra.data.shape[1]
        r1.spectra.decrease_wavelengths(2)
        num_of_new_wavelengths = r1.spectra.data.shape[1]
        self.assertEqual(num_of_new_wavelengths*2, num_of_orig_wavelengths)
        

if __name__ == '__main__':
    unittest.main()
