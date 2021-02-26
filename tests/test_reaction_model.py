import unittest
from unittest import mock

import pint
from pyomo.core.base.var import _GeneralVarData

import kipet


class TestKipetModel(unittest.TestCase):
    
    
    """Tests the kipet_model class"""
    
    def setUp(self):
        
        self.u = pint.UnitRegistry()
    
    def make_kipet_model_instance_and_add_reaction(self):
        
        kipet_model = kipet.KipetModel()
        r1 = kipet_model.new_reaction('reaction_model')
        return kipet_model, r1
    
    def make_simple_reaction_model_with_only_components(self, kipet_model, name='1', subset=10):
        
        r1 = kipet_model.new_reaction(f'reaction_model_{name}')
        
        r1.add_parameter('k1', value=2.0, bounds=(0.0, 5.0))
        r1.add_parameter('k2', value=0.2, bounds=(0.0, 2.0))
        
        r1.add_component('A', value=0.001, variance=1e-10)
        r1.add_component('B', value=0.0, variance=1e-11)
        r1.add_component('C', value=0.0, variance=1e-8)
       
        return r1
        
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
        
    def test_reaction_model_init(self):
        """
        Test that it can create a KipetModel instance
        """
        kipet_model, r1 = self.make_kipet_model_instance_and_add_reaction()
        self.assertIsInstance(r1, kipet.library.top_level.reaction_model.ReactionModel)
    
    def test_add_parameter(self):
        """
        Test adding a parameter to ReactionModel
        """
        kipet_model, r1 = self.make_kipet_model_instance_and_add_reaction()
        r1.add_parameter('k1', value=2.0, bounds=(0.0, 5.0), fixed=True, units='m')
        self.assertIn('k1', r1.parameters.names)
        self.assertEqual(2.0, r1.parameters['k1'].value)
        self.assertEqual(0.0, r1.parameters['k1'].lb)
        self.assertEqual(5.0, r1.parameters['k1'].ub)
        self.assertEqual(True, r1.parameters['k1'].fixed)
        self.assertEqual(str(self.u('m').units), str(r1.parameters['k1'].units.units))
    
    def test_add_component(self):
        """
        Test adding a parameter to ReactionModel
        """
        kipet_model, r1 = self.make_kipet_model_instance_and_add_reaction()
        r1.add_component('A', value=2.0, bounds=(0.0, 5.0), known=False, variance=1.0, units='m')
        self.assertIn('A', r1.components.names)
        self.assertEqual(2.0, r1.components['A'].value)
        self.assertEqual(0.0, r1.components['A'].lb)
        self.assertEqual(5.0, r1.components['A'].ub)
        self.assertEqual(False, r1.components['A'].known)
        self.assertEqual(str(self.u('m').units), str(r1.components['A'].units.units))
    
    def test_add_state(self):
        """
        Test adding a parameter to ReactionModel
        """
        kipet_model, r1 = self.make_kipet_model_instance_and_add_reaction()
        r1.add_state('A', value=2.0, bounds=(0.0, 5.0), known=False, variance=1.0, units='m')
        self.assertIn('A', r1.states.names)
        self.assertEqual(2.0, r1.states['A'].value)
        self.assertEqual(0.0, r1.states['A'].lb)
        self.assertEqual(5.0, r1.states['A'].ub)
        self.assertEqual(False, r1.states['A'].known)
        self.assertEqual(str(self.u('m').units), str(r1.states['A'].units.units))
        
    def test_add_constant(self):
        """
        Test adding a parameter to ReactionModel
        """
        kipet_model, r1 = self.make_kipet_model_instance_and_add_reaction()
        r1.add_constant('A', value=2.0, units='m')
        self.assertIn('A', r1.constants.names)
        self.assertEqual(2.0, r1.constants['A'].value)
        self.assertEqual(str(self.u('m').units), str(r1.constants['A'].units.units))
        
    def test_reaction_model_cmod_generator(self):
        """
        Test generation of vars for model creation in ReactionModel
        """
        kipet_model, r1 = self.make_kipet_model_instance_and_add_reaction()
        r1.add_component('A', value=2.0, bounds=(0.0, 5.0), known=False, variance=1.0, units='m')
        c = r1.get_model_vars()
        self.assertIn('A', c.var_dict)
        self.assertIsInstance(c.A, _GeneralVarData)
        
    def test_reaction_model_load_state_data(self):
        """
        Test adding data ReactionModel
        """
        kipet_model, r1 = self.make_kipet_model_instance_and_add_reaction()
        r1.add_component('A', value=2.0, bounds=(0.0, 5.0))
        filename = 'example_data/Ex_1_C_data.txt'
        r1.add_data(file=filename)   
        self.assertIn('ds1', r1.data)
        self.assertIn('A', r1.data['ds1'].columns)
        
    def test_reaction_model_load_spectral_data(self):
        """
        Test adding spectra data to ReactionModel
        """
        kipet_model, r1 = self.make_kipet_model_instance_and_add_reaction()
        r1.add_component('A', value=2.0, bounds=(0.0, 5.0))
        filename = 'example_data/Dij.txt'
        r1.add_data(category='spectral', file=filename)
        df_data = kipet_model.read_data_file(filename)
        self.assertTrue(all(df_data == r1.spectra.data))
        
    def test_adding_ode_to_reaction_model(self):
        """
        Test adding ode equation to ReactionModel
        """
        kipet_model, r1 = self.make_kipet_model_instance_and_add_reaction()
        r1.add_component('A', value=2.0, bounds=(0.0, 5.0))
        c = r1.get_model_vars()
        r1.add_ode('A', c.A )
        self.assertIn('A', r1.odes_dict)
        self.assertEqual('Z[0,A]', r1.odes_dict['A'].expression.name)
  
    def test_adding_algebraic_variable(self):
        """
        Test adding algebraic variable to ReactionModel
        """
        kipet_model, r1 = self.make_kipet_model_instance_and_add_reaction()
        r1.add_alg_var('y', description='alg var')
        self.assertIn('y', r1.algebraics.names)
                
    def test_adding_algebraic_variable_description(self):
        """
        Test adding algebraic variable with description to ReactionModel
        """
        kipet_model, r1 = self.make_kipet_model_instance_and_add_reaction()
        r1.add_alg_var('y', description='alg var')
        self.assertIn('alg var', r1.algebraics['y'].description)

    def test_adding_algebraic_equation(self):
        """
        Test adding algebraic equation to ReactionModel
        """
        kipet_model = kipet.KipetModel()
        r1 = self.make_simple_reaction_model_with_only_components(kipet_model)
        r1.add_alg_var('y', description='alg var')
        r1.add_component('B')
        r1.add_component('C')
        c = r1.get_model_vars()
        r1.add_algebraic('y', (c.B)/(c.B + c.C) )
        self.assertIn('y', c.var_dict)    
        self.assertIn('A', c.var_dict)    
        self.assertIn('B', c.var_dict)
        self.assertIn('y', r1.algs_dict)

    def test_adding_algebraic_equation_with_division_tol(self):
        """
        Test adding algebraic equation with division tolerance added to the 
        numerator and denominator to ReactionModel
        """
        kipet_model = kipet.KipetModel()
        r1 = self.make_simple_reaction_model_with_only_components(kipet_model)
        r1.add_alg_var('y', description='alg var')
        r1.add_component('B')
        r1.add_component('C')
        c = r1.get_model_vars()
        r1.add_algebraic('y', (c.B)/(c.B + c.C) )
        self.assertEqual('Z[0,B] + 1e-12', r1.algs_dict['y'].expression.args[0].to_string())

class TestTemplateBuilder(unittest.TestCase):   

    """Tests the TemplateBuilder module"""
    
    def setUp(self):
        
        self.u = pint.UnitRegistry()
        self.kipet_model = kipet.KipetModel()
         
        r1 = self.kipet_model.new_reaction('fed_batch_parest')
    
        r1.add_parameter('k1', value = 0.05, units='ft**3/mol/min')
    
        r1.add_component('A', value=2.0, units='mol/L')
        r1.add_component('B', value=0.0, units='mol/L')
        r1.add_component('C', value=0.0, units='mol/L')
        
        r1.add_state('V', value = 0.264172, units='gal')
        
        # Volumetric flow rate for B feed
        r1.add_alg_var('R1', units='mol/L/min', description='Reaction 1')
        
        filename = 'example_data/abc_fedbatch.csv'
        r1.add_data('C_data', file=filename, units='mol/L', remove_negatives=True)
        
        # Step function for B feed - steps can be added
        r1.add_step('s_Qin_B', coeff=1, time=15, switch='off')
        
        r1.add_constant('Qin_B', value=6, units='L/hour')
        # Concentration of B in feed
        r1.add_constant('Cin_B', value=2.0, units='mol/L')
        
        #r1.check_component_units()
        c = r1.get_model_vars()
    
        # c now holds of all of the pyomo variables needed to define the equations
        # Using this object allows for a much simpler construction of expressions
        r1.add_algebraic('R1', c.k1*c.A*c.B )
        
        Qin_B = c.Qin_B*(c.s_Qin_B)
        QV = Qin_B/c.V
        
        r1.add_ode('A', -c.A*QV - c.R1 )
        r1.add_ode('B', (c.Cin_B - c.B)*QV - c.R1 )
        r1.add_ode('C', -c.C*QV + c.R1)
        r1.add_ode('V', Qin_B)
        
        self.r1 = r1
        
    def test_template_builder_add_component_data(self):
        """Test that components in reaction model equal those in 
        TemplateBuilder
        """
        self.r1.builder.add_model_element(self.r1.components)       
        self.assertEqual(self.r1.builder.template_component_data, self.r1.components)
        
    def test_template_builder_add_odes(self):
        
        self.r1.builder.set_odes_rule(self.r1.odes)       
        self.assertEqual(self.r1.odes, self.r1.builder._odes)
        
        


if __name__ == '__main__':
    unittest.main()
