import unittest
from unittest import mock

import kipet


class TestKipetModel(unittest.TestCase):
    
    
    """Tests the kipet_model class"""
    
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
        
    def test_kipet_model_int(self):
        """
        Test that it can create a KipetModel instance
        """
        kipet_model = kipet.KipetModel()
        self.assertIsInstance(kipet_model, kipet.KipetModel)
        
    def test_kipet_model_can_make_reaction_model(self):
        """
        Test that KipetModel instance can make a ReactionModel instance
        """
        kipet_model, r1 = self.make_kipet_model_instance_and_add_reaction()
        self.assertIsInstance(r1, kipet.library.top_level.reaction_model.ReactionModel)
        self.assertEqual('reaction_model', r1.name)
        
    def test_kipet_model_can_add_multiple_models(self):
        """
        Test that KipetModel instance can add a list of ReactionModel instances
        """
        kipet_model = kipet.KipetModel()
        r1 = kipet_model.new_reaction('reaction_model_1')
        r2 = kipet_model.new_reaction('reaction_model_2')
        model_list = [r1, r2]
        kipet_model.add_model_list(model_list)
        self.assertEqual(2, len(kipet_model.models))
     
    def test_kipet_model_can_only_add_reaction_model(self):
        """
        Test that only ReactionModels can be added to KipetModel.models
        """
        kipet_model = kipet.KipetModel()
        r1 = []
        self.assertRaises(ValueError, kipet_model.add_reaction, r1)
     
    def test_kipet_model_can_copy_reaction_models(self):
        """
        Test that KipetModel instance can copy a ReactionModel instances
        """
        kipet_model, r1 = self.make_kipet_model_instance_and_add_reaction()
        r1.add_component('A', value=1)
        r2 = kipet_model.new_reaction('reaction_model_2', model=r1)
        self.assertEqual(r1.components, r2.components)
    
    def test_kipet_model_can_copy_reaction_models_with_ignore(self):
        """
        Test that KipetModel instance can copy a ReactionModel instances
        """
        kipet_model = kipet.KipetModel()
        r1 = kipet_model.new_reaction('reaction_model_1')
        r1.add_component('A', value=1)
        r2 = kipet_model.new_reaction('reaction_model_2', model=r1, ignore=['components'])
        self.assertNotEqual(r1.components, r2.components)
    
    def test_load_data(self):
        """Test that KipetModel read_data_file method is working"""
        
        from pandas import DataFrame
        kipet_model = kipet.KipetModel()
        filename = 'example_data/Ex_1_C_data.txt'
        df_data = kipet_model.read_data_file(filename)
        self.assertIsInstance(df_data, DataFrame)
        
        
    # def test_mee(self):
        
    #     from unittest.mock import MagicMock
        
    #     kipet_model = kipet.KipetModel()
    #     r1 = self.make_simple_reaction_model_with_data(kipet_model, name='1', subset=10)
    #     r2 = self.make_simple_reaction_model_with_data(kipet_model, name='2', subset=11)
        
    #      # = kipet.KipetModel()
        
    #     kipet_model._create_multiple_experiments_estimator = MagicMock(return_value=True)
    #     kipet_model.run_opt()
        
        
        
        #kipet_model._create_multiple_experiments_estimator(3, 4, 5, key='value')
            






if __name__ == '__main__':
    unittest.main()


#%%

from unittest.mock import MagicMock

thing = kipet.KipetModel()

thing._create_multiple_experiments_estimator = MagicMock(return_value=True)

thing._create_multiple_experiments_estimator(3, 4, 5, key='value')




