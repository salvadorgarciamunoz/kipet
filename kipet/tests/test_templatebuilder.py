from kipet.model.TemplateBuilder import TemplateBuilder
import pandas as pd
import pyomo.environ as pe
import pyomo.dae as dae
import unittest


class TestTemplateBuilder(unittest.TestCase):

    def test_add_parameter(self):

        builder = TemplateBuilder()
        builder.add_parameter('k', 0.01)

        self.assertEqual(builder.num_parameters, 1)
        self.assertTrue('k' in builder._parameters)
        self.assertEqual(builder._parameters['k'], 0.01)
        self.assertFalse('k' in builder._parameters_bounds)

        builder.add_parameter('k1', 0.01, bounds=(0, 1))

        self.assertEqual(builder.num_parameters, 2)
        self.assertTrue('k1' in builder._parameters)
        self.assertTrue('k1' in builder._parameters_bounds)
        self.assertEqual(builder._parameters_bounds['k1'][0], 0)
        self.assertEqual(builder._parameters_bounds['k1'][1], 1)

        builder.add_parameter(['B', 'C'])
        self.assertTrue('B' in builder._parameters)
        self.assertTrue('C' in builder._parameters)
        self.assertIsNone(builder._parameters['B'])
        self.assertIsNone(builder._parameters['C'])
        self.assertEqual(builder.num_parameters, 4)

        builder.add_parameter(['D'], bounds=[(0.0, 1.0)])
        self.assertEqual(builder.num_parameters, 5)
        self.assertTrue('D' in builder._parameters_bounds)
        self.assertEqual(builder._parameters_bounds['D'][0], 0)
        self.assertEqual(builder._parameters_bounds['D'][1], 1)

        builder.add_parameter({'E': 7.0})
        self.assertEqual(builder.num_parameters, 6)
        self.assertEqual(builder._parameters['E'], 7)

        builder.add_parameter({'F': 7.0}, bounds={'F': (4, 5)})
        self.assertEqual(builder.num_parameters, 7)
        self.assertEqual(builder._parameters['F'], 7)
        self.assertTrue('F' in builder._parameters_bounds)
        self.assertEqual(builder._parameters_bounds['F'][0], 4)
        self.assertEqual(builder._parameters_bounds['F'][1], 5)

    def test_add_mixture_component(self):

        builder = TemplateBuilder()
        builder.add_mixture_component('A', 1)

        self.assertEqual(builder.num_mixture_components, 1)
        self.assertTrue('A' in builder._init_conditions)
        self.assertTrue('A' in builder._component_names)
        self.assertEqual(builder._init_conditions['A'], 1)

        builder.add_mixture_component({'B': 0})

        self.assertEqual(builder.num_mixture_components, 2)
        self.assertTrue('B' in builder._init_conditions)
        self.assertTrue('B' in builder._component_names)
        self.assertEqual(builder._init_conditions['B'], 0)

    def test_add_spectral_data(self):

        builder = TemplateBuilder()
        d_frame = pd.DataFrame()
        self.assertFalse(builder.has_spectral_data())
        builder.add_spectral_data(d_frame)
        self.assertTrue(builder.has_spectral_data())

    def test_add_absorption_data(self):
        builder = TemplateBuilder()
        s_frame = pd.DataFrame()
        self.assertFalse(builder.has_adsorption_data())
        builder.add_absorption_data(s_frame)
        self.assertTrue(builder.has_adsorption_data())

    def test_add_complementary_state_variable(self):
        builder = TemplateBuilder()
        builder.add_complementary_state_variable('A', 1)

        self.assertEqual(builder.num_complementary_states, 1)
        self.assertTrue('A' in builder._init_conditions)
        self.assertTrue('A' in builder._complementary_states)
        self.assertEqual(builder._init_conditions['A'], 1)

        builder.add_complementary_state_variable({'B': 0})

        self.assertEqual(builder.num_complementary_states, 2)
        self.assertTrue('B' in builder._init_conditions)
        self.assertTrue('B' in builder._complementary_states)
        self.assertEqual(builder._init_conditions['B'], 0)

    def test_add_measurement_times(self):
        builder = TemplateBuilder()
        times = range(5)
        builder.add_measurement_times(times)
        for i in times:
            self.assertIn(i, builder._meas_times)
            self.assertIn(i, builder.measurement_times)

    def test_add_complementary_state(self):
        builder = TemplateBuilder()
        builder.add_complementary_state_variable('T', 278.0)

        self.assertEqual(builder.num_complementary_states, 1)
        self.assertTrue('T' in builder._init_conditions)
        self.assertTrue('T' in builder._complementary_states)
        self.assertEqual(builder._init_conditions['T'], 278.0)

        builder.add_complementary_state_variable({'V': 10.0})

        self.assertEqual(builder.num_complementary_states, 2)
        self.assertTrue('V' in builder._init_conditions)
        self.assertTrue('V' in builder._complementary_states)
        self.assertEqual(builder._init_conditions['V'], 10.0)

    def test_add_algebraic_variable(self):
        builder = TemplateBuilder()

        builder.add_algebraic_variable('r1')
        self.assertEqual(builder.num_algebraics, 1)
        self.assertTrue('r1' in builder._algebraics)

        builder.add_algebraic_variable(['r2', 'r3'])
        self.assertEqual(builder.num_algebraics, 3)
        self.assertTrue('r2' in builder._algebraics)
        self.assertTrue('r3' in builder._algebraics)

    def test_set_odes_rule(self):
        builder = TemplateBuilder()

        # define explicit system of ODEs
        def rule_odes(m, t):
            exprs = dict()
            exprs['A'] = -m.P['k'] * m.Z[t, 'A']
            exprs['B'] = m.P['k'] * m.Z[t, 'A']
            return exprs

        builder.set_odes_rule(rule_odes)

        m = pe.ConcreteModel()
        m.P = pe.Param(['k'], initialize=0.01)
        m.t = dae.ContinuousSet(bounds=(0, 1))
        m.Z = pe.Var(m.t, ['A', 'B'])

        self.assertIsInstance(builder._odes(m, 0), dict)

        def rule(x, y, z):
            return x + y + z

        self.assertRaises(RuntimeError, builder.set_odes_rule, rule)

    def test_set_algebraics_rule(self):
        builder = TemplateBuilder()

        def rule(x, y, z):
            return x + y + z

        self.assertRaises(RuntimeError, builder.set_algebraics_rule, rule)

    def test_construct(self):

        mixture_components = {'A': 1, 'B': 0}
        extra_states = {'T': 278.0}
        parameters = {'k': 0.01}
        algebraics = ['r1','r2']

        builder = TemplateBuilder(concentrations=mixture_components,
                                  parameters=parameters,
                                  extra_states=extra_states,
                                  algebraics=algebraics)

        self.assertEqual(builder.num_mixture_components, 2)
        self.assertTrue('A' in builder._init_conditions)
        self.assertTrue('A' in builder._component_names)
        self.assertEqual(builder._init_conditions['A'], 1)
        self.assertTrue('B' in builder._init_conditions)
        self.assertTrue('B' in builder._component_names)
        self.assertEqual(builder._init_conditions['B'], 0)

        self.assertEqual(builder.num_parameters, 1)
        self.assertTrue('k' in builder._parameters)
        self.assertEqual(builder._parameters['k'], 0.01)
        self.assertFalse('k' in builder._parameters_bounds)

        self.assertEqual(builder.num_complementary_states, 1)
        self.assertTrue('T' in builder._init_conditions)
        self.assertTrue('T' in builder._complementary_states)
        self.assertEqual(builder._init_conditions['T'], 278.0)

        self.assertEqual(builder.num_algebraics, 2)
        self.assertTrue('r1' in builder._algebraics)
        self.assertTrue('r2' in builder._algebraics)

        parameters = ['k']
        builder = TemplateBuilder(parameters=parameters)
        self.assertEqual(builder.num_parameters, 1)
        self.assertTrue('k' in builder._parameters)
