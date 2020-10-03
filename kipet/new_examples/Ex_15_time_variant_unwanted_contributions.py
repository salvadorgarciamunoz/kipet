"""Example 15: Time variant unwanted contributions with the new KipetModel
 
"""
# Standard library imports
import sys # Only needed for running the example from the command line

# Third party imports

# Kipet library imports
from kipet.kipet import KipetModel

if __name__ == "__main__":

    with_plots = True
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False
 
    # Define the general model
    kipet_model = KipetModel()
    
    # Add the data
    filename = kipet_model.set_directory('Dij_tv_G.txt')
    kipet_model.add_dataset('D_frame', category='spectral', file=filename)
    
    # Add the model parameters
    kipet_model.add_parameter('k1', init=1.4, bounds=(0.0, 2.0))
    kipet_model.add_parameter('k2', init=0.25, bounds=(0.0, 0.5))
    
    # Declare the components and give the initial values
    kipet_model.add_component('A', state='concentration', init=1e-2)
    kipet_model.add_component('B', state='concentration', init=0.0)
    kipet_model.add_component('C', state='concentration', init=0.0)
    
    # define explicit system of ODEs
    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.Z[t,'A']
        exprs['B'] = m.P['k1']*m.Z[t,'A']-m.P['k2']*m.Z[t,'B']
        exprs['C'] = m.P['k2']*m.Z[t,'B']
        return exprs
    
    kipet_model.add_equations(rule_odes)
    
    # Settings
    kipet_model.settings.general.initialize_pe = False
    kipet_model.settings.general.no_user_scaling = True
    
    kipet_model.settings.collocation.nfe = 100
    kipet_model.settings.parameter_estimator.G_contribution = 'time_variant_G'

    # Unwanted contribution handling
    kipet_model.builder.add_qr_bounds_init(bounds=(0,None),init=1.1)
    kipet_model.builder.add_g_bounds_init(bounds=(0,None))

    # Run KIPET
    kipet_model.run_opt()

    if with_plots:
        kipet_model.results.plot()
