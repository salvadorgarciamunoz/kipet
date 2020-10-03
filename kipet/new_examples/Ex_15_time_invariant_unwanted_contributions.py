"""Example 15: Time invariant unwanted contributions with the new KipetModel
 
"""
# Standard library imports
import sys # Only needed for running the example from the command line

# Third party imports
import pandas as pd

# Kipet library imports
from kipet.kipet import KipetModel
from kipet.library.common.read_write_tools import read_file

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
    #kipet_model.settings.general.initialize_pe = False
    #kipet_model.settings.general.no_user_scaling = True
    
    kipet_model.settings.collocation.nfe = 100
    
    # For the "time-invariant" unwanted contribuiton cases, stoichiometric coefficients
    # of each reaction (St) and/or dosing concentration for each dosing time (Z_in) are required
    # to call the different objective function.
    St = {}
    St['r1'] = [-1, 1, 0]
    St['r2'] = [0, -1, 0]
    
    # In this case, there is no dosing time. 
    # Therefore, the following expression just an input example 
    # if the user has dosing concentraion in the model.
    # Z_in = dict()
    # Z_in["t=5"] = [0,0,5]

    # Assuming we know the "time_invariant" unwanted contribution is included,
    # so set the option time_invariant_G=True. St and Z_in are transmitted as well.
    # finally we run the optimization
    kipet_model.settings.parameter_estimator.G_contribution = 'time_invariant_G'
    kipet_model.settings.parameter_estimator.St = St    
    # kipet_model.settings.parameter_estimator.Z_in = Z_in

    # Run KIPET
    kipet_model.run_opt()
    kipet_model.results.show_parameters

    if with_plots:
        kipet_model.results.plot()
        
    """We can now compare the results with the known profiles"""
    
    # Read the true S to compare with results
    S_true_filename = kipet_model.set_directory("S_True_for_unwanted_G.csv")
    S_True = read_file(S_true_filename)

    # In this example, we know the magnitude of unwanted contribution.
    # Therefore, we can calculate the matched S according to "" to compare the results.
    index = list(S_True.index)
    column = list(S_True.columns)
    data = []
    for i in index:
        sgi = 2.5E-6*i/0.01
        row = [sgi,sgi,sgi]
        data.append(row)
        
    Sg = pd.DataFrame(data, columns = column, index = index)
    S_matched = S_True + Sg
    # Make sure the columns have the same names as in the original
    S_matched.columns = ['A', 'B', 'C']
        
    # Use the "label" kwarg to add some info to the legend in the plot
    if with_plots:
        kipet_model.results.plot('S', extra_data={'data': S_matched, 'label': 'matched'})
        
