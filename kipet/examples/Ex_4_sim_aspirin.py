"""Example 4: Simulated Asprin reaction with new KipetModel

"""
# Standard library imports
import sys # Only needed for running the example from the command line

# Third party imports
import pandas as pd
from pyomo.environ import exp

# Kipet library imports
from kipet import KipetModel

if __name__ == "__main__":
    
    with_plots = True
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False
    
    # This holds the model
    kipet_model = KipetModel()
    
    r1 = kipet_model.new_reaction('reaction-1')

    # Components
    r1.add_component('SA', value=1.0714, description='Salicitilc acid')
    r1.add_component('AA', value=9.3828, description='Acetic anhydride')
    r1.add_component('ASA', value=0.0177, description='Acetylsalicylic acid')
    r1.add_component('HA', value=0.0177, description='Acetic acid')
    r1.add_component('ASAA', value=0.000015, description='Acetylsalicylic anhydride')
    r1.add_component('H2O', value=0.0, description='Water')
    

    # Parameters
    params = dict()
    params['k0'] = 0.0360309
    params['k1'] = 0.1596062
    params['k2'] = 6.8032345
    params['k3'] = 1.8028763
    params['kd'] = 7.1108682
    params['kc'] = 0.7566864
    params['Csa'] = 2.06269996

    for param, init_value in params.items():
        r1.add_parameter(param, value=init_value)

    # Additional state variables
    extra_states = dict()
    extra_states['V'] = 0.0202
    extra_states['Masa'] = 0.0
    extra_states['Msa'] = 9.537
    
    for comp, init_value in extra_states.items():
        r1.add_state(comp, value=init_value)

    # Algebraics
    reactions = ['r0','r1','r2','r3','r4','r5']
    
    # Fix a trajectory using the data key word - it should be the name of the
    # dataset where the data is
    r1.add_alg_var('f',  value=0, description='flow f', data='traj')
    r1.add_alg_var('r0', value=0, description='Reaction 0')
    r1.add_alg_var('r1', value=0, description='Reaction 1')
    r1.add_alg_var('r2', value=0, description='Reaction 2')
    r1.add_alg_var('r3', value=0, description='Reaction 3')
    r1.add_alg_var('r4', value=0, description='Reaction 4')
    r1.add_alg_var('r5', value=0, description='Reaction 5')
    r1.add_alg_var('v_sum', value=0, description='Volumne Sum')
    r1.add_alg_var('Csat', value=0, description='C saturation', data='traj')

    gammas = dict()
    gammas['SA']=    [-1, 0, 0, 0, 1, 0]
    gammas['AA']=    [-1,-1, 0,-1, 0, 0]
    gammas['ASA']=   [ 1,-1, 1, 0, 0,-1]
    gammas['HA']=    [ 1, 1, 1, 2, 0, 0]
    gammas['ASAA']=  [ 0, 1,-1, 0, 0, 0]
    gammas['H2O']=   [ 0, 0,-1,-1, 0, 0]

    epsilon = dict()
    epsilon['SA']= 0.0
    epsilon['AA']= 0.0
    epsilon['ASA']= 0.0
    epsilon['HA']= 0.0
    epsilon['ASAA']= 0.0
    epsilon['H2O']= 1.0
    
    partial_vol = dict()
    partial_vol['SA']=0.0952552311614
    partial_vol['AA']=0.101672206869
    partial_vol['ASA']=0.132335206093
    partial_vol['HA']=0.060320218688
    partial_vol['ASAA']=0.186550717015
    partial_vol['H2O']=0.0883603912169

    filename = 'example_data/extra_states.txt'
    r1.add_data('traj', category='trajectory', file=filename)
    
    filename = 'example_data/concentrations.txt'
    r1.add_data('conc', category='trajectory', file=filename)
    
    filename = 'example_data/init_Z.csv'
    r1.add_data('init_Z', category='trajectory', file=filename)
    
    filename = 'example_data/init_X.csv'
    r1.add_data('init_X', category='trajectory', file=filename)
    
    filename = 'example_data/init_Y.csv'
    r1.add_data('init_Y', category='trajectory', file=filename)

    c = r1.get_model_vars()

    # Algebraics
    r1.add_algebraic('r0', c.k0*c.SA*c.AA )
    r1.add_algebraic('r1', c.k1*c.ASA*c.AA )
    r1.add_algebraic('r2', c.k2*c.ASAA*c.H2O )
    r1.add_algebraic('r3', c.k3*c.AA*c.H2O )
    
    step = 1/(1 + exp(-c.Msa/1e-4))
    r1.add_algebraic('r4', c.kd*(c.Csa - c.SA + 1e-6)**1.90*step )
    
    diff = c.ASA - c.Csat
    r1.add_algebraic('r5', 0.3950206559*c.kc*(diff+((diff)**2+1e-6)**0.5)**1.34 )
    
    Cin = 39.1
    v_sum = 0.0
   
    for com in r1.components.names:
        v_sum += partial_vol[com]*(sum(gammas[com][j]*r1.ae(f'r{j}') for j in range(6)) + epsilon[com]*c.f/c.V*Cin)
    
    r1.add_algebraic('v_sum', v_sum)
    
    # ODES
    Cin = 41.4
    r1.add_ode('V', c.V*c.v_sum )
    
    for com in r1.components.names:
        r1.add_ode(com, sum(gammas[com][j]*r1.ae(f'r{j}') for j in range(6)) + epsilon[com]*c.f/c.V*Cin - c.v_sum*getattr(c, com))
    
    r1.add_ode('Masa', 180.157*c.V*c.r5 )
    r1.add_ode('Msa', -138.121*c.V*c.r4 )

    #Create the model
    r1.set_times(0, 210.5257)

    # Settings
    r1.settings.collocation.nfe = 100
    r1.settings.simulator.solver_opts.update({'halt_on_ampl_error' :'yes'})
    
    r1.initialize_from_trajectory('Z', 'init_Z')
    r1.initialize_from_trajectory('X', 'init_X')
    r1.initialize_from_trajectory('Y', 'init_Y')

    # Run the simulation
    r1.simulate()

    # # Plot the results
    if with_plots:
        r1.plot()