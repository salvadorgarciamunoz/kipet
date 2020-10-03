"""Example 4: Simulated Asprin reaction with new KipetModel

"""
# Standard library imports
import sys # Only needed for running the example from the command line

# Third party imports
import pandas as pd
from pyomo.environ import exp

# Kipet library imports
from kipet.kipet import KipetModel

if __name__ == "__main__":
    
    with_plots = True
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False
    
    # This holds the model
    kipet_model = KipetModel()
    
    # Data set-up: Use trajectory as the category for initialization data
    # as this is not added to the pyomo model
    
    filename = kipet_model.set_directory('extra_states.txt')
    kipet_model.add_dataset('traj', category='trajectory', file=filename)
    
    filename = kipet_model.set_directory('concentrations.txt')
    kipet_model.add_dataset('conc', category='trajectory', file=filename)
    
    filename = kipet_model.set_directory('init_Z.csv')
    kipet_model.add_dataset('init_Z', category='trajectory', file=filename)
    
    filename = kipet_model.set_directory('init_X.csv')
    kipet_model.add_dataset('init_X', category='trajectory', file=filename)
    
    filename = kipet_model.set_directory('init_Y.csv')
    kipet_model.add_dataset('init_Y', category='trajectory', file=filename)

    # Components
    components = dict()
    components['SA'] = 1.0714                  # Salicitilc acid
    components['AA'] = 9.3828               # Acetic anhydride
    components['ASA'] = 0.0177                 # Acetylsalicylic acid
    components['HA'] = 0.0177                  # Acetic acid
    components['ASAA'] = 0.000015                # Acetylsalicylic anhydride
    components['H2O'] = 0.0                 # water

    for comp, init_value in components.items():
        kipet_model.add_component(comp, state='concentration', init=init_value)

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
        kipet_model.add_parameter(param, init=init_value)

    # Additional state variables
    extra_states = dict()
    extra_states['V'] = 0.0202
    extra_states['Masa'] = 0.0
    extra_states['Msa'] = 9.537
    
    for comp, init_value in extra_states.items():
        kipet_model.add_component(comp, state='state', init=init_value)

    # Algebraics
    algebraics = ['f','r0','r1','r2','r3','r4','r5','v_sum','Csat']

    kipet_model.add_algebraic_variables(algebraics)

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
    
    def rule_algebraics(m,t):
        r = list()
        r.append(m.Y[t,'r0']-m.P['k0']*m.Z[t,'SA']*m.Z[t,'AA'])
        r.append(m.Y[t,'r1']-m.P['k1']*m.Z[t,'ASA']*m.Z[t,'AA'])
        r.append(m.Y[t,'r2']-m.P['k2']*m.Z[t,'ASAA']*m.Z[t,'H2O'])
        r.append(m.Y[t,'r3']-m.P['k3']*m.Z[t,'AA']*m.Z[t,'H2O'])

        # dissolution rate
        step = 1.0/(1.0+exp(-m.X[t,'Msa']/1e-4))
        rd = m.P['kd']*(m.P['Csa']-m.Z[t,'SA']+1e-6)**1.90*step
        r.append(m.Y[t,'r4']-rd)
        
        # crystalization rate
        diff = m.Z[t,'ASA'] - m.Y[t,'Csat']
        rc = 0.3950206559*m.P['kc']*(diff+((diff)**2+1e-6)**0.5)**1.34
        r.append(m.Y[t,'r5']-rc)

        Cin = 39.1
        v_sum = 0.0
        V = m.X[t,'V']
        f = m.Y[t,'f']
        for c in m.mixture_components:
            v_sum += partial_vol[c]*(sum(gammas[c][j]*m.Y[t,'r{}'.format(j)] for j in range(6))+ epsilon[c]*f/V*Cin)
        r.append(m.Y[t,'v_sum']-v_sum)

        return r

    kipet_model.add_algebraics(rule_algebraics)
    
    def rule_odes(m,t):
        exprs = dict()

        V = m.X[t,'V']
        f = m.Y[t,'f']
        Cin = 41.4
        # volume balance
        vol_sum = 0.0
        for c in m.mixture_components:
            vol_sum += partial_vol[c]*(sum(gammas[c][j]*m.Y[t,'r{}'.format(j)] for j in range(6))+ epsilon[c]*f/V*Cin)
        exprs['V'] = V*m.Y[t,'v_sum']

        # mass balances
        for c in m.mixture_components:
            exprs[c] = sum(gammas[c][j]*m.Y[t,'r{}'.format(j)] for j in range(6))+ epsilon[c]*f/V*Cin - m.Y[t,'v_sum']*m.Z[t,c]

        exprs['Masa'] = 180.157*V*m.Y[t,'r5']
        exprs['Msa'] = -138.121*V*m.Y[t,'r4']
        return exprs

    kipet_model.add_equations(rule_odes)
    
    # Create the model
    kipet_model.set_times(0, 210.5257)

    # Settings
    kipet_model.settings.collocation.nfe = 100
    
    kipet_model.settings.simulator.solver_opts.update({'halt_on_ampl_error' :'yes'})
    
    # If you need to fix a trajectory or initialize, do so here:
    kipet_model.fix_from_trajectory('Y', 'Csat', 'traj') 
    kipet_model.fix_from_trajectory('Y', 'f', 'traj')
    
    kipet_model.initialize_from_trajectory('Z', 'init_Z')
    kipet_model.initialize_from_trajectory('X', 'init_X')
    kipet_model.initialize_from_trajectory('Y', 'init_Y')

    # Run the simulation
    kipet_model.simulate()

    # Plot the results
    if with_plots:   
        kipet_model.results.plot('Z')
        kipet_model.datasets['conc'].show_data()
        kipet_model.results.plot('Y', 'Csat', extra_data={'data': kipet_model.datasets['traj'].data['Csat'], 'label': 'traj'})
        kipet_model.results.plot('X', 'V', extra_data={'data': kipet_model.datasets['traj'].data['V'], 'label': 'traj'})
        kipet_model.results.plot('X', 'Msa', extra_data={'data': kipet_model.datasets['traj'].data['Msa'], 'label': 'traj'})
        kipet_model.results.plot('Y', 'f')
        kipet_model.results.plot('X', 'Masa', extra_data={'data': kipet_model.datasets['traj'].data['Masa'], 'label': 'traj'})