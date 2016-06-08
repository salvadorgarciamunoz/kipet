import datetime
import pandas as pd
import numpy as np

class ResultsObject(object):
    def __init__(self):
        """
        A class to store simulation and optimization results.
        """

        # Data series
        self.time = None
        self.generated_datetime = datetime.datetime
        self.results_name = None
        self.solver_statistics = {}
        self.panel = None
       
    def extract_results_from_pyomo_model(self,pyomo_model):
        self.time = sorted(pyomo_model.time)
        mixture_component_names = [name for name in pyomo_model.mixture_component_names]
        n_times = len(self.time)
        n_components = len(mixture_component_names)
        
        tmp_containers = {}
        tmp_containers['component_concentration'] = []
        
        for t in self.time:
            for name in mixture_component_names:
                tmp_containers['component_concentration'].append(pyomo_model.C[t,name].value)
        tmp_dict = dict()
        tmp_dict['concentration'] = tmp_containers['component_concentration']
    
        for key,val in tmp_dict.iteritems():
            tmp_dict[key] = np.array(val).reshape((n_times,n_components))
        
        
        self.panel =  pd.Panel(tmp_dict, major_axis=self.time, minor_axis=mixture_component_names)
        
        #return results
        #print results['concentration']
        #plt.plot(results['concentration'])
