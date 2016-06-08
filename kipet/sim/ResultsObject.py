import datetime
import pandas as pd
import numpy as np

class ResultsObject(object):
    def __init__(self):
        """
        A class to store simulation and optimization results.
        """
        # Data series
        self.generated_datetime = datetime.datetime
        self.results_name = None
        self.solver_statistics = {}
        self.C = None
        self.C_noise = None
        self.S = None
       
