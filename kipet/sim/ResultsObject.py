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
        self.Z = None
        self.X = None
        self.C = None
        self.S = None
        self.sigma_sq = None
        self.device_variance = None
        self.P = None
        self.dZdt = None
        self.dXdt = None
