"""
Data Handling for Kipet

Created on Mon Aug 24 04:01:57 2020

@author: kevin
"""
import inspect
import os

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from kipet.library.data_tools import *


data_categories = ['concentration', 'spectra', 'state']

class DataBlock(object):
    
    """The specific data object"""
    
    def __init__(self, 
                 name, 
                 category, 
                 data=None, 
                 units=None, 
                 notes=None, 
                 file=None,
                 description=None,
                 ):
        
        self.name = name
        self.category = category
        if category not in data_categories:
            raise ValueError(f'The data type must be in one of the following: {", ".join(data_categories)}')

        self.data = data
        self.file = file
        self.units = units
        self.notes = notes
        self.description = description

        self._check_input()

    def __repr__(self):
        
        if self.data is not None:
            return f'DataBlock(name={self.name}, category={self.category}, data={self.data.shape})'
        else:
            return f'DataBlock(name={self.name}, category={self.category}, data=None)'
    def __str__(self):
        
        if self.data is not None:
            return f'DataBlock(name={self.name}, category={self.category}, data={self.data.shape})'
        else:
            return f'DataBlock(name={self.name}, category={self.category}, data=None)'

    def _check_input(self):
        """Checks whether the data has been provided as a pandas dataframe or
        if a filename has been provided. If not, the DataBlock is initialized
        without any data
        """
        if self.data is not None:
            if not isinstance(self.data, pd.DataFrame):
                raise ValueError('Data must be a pandas DataFrame instance')
            
        elif self.data is None and self.file is not None:
            self.data = read_file(self.file)
            
        else:
            print('DataBlock object initialized without data')
    
        return None
    
    
    def show_data(self):
        """Method to show the data using Plotly"""
        
        if self.data is None:
            print('No data to plot')
            return None
        
        if self.category in ['concentration', 'state']:
            self._plot_2D_data()
            
        if self.category == 'spectra':
            self._plot_spectral_data()
    
        return None
    
    def _plot_2D_data(self):
        """Simple plots for showing concentration or complementary state data
        
        """
        y_axis_text = 'Measured Value'
        if self.description is not None:
            y_axis_text = self.description
            
        if self.units is not None:
            y_axis_text = ' '.join([y_axis_text, '[' + self.units[0] +']'])
            
        x_axis_text = 'Time'
        
        if self.units is not None and len(self.units) == 2:
            x_axis_text = ' '.join([x_axis_text, '[' + self.units[1] +']'])
        
        fig = go.Figure()
        for cols in self.data.columns:
            fig.add_trace(go.Scatter(x=self.data.index,
                                     y=self.data[cols],
                                     mode='markers',
                                     name=cols)
                          )
           
        fig.update_layout(title_text=f'{self.category.capitalize()} data: {self.name}',
                          xaxis_title=f'{x_axis_text}',
                          yaxis_title=f'{y_axis_text}',
                          title_font_size=30)

        plot(fig)

        return None

    def _plot_spectral_data(self, dimension='3D'):
        """ Plots spectral data
        
            Args:
                dataFrame (DataFrame): spectral data
              
            Returns:
                None
    
        """
        if dimension=='3D':
            
            fig = go.Figure()
            fig.add_trace(go.Surface(x=self.data.columns,
                               y=self.data.index,
                               z=self.data.values,
                              ))
            
            fig.update_layout(scene = dict(
                                xaxis_title='Wavelength',
                                yaxis_title='Time',
                                zaxis_title='Spectra'),
                                margin=dict(r=100, b=50, l=100, t=50),
                                title_text=f'{self.name}: {self.category.capitalize()} data',
                                title_font_size=30)
            
            plot(fig)
            
            return None

    # Properties
    @property
    def shape(self):
        if self.data is not None:
            return self.data.shape
        else:
            return None

    @property
    def size(self):
        if self.data is not None:
            return len(self.data.columns)
        else:
            return None
    
    @property
    def species(self):
        if self.data is not None and self.data.category == 'concentration':
            return list(self.data.columns)
        elif self.data is None:
            return []
        else:
            return []

if __name__ == '__main__':

    dataDirectory = os.path.abspath(
        os.path.join( os.path.dirname( os.path.abspath( inspect.getfile(
            inspect.currentframe() ) ) ), '..', 'examples/data_sets'))
    
    # Concentration Test
    filename =  os.path.join(dataDirectory,'Ex_1_C_data.txt')
    C_frame = read_concentration_data_from_txt(filename)
    
    dc = DataBlock('C1', 'concentration', C_frame)
    dc1 = DataBlock('C2', 'concentration')
    dc2 = DataBlock('C3', 'concentration', file=filename, description='Concentration', units=('mol/L', 'min'))
    
    
    dc1.show_data()
    
    # Spectra Test
    filename =  os.path.join(dataDirectory,'Dij.txt')
    D_frame = read_file(filename)
    
    ds = DataBlock('S1', 'spectra', D_frame)
    ds1 = DataBlock('S2', 'spectra', file=filename)
    
    #ds1.show_data()