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

from kipet.core_methods.data_tools import *
from kipet.visuals.plots import colors

data_categories = ['concentration', 'spectral', 'state', 'trajectory', 'custom']

class DataBlock():
    
    def __init__(self):
        
        self.datasets = {}
        
    def __getitem__(self, value):
        
        return self.datasets[value]
         
    def __str__(self):
        
        format_string = "{:<20}{:<15}{:<30}{:<15}{:<30}\n"
        data_str = 'DataBlock:\n'
        data_str += format_string.format(*['Name', 'Category', 'Components', 'Size', 'File'])
        
        for dataset in self.datasets.values():
            
            species_str = ', '.join([s for s in dataset.species[:3]])
            if len(dataset.species) > 3:
                species_str += f', ...'
            
            if dataset.file is not None:
                data_str += format_string.format(f'{dataset.name}', f'{dataset.category}', f'{species_str}', f'{dataset.data.shape}', f'{dataset.file.name}')
            else:
                data_str += format_string.format(f'{dataset.name}', f'{dataset.category}', f'{species_str}', f'{dataset.data.shape}', 'None')
                
        
        return data_str

    def __repr__(self):
        return self.__str__()

    def __iter__(self):
        for param, data in self.datasets.items():
            yield data
            
    def __len__(self):
        return len(self.datasets)
    
            
    def add_dataset(self, *args, **kwargs):
        
        """Should handle a series of different input methods:
          
        KP = KineticParameter('k1', init=1.0, bounds=(0.01, 10))
        builder.add_parameter_temp(KP)
        
        - or -
        
        builder.add_parameter('k1', init=1.0, bounds=(0.01, 10))
        
        - or -
        
        builder.add_parameter('k1', 1.0, (0.01, 10))
            
        """        
        category = kwargs.pop('category', None)
        data = kwargs.pop('data', None)
        file = kwargs.pop('file', None)
        
        units = None
        notes = None
        description = None
        
        if isinstance(args[0], DataSet):
            self.datasets[args[0].name] = args[0]
        
        else:
            if isinstance(args[0], str):
                
                dataset = DataSet(
                    args[0], 
                    category=category, 
                    data=data, 
                    units=units, 
                    notes=notes, 
                    file=file,
                    description=description,
                 )
                self.datasets[args[0]] = dataset
    
        return None
    
    def _check_duplicates(self):
        """Easy data mapping to ElementBlocks"""

        blocks = ['components', 'states', 'algebraics']
        
        all_data_cols = []
        for dataset in self.datasets.values():
            if dataset.category in ['spectral', 'trajectory']:
                continue
            all_data_cols.extend(dataset.data.columns)
        
        num_duplicates = len(all_data_cols) - len(set(all_data_cols))
        if num_duplicates != 0:
            raise ValueError('Duplicates ({num_duplicates}) in component, state, and algebraic data detected!')

    @property
    def names(self):
        return [dataset.name for dataset in self.datasets.values()]
        

    @property
    def time_span(self):
        
        start_time = min([dataset.time_span[0] for dataset in self.datasets.values()])
        stop_time = max([dataset.time_span[1] for dataset in self.datasets.values()])

        return (start_time, stop_time)

class DataSet(object):
    
    """The specific data object"""
    
    def __init__(self, 
                 name, 
                 category=None, 
                 data=None, 
                 units=None, 
                 notes=None, 
                 file=None,
                 description=None,
                 ):
        
        self.name = name
        self.category = category
        # if category not in data_categories:
        #     raise ValueError(f'The data type must be in one of the following: {", ".join(data_categories)}')

        self.data = data
        self.file = file
        self.units = units
        self.notes = notes
        self.description = description

        self._check_input()

    def __repr__(self):
        
        if self.data is not None:
            return f'DataSet(name={self.name}, category={self.category}, data={self.data.shape})'
        else:
            return f'DataSet(name={self.name}, category={self.category}, data=None)'
    def __str__(self):
        
        if self.data is not None:
            return f'DataSet(name={self.name}, category={self.category}, data={self.data.shape})'
        else:
            return f'DataSet(name={self.name}, category={self.category}, data=None)'

    def _check_input(self):
        """Checks whether the data has been provided as a pandas dataframe or
        if a filename has been provided. If not, the DataBlock is initialized
        without any data
        """
        if self.data is not None:
            if not isinstance(self.data, pd.DataFrame):
                raise ValueError('Data must be a pandas DataFrame instance')
            
        elif self.data is None and self.file is not None:
            # if self.category == 'spectral':
            #     self.data = read_file
            print(self.file)
            self.data = read_file(self.file)
            
        else:
            print('DataBlock object initialized without data')
    
        return None
    
    def remove_negatives(self):
        """Replaces the negative values with zero"""
        self.data[self.data < 0] = 0
        return None
    
    def show_data(self):
        """Method to show the data using Plotly"""
        
        if self.data is None:
            print('No data to plot')
            return None
        
        if self.category in ['concentration', 'state', 'trajectory', 'custom']:
            self._plot_2D_data()
            
        if self.category == 'spectral':
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
        for i, cols in enumerate(self.data.columns):
            fig.add_trace(go.Scatter(x=self.data.index,
                                     y=self.data[cols],
                                     mode='markers',
                                     name=cols,
                                     marker=dict(size=10, opacity=0.5, color=colors[i])),
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
        if self.data is not None and self.category in ['concentration', 'state', 'trajectory', 'custom']:
            return list(self.data.columns)
        elif self.data is None:
            return []
        else:
            return []
        
    @property
    def time_span(self):
        return self.data.index.min(), self.data.index.max()


if __name__ == '__main__':

    pass