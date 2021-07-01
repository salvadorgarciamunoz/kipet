"""
Classes used to manage the data in KIPET
"""
# Third party imports
import pandas as pd
import plotly.graph_objs as go

# KIPET library imports
from kipet.input_output.kipet_io import read_file
from kipet.visuals.plots import colors


class DataBlock:
    """Class to manage data in KIPET.
    
    This class is not intended to be used directly by the user.
    
    :Methods:
        
        - :func:`add_dataset`
        - :func:`names`
        - :func:`time_span`
        
    """

    def __init__(self):
        """Initialize the DataBlock instance
        
        This class is intialized without any attributes.
        
        """
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
                species_str += ', ...'

            if dataset.file is not None:
                data_str += format_string.format(f'{dataset.name}', f'{dataset.category}', f'{species_str}',
                                                 f'{dataset.data.shape}', f'{dataset.file.name}')
            else:
                data_str += format_string.format(f'{dataset.name}', f'{dataset.category}', f'{species_str}',
                                                 f'{dataset.data.shape}', 'None')

        return data_str

    def __repr__(self):
        return self.__str__()

    def __iter__(self):
        for param, data in self.datasets.items():
            yield data

    def __len__(self):
        return len(self.datasets)

    def add_dataset(self, dataset):
        """Method to add a dataset to the DataBlock

        This takes in information regarding a dataset, creates the DataSet object, and adds the DataSet to
        the datasets attribute.
        
        :param DataSet dataset: A DataSet object to add to the DataBlock
        
        :return: None
        
        """
        self.datasets[dataset.name] = dataset

        return None

    @property
    def _check_duplicates(self):
        """Checks for duplications of data for a specific model element. This does not check spectral or trajectory
        data sets.

        :Raises:

            A ValueError if a component has more than one data set attributed to it.

        :return: None

        """
        all_data_cols = []
        for dataset in self.datasets.values():
            if dataset.category in ['spectral', 'trajectory']:
                continue
            all_data_cols.extend(dataset.data.columns)

        num_duplicates = len(all_data_cols) - len(set(all_data_cols))
        if num_duplicates != 0:
            raise ValueError('Duplicates ({num_duplicates}) in component, state, and algebraic data detected!')

        return None

    @property
    def names(self):
        """Returns the list of dataset names
        
        :return: List of names
        :rtype: list
        
        """
        return [dataset.name for dataset in self.datasets.values()]

    @property
    def time_span(self):
        """Determine the span of time from the beginning to the end of the 
        measured data
        
        :return: the start and end times
        :rtype: tuple
        
        """
        start_time = min([dataset.time_span[0] for dataset in self.datasets.values()])
        stop_time = max([dataset.time_span[1] for dataset in self.datasets.values()])

        return start_time, stop_time


class DataSet:
    """The specific data object contained in the DataBlock
    
    This class is not intended to be used directly by the user.
    
    :Methods:
        
        - :func:`remove_datasets`
        - :func:`show_data`
        - :func:`shape`
        - :func:`size`
        - :func:`species`
        - :func:`time_span`
        
    """

    def __init__(self,
                 name=None,
                 category=None,
                 data=None,
                 file=None,
                 ):

        self.name = name
        self.category = category
        self.data = data
        self.file = file
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
        without any data.
        
        :return: None
        
        """
        if self.data is not None:
            if not isinstance(self.data, pd.DataFrame):
                raise ValueError('Data must be a pandas DataFrame instance')

        elif self.data is None and self.file is not None:
            print(self.file)
            self.data = read_file(self.file)

        else:
            print('DataBlock object initialized without data')

        return None

    def remove_negatives(self):
        """Replaces the negative values in the dataframe with zeros
        
        :return: None
        
        """
        # self.data[self.data < 0] = 0
        self.data.mask(self.data < 0, 0)
        return None

    def show_data(self):
        """Method to show the data using Plotly
        
        This will first determine if the proper data has been provided for
        plotting and will then call the correct plotting method depending on
        the data category.
        
        :return: None
        
        """
        if self.data is None:
            print('No data to plot')
            return None

        if self.category in ['concentration', 'state', 'trajectory', 'custom']:
            self._plot_2d_data()

        if self.category == 'spectral':
            self._plot_spectral_data()

        return None

    def _plot_2d_data(self):
        """Simple plots for showing concentration or complementary state data
        
        :return: None
        
        """
        y_axis_text = 'Measured Value'
        if self.description is not None:
            y_axis_text = self.description

        if self.units is not None:
            y_axis_text = ' '.join([y_axis_text, '[' + self.units[0] + ']'])

        x_axis_text = 'Time'

        if self.units is not None and len(self.units) == 2:
            x_axis_text = ' '.join([x_axis_text, '[' + self.units[1] + ']'])

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

        fig.plot()

        return None

    def _plot_spectral_data(self):
        """ Plots spectral data in 3D plot
        
        :return: None
        
        """
        fig = go.Figure()
        fig.add_trace(go.Surface(x=self.data.columns,
                                 y=self.data.index,
                                 z=self.data.values,
                                 ))

        fig.update_layout(scene=dict(
            xaxis_title='Wavelength',
            yaxis_title='Time',
            zaxis_title='Spectra'),
            margin=dict(r=100, b=50, l=100, t=50),
            title_text=f'{self.name}: {self.category.capitalize()} data',
            title_font_size=30)

        fig.plot()

        return None

    # Properties
    @property
    def shape(self):
        """Get the shape of the data
        
        :return: shape data or None
        :rtype: tuple/None
        
        """
        if self.data is not None:
            return self.data.shape
        else:
            return None

    @property
    def size(self):
        """Get the size of the data
        
        :return: length of the data
        :rtype: int/None
        
        """
        if self.data is not None:
            return len(self.data.columns)
        else:
            return None

    @property
    def species(self):
        """Get the names of the elements in the dataset
        
        :return: Names of the data elements
        :rtype: list
        
        """
        if self.data is not None and self.category in ['concentration', 'state', 'trajectory', 'custom']:
            return list(self.data.columns)
        elif self.data is None:
            return []
        else:
            return []

    @property
    def time_span(self):
        """Get the time span of the measured data in the dataset
        
        :return: The start and end times of the measured data
        :rtype: tuple
        
        """
        return self.data.index.min(), self.data.index.max()


if __name__ == '__main__':
    pass
