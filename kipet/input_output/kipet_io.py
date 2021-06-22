"""
This module contains the main methods used for reading, writing, and manipulating
data.

The read_data and write_data methods are accessible at the top level (i.e. kipet.read_data)
"""
# Standard library imports
import os
import pathlib
import re
import sys
import traceback

# Third party imports
import numpy as np
import pandas as pd


# KIPET library imports
# from kipet.model_tools.pyomo_model_tools import _df_from_pyomo_data

# Public methods that can be used in KIPET
__all__ = ['write_data', 'read_data', 'read_file', 'add_noise_to_data', 'Tee']


# Context manager that copies stdout and any exceptions to a log file
class Tee:
    
    def __init__(self, filename):
    
        self.file = open(filename, 'a+')
        self.stdout = sys.stdout
        #self.stderr = sys.stderr
    
    def __enter__(self):
    
        sys.stdout = self
    
    def __exit__(self, exc_type, exc_value, tb):
    
        sys.stdout = self.stdout
        if exc_type is not None:
            self.file.write(traceback.format_exc())
        self.file.close()
    
    def write(self, data):
        
        self.file.write(data)
        self.stdout.write(data)
    
    def flush(self):
        
        self.file.flush()
        self.stdout.flush()

def write_data(filename, data):
    """Method to write data to a file using KIPET

    Convenient method to save modified data in a format ready to use with ReactionModels.

    :parameter str filename: The name of the file (plus relative directory)
    :parameter pandas.DataFrame data: The pandas DataFrame to be written to the file

    :returns: None
    
    """
    calling_file_name = os.path.dirname(os.path.realpath(sys.argv[0]))
    write_file(pathlib.Path(calling_file_name).joinpath(filename), data)

    return None


def read_data(filename):
    """Method to read data file using KipetModel
    
    This is useful if you need to modify a datafile before using it with a ReactionModel.

    :parameter str filename: The name of the data file (expected to be in the data
       directory, otherwise use an absolute path).

    :return loaded_data: The data read from the file
    :rtype: pandas.DataFrame
    
    """
    _filename = filename
    calling_file_name = os.path.dirname(os.path.realpath(sys.argv[0]))
    _filename = pathlib.Path(calling_file_name).joinpath(_filename)
    loaded_data = read_file(_filename)

    return loaded_data


def read_file(filename):
    """ Reads data from a csv or txt file and converts it to a DataFrame

        :param str filename: name of input file (abs path)
          
        :return df_data: DataFrame
        :rtype: pandas.DataFrame

        :Raises:

            ValueError if the file type is not CSV or TXT

    """
    filename = pathlib.Path(filename)
    data_dict = {}

    if filename.suffix == '.txt':
        with open(filename, 'r') as f:
            for line in f:
                if line not in ['', '\n', '\t', '\t\n']:
                    l = line.split()
                    if is_float_re(l[1]):
                        l[1] = float(l[1])
                    data_dict[float(l[0]), l[1]] = float(l[2])

        df_data = dict_to_df(data_dict)
        df_data.sort_index(ascending=True, inplace=True)

    elif filename.suffix == '.csv':
        df_data = pd.read_csv(filename, index_col=0)

    else:
        raise ValueError(f'The file extension {filename.suffix} is currently not supported')
        return None

    return df_data


def write_file(filename, dataframe, filetype='csv'):
    """ Write data from a dataframe to file.
    
        :param str filename: Name of output file (abs_path)
        :param pandas.DataFrame dataframe: Data to write to file
        :param str filetype: Choice of file output (CSV, TXT)
        
        :return: None

    """
    if filetype not in ['csv', 'txt']:
        print('Savings as CSV - invalid file extension given')
        filetype = 'csv'

    suffix = '.' + filetype

    filename = pathlib.Path(filename)

    if filename.suffix == '':
        filename = filename.with_suffix(suffix)
    else:
        suffix = filename.suffix
        if suffix not in ['.txt', '.csv']:
            print('Savings as CSV - invalid file extension given')
            filename = pathlib.Path(filename.stem).with_suffix('.csv')

    print(f'In write method: {filename.absolute()}')
    if filename.suffix == '.csv':
        dataframe.to_csv(filename)

    elif filename.suffix == '.txt':
        with open(filename, 'w') as f:
            for i in dataframe.index:
                for j in dataframe.columns:
                    if not np.isnan(dataframe[j][i]):
                        f.write("{0} {1} {2}\n".format(i, j, dataframe[j][i]))

    print(f'Data saved     : {filename}')
    return None


def read_spectral_data_from_csv(filename, instrument=False, negatives_to_zero=False):
    """ Reads csv with spectral data

        :param str filename: Name of input file
        :param bool instrument: Indicates if data is direct from instrument
        :param bool negatives_to_zero: If data contains negatives and baseline shift is not
                                        done then this forces negative values to zero.

        :return data: The dataframe of spectral data after formatting
        :rtype: pandas.DataFrame

    """
    data = pd.read_csv(filename, index_col=0)
    if instrument:
        # this means we probably have a date/timestamp on the columns
        data = pd.read_csv(filename, index_col=0, parse_dates=True)
        data = data.T
        for n in data.index:
            h, m, s = n.split(':')
            sec = (float(h) * 60 + float(m)) * 60 + float(s)
            data.rename(index={n: sec}, inplace=True)
        data.index = [float(n) for n in data.index]
    else:
        data.columns = [float(n) for n in data.columns]

    # If we have negative values then this makes them equal to zero
    if negatives_to_zero:
        data[data < 0] = 0

    return data


def is_float_re(str_var):
    """Checks if a value is a float or not

    :param str str_var: Checks if the string value is a float

    :return: boolean indicating if the string can be considered a float
    :rtype: bool

    """
    _float_regexp = re.compile(r"^[-+]?(?:\b[0-9]+(?:\.[0-9]*)?|\.[0-9]+\b)(?:[eE][-+]?[0-9]+\b)?$").match
    return True if _float_regexp(str_var) else False


def dict_to_df(data_dict):
    """Takes a dictionary of typical pyomo data and converts it to a dataframe

    :param dict data_dict: dict of the Pyomo data

    :return dfs: DataFrame of the unpacked data
    :rtype: pandas.DataFrame
    
    """
    dfs_stacked = pd.Series(index=data_dict.keys(), data=list(data_dict.values()))
    dfs = dfs_stacked.unstack()
    return dfs


def add_noise_to_data(data, noise):
    """Wrapper for adding noise to data after data has been added to
    the specific ReactionModel
    
    :parameter pandas.DataFrame data: The dataset to which noise is to be added.
    :parameter float noise: The variance of the added noise.
    
    :return noised_data: The dataset after noised has been added.
    :rtype: pandas.DataFrame

    """
    from kipet.calculation_tools.prob_gen_tools import add_noise_to_signal
    noised_data = add_noise_to_signal(data, noise)

    return noised_data
