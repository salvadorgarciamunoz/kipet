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

# Third party imports
import numpy as np
import pandas as pd

# KIPET library imports
#from kipet.post_model_build.pyomo_model_tools import _df_from_pyomo_data

# Constants
DEFAULT_DIR = pathlib.Path.cwd()

# Public methods that can be used in kipet
__all__ = ['write_data', 'read_data', 'add_noise_to_data']


def write_data(filename, data):
    """Method to write data to a file using KipetModel

    Convenient method to save modified data in a format ready to use with ReactionModels.

    :parameter str filename: The name of the file (plus relative directory)

    :parameter pandas.DataFrame data: The pandas DataFrame to be written to the file

    :returns: None
    
    """
    _filename = filename
    calling_file_name = os.path.dirname(os.path.realpath(sys.argv[0]))
    _filename = pathlib.Path(calling_file_name).joinpath(_filename)
    write_file(_filename, data)
    
    return None


def read_data(filename):
    """Method to read data file using KipetModel
    
    This is useful if you need to modify a datafile before using it with a ReactionModel.

    :parameter str filename: The name of the data file (expected to be in the data
       directory, otherwise use an absolute path).

    :return: The data read from the file
    :rtype: pandas.DataFrame
    
    """
    _filename = filename
    calling_file_name = os.path.dirname(os.path.realpath(sys.argv[0]))
    _filename = pathlib.Path(calling_file_name).joinpath(_filename)
    read_data = read_file(_filename)
    
    return read_data


def read_file(filename, directory=DEFAULT_DIR):       
    """ Reads data from a csv or txt file and converts it to a DataFrame
    
        Args:
            filename (str): name of input file (abs path)
          
        Returns:
            DataFrame

    """
    
    filename = pathlib.Path(filename)
    data_dict = {}
    
    if filename.suffix == '.txt':
        with open(filename, 'r') as f:
            for line in f:
                if line not in ['','\n','\t','\t\n']:
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
    """ Write data to file.
    
        Args:
            filename (str): name of output file (abs_path)
          
            dataframe (DataFrame): pandas DataFrame
        
            filetype (str): choice of output (csv, txt)
        
        Returns:
            None

    """
    # How can you write a general settings file/class/object?
    print(f'Here is the filename: {filename}')
    
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
                        f.write("{0} {1} {2}\n".format(i,j,dataframe[j][i]))
                        
    print(f'Data saved     : {filename}')
    return None

def read_spectral_data_from_csv(filename, instrument = False, negatives_to_zero = False):
    """ Reads csv with spectral data
    
        Args:
            filename (str): name of input file
            instrument (bool): if data is direct from instrument
            negatives_to_zero (bool): if data contains negatives and baseline shift is not
                                        done then this forces negative values to zero.

        Returns:
            DataFrame

    """
    data = pd.read_csv(filename, index_col=0)
    if instrument:
        #this means we probably have a date/timestamp on the columns
        data = pd.read_csv(filename,index_col=0, parse_dates = True)
        data = data.T
        for n in data.index:
            h,m,s = n.split(':')
            sec = (float(h)*60+float(m))*60+float(s)
            data.rename(index={n:sec}, inplace=True)
        data.index = [float(n) for n in data.index]
    else:
        data.columns = [float(n) for n in data.columns]

    #If we have negative values then this makes them equal to zero
    if negatives_to_zero:
        data[data < 0] = 0

    return data

def is_float_re(str_var):
    """Checks if a value is a float or not"""
    _float_regexp = re.compile(r"^[-+]?(?:\b[0-9]+(?:\.[0-9]*)?|\.[0-9]+\b)(?:[eE][-+]?[0-9]+\b)?$").match
    return True if _float_regexp(str_var) else False


def dict_to_df(data_dict):
    """Takes a dictionary of typical pyomo data and converts it to a dataframe
    
    """    
    dfs_stacked = pd.Series(index=data_dict.keys(), data=list(data_dict.values()))
    dfs = dfs_stacked.unstack()
    return dfs


def add_noise_to_data(data, noise):
    """Wrapper for adding noise to data after data has been added to
    the specific ReactionModel
    
    :parameter pandas.DataFrame data: The dataset to which noise is to be added.
    
    :parameter float noise: The variance of the added noise.
    
    :return: The dataset after noised has been added.
    :rtype: pandas.DataFrame

    """
    from kipet.common.prob_gen_tools import add_noise_to_signal
    noised_data = add_noise_to_signal(data, noise)
    
    return noised_data 