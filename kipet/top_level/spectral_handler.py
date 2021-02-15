"""
Spectral Data Handling for Kipet
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

class SpectralData():
    
    """All of the spectral tools will be moved here"""   
 
    def __init__(self, name, data=None, remove_negatives=False):
        
        self.name = name
        self.data = data
        self.data_orig = data
        self.remove_negatives=remove_negatives
        
        if self.remove_negatives and self.data is not None:
            self._remove_negatives()
        
        self._check_columns()
        
    def add_data(self, data):
        
        setattr(self, 'data', data)

        if self.remove_negatives:
            self._remove_negatives()
        if self.data_orig is None:
            setattr(self, 'data_orig', data)
            
        self._check_columns()
        
    def _check_columns(self):
        
        if hasattr(self, 'data') and self.data is not None:
            old_columns = self.data.columns
            new_columns = [float(col) for col in old_columns]
            self.data.columns = new_columns
        if hasattr(self, 'data_orig') and self.data_orig is not None:
            old_columns = self.data.columns
            new_columns = [float(col) for col in old_columns]
            self.data.columns = new_columns
            
    def reset(self):
        
        self.data = self.data_orig
        return None
        
    def plot(self, data_set='data'):
        """ Plots spectral data
        
            Args:
                dataFrame (DataFrame): spectral data
              
            Returns:
                None
    
        """
        data = getattr(self, data_set)
       
        fig = go.Figure()
        fig.add_trace(go.Surface(x=data.columns,
                           y=data.index,
                           z=data.values,
                          ))
        
        fig.update_layout(scene = dict(
                            xaxis_title='Wavelength',
                            yaxis_title='Time',
                            zaxis_title='Absorbance'),
                            margin=dict(r=100, b=50, l=100, t=50),
                            title_text=f'{self.name}: Spectral Data',
                            title_font_size=30)
        
        plot(fig)
            
        return None
    
    def _remove_negatives(self):
        
        self.data[self.data < 0] = 0
    
    def savitzky_golay(self, window_size=3, orderPoly=2, orderDeriv=0, in_place=True):
        """
        Implementation of the Savitzky-Golay filter for Kipet. Used for smoothing data, with
        the option to also differentiate the data. Can be used to remove high-frequency noise.
        Creates a least-squares fit of data within each time window with a high order polynomial centered
        centered at the middle of the window of points.
        
        Args:
            dataFrame (DataFrame): the data to be smoothed (either concentration or spectral data)
            window_size (int): the length of the window. Must be an odd integer number
            orderPoly (int): order of the polynoial used in the filter. Should be less than window_size-1
            orderDeriv (int) (optional): the order of the derivative to compute (default = 0 means only smoothing)
            
        Returns:
            DataFrame containing the smoothed data
        
        References:
            This code is an amalgamation of those developed in the scipy.org cookbook and that employed in Matlab 
            by WeiFeng Chen.
            Original paper: A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of Data by 
            Simplified Least Squares Procedures. Analytical Chemistry, 1964, 36 (8), pp 1627-1639.
        """
        # data checks
        dataFrame = self.data
            #dataFrame = self.data_orig
        try:
            window_size = np.abs(np.int(window_size))
            orderPoly = np.abs(np.int(orderPoly))
        except ValueError:
            raise ValueError("window_size and order have to be of type int")
        if window_size % 2 != 1 or window_size < 1:
            raise TypeError("window_size size must be a positive odd number")
        if window_size < orderPoly + 2:
            raise TypeError("window_size is too small for the polynomials order")    
        if orderPoly >= window_size:
            raise ValueError("polyorder must be less than window_length.")
    
        if not isinstance(dataFrame, pd.DataFrame):
            raise TypeError("data must be inputted as a pandas DataFrame, try using read_spectral_data_from_txt or similar function first")
        print("Applying the Savitzky-Golay filter")
        
        order_range = range(orderPoly+1)
        half_window = (window_size -1) // 2
        # precompute coefficients
        b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
        m = np.linalg.pinv(b).A[orderDeriv]
        #rate = 1
        #m = np.linalg.pinv(b).A[orderDeriv] * rate**orderDeriv * factorial(orderDeriv)
        D = np.array(dataFrame)
        no_noise = np.array(dataFrame)
        # pad the signal at the extremes with values taken from the signal itself
        for t in range(len(dataFrame.index)):
            row = list()
            for l in range(len(dataFrame.columns)):
                row.append(D[t,l])
            firstvals = row[0] - np.abs( row[1:half_window+1][::-1] - row[0] )
            lastvals = row[-1] + np.abs(row[-half_window-1:-1][::-1] - row[-1])
            y = np.concatenate((firstvals, row, lastvals))
            new_row = np.convolve( m, y, mode='valid')
            no_noise[t]=new_row
            
        if orderDeriv == 0:
            for t in range(len(dataFrame.index)):
                for l in range(len(dataFrame.columns)):
                    if no_noise[t,l] < 0:
                        no_noise[t,l] = 0
        
        data_frame = pd.DataFrame(data=no_noise,
                                  columns = dataFrame.columns,
                                  index=dataFrame.index)
        
        if in_place:
            self.data = data_frame
        
        return data_frame
    
    def snv(self, offset=0, in_place=True):
        """
        Implementation of the Standard Normal Variate (SNV) filter for Kipet which is a weighted normalization
        method that is commonly used to remove scatter effects in spectroscopic data, this pre-processing 
        step can be applied before the SG filter or used on its own. SNV can be sensitive to noisy entries 
        in the spectra and can increase nonlinear behaviour between S and C as it is not a linear transformation.
        
        
        Args:
            dataFrame (DataFrame): the data to be processed (either concentration or spectral data)
            offset (float): user-defined offset which can be used to avoid over-normalization for samples
                            with near-zero standard deviation. Guide for choosing this value is for something 
                            near the expected noise level to be specified. Default value is zero.
            
        Returns:
            DataFrame containing pre-processed data
        
        References:
    
        """
        dataFrame = self.data
        # if dataFrame is None:
        #     dataFrame = self.data_orig
            
        # data checks
        if not isinstance(dataFrame, pd.DataFrame):
            raise TypeError("data must be inputted as a pandas DataFrame, try using read_spectral_data_from_txt or similar function first")
        print("Applying the SNV pre-processing")    
    
        D = np.array(dataFrame)
        snv_proc = np.array(dataFrame)
        for t in range(len(dataFrame.index)):
            row = list()
            sum_spectra = 0
            for l in range(len(dataFrame.columns)):
                row.append(D[t,l])
                sum_spectra += D[t,l]
            mean_spectra = sum_spectra/(len(dataFrame.columns))
            std = 0
            for l in range(len(dataFrame.columns)):
                std += (mean_spectra-D[t,l])**2
            new_row = list()
            for l in range(len(dataFrame.columns)):
                if offset ==0:
                    w = (D[t,l]-mean_spectra)*(std/(len(dataFrame.columns)-1))**0.5
                else:
                    w = (D[t,l]-mean_spectra)*(std/(len(dataFrame.columns)-1))**0.5 + 1/offset
                new_row.append(w)
                    
            snv_proc[t]=new_row
    
        data_frame = pd.DataFrame(data=snv_proc,
                                  columns = dataFrame.columns,
                                  index=dataFrame.index)
        if in_place:
            self.data = data_frame
        return data_frame
    
    def msc(self, reference_spectra=None, in_place=True):
        """
        Implementation of the Multiplicative Scatter Correction (MSC) filter for Kipet which is simple pre-processing
        method that attempts to remove scaling effects and offset effects in spectroscopic data. This pre-processing 
        step can be applied before the SG filter or used on its own. This approach requires a reference spectrum which
        must be determined beforehand. In this implementation, the default reference spectrum is the average spectrum 
        of the dataset provided, however an optional argument exists for user-defined reference spectra to be provided.    
        
        Args:
            dataFrame (DataFrame):          the data to be processed (either concentration or spectral data)
            reference_spectra (DataFrame):  optional user-provided reference spectra argument. Default is to automatically
                                            determine this using the average spectra values.
            
        Returns:
            DataFrame pre-processed data
        
        References:
    
        """
        dataFrame = self.data
        # if dataFrame is None:
        #     dataFrame = self.data_orig
        # data checks
        if not isinstance(dataFrame, pd.DataFrame):
            raise TypeError("data must be inputted as a pandas DataFrame, try using read_spectral_data_from_txt or similar function first")
        print("Applying the MSC pre-processing")  
        
        #Want to make it possible to include user-defined reference spectra
        #this is not great as we could provide the data with some conditioning 
        #in order to construct references based on different user inputs
        if reference_spectra != None:
            if not isinstance(reference_spectra, pd.DataFrame):
                raise TypeError("data must be inputted as a pandas DataFrame, try using read_spectral_data_from_txt or similar function first")
            
            if len(dataFrame.columns) != len(reference_spectra.columns) and len(dataFrame.rows) != len(reference_spectra.rows):
                raise NotImplementedError("the reference spectra must have the same number of entries as the data")
        
        D = np.array(dataFrame)
        ref = np.array(dataFrame)
        msc_proc = np.array(dataFrame)
        
        # the average spectrum is calculated as reference spectra for MSC when none is given by user
        if reference_spectra == None:
            sum_spectra = 0
            
            for t in range(len(dataFrame.index)):
                sum_spectra = 0
                for l in range(len(dataFrame.columns)):
                    sum_spectra += D[t,l]
                mean_spectra = sum_spectra/(len(dataFrame.columns))
                for l in range(len(dataFrame.columns)):
                    ref[t,l] = mean_spectra 
        else:
            #should add in some checks and additional ways to formulate these depending on what input the user provides
            #need to find out the type of data usually inputted here in order to do this
            ref = reference_spectra
        for t in range(len(dataFrame.index)):
            row = list()
            fit = np.polyfit(ref[t,:],D[t,:],1, full=True)
            row[:] = (D[t,:] - fit[0][1]) / fit[0][0]
            msc_proc[t,:]=row  
    
        data_frame = pd.DataFrame(data=msc_proc,
                                  columns = dataFrame.columns,
                                  index=dataFrame.index)
        if in_place:
            self.data = data_frame
        return data_frame
    
    def baseline_shift(self, shift=None, in_place=True):
        """
        Implementation of basic baseline shift. 2 modes are avaliable: 1. Automatic mode that requires no
        user arguments. The method identifies the lowest value (NOTE THAT THIS ONLY WORKS IF LOWEST VALUE
        IS NEGATIVE) and shifts the spectra up until this value is at zero. 2. Baseline shift provided by
        user. User provides the number that is added to every wavelength value in the full spectral dataset.
        
        
        Args:
            dataFrame (DataFrame): the data to be processed (spectral data)
            shift (float, optional): user-defined baseline shift
            
        Returns:
            DataFrame containing pre-processed data
        
        References:
    
        """
        dataFrame = self.data
        # if dataFrame is None:
        #     dataFrame = self.data_orig
        # data checks
        if not isinstance(dataFrame, pd.DataFrame):
            raise TypeError("data must be inputted as a pandas DataFrame, try using read_spectral_data_from_txt or similar function first")
        print("Applying the baseline shift pre-processing") 
        if shift == None:
            shift = float(dataFrame.min().min())*(-1)
        
        print("shifting dataset by: ", shift)    
        D = np.array(dataFrame)
        for t in range(len(dataFrame.index)):
            for l in range(len(dataFrame.columns)):
                D[t,l] = D[t,l]+shift
        
        data_frame = pd.DataFrame(data=D, columns = dataFrame.columns, index = dataFrame.index)
        
        if in_place:
            self.data = data_frame
        return data_frame
    
    def decrease_wavelengths(self, A_set=2, specific_subset=None, in_place=True):
        '''
        Takes in the original, full dataset and removes specific wavelengths, or only keeps every
        multiple of A_set. Returns a new, smaller dataset that should be easier to solve
        
        Args:
            original_dataset (DataFrame):   the data to be processed
            A_set (float, optional):  optional user-provided multiple of wavelengths to keep. i.e. if
                                        3, every third value is kept. Default is 2.
            specific_subset (list or dict, optional): If the user already knows which wavelengths they would like to
                                        remove, then a list containing these can be included.
            
        Returns:
            DataFrame with the smaller dataset
        
        '''
        original_dataset = self.data
        # if original_dataset is None:
        #     original_dataset = self.data_orig
            
        if specific_subset != None:
            if not isinstance(specific_subset, (list, dict)):
                raise RuntimeError("subset must be of type list or dict!")
                 
            if isinstance(specific_subset, dict):
                lists1 = sorted(specific_subset.items())
                x1, y1 = zip(*lists1)
                specific_subset = list(x1)
                
            new_D = pd.DataFrame(np.nan,index=original_dataset.index, columns = specific_subset)
            for t in original_dataset.index:
                for l in original_dataset.columns.values:
                    if l in subset:
                        new_D.at[t,l] = self.model.D[t,l]           
        else:
            count=0
            for l in original_dataset.columns.values:
                remcount = count%A_set
                if remcount==0:
                    original_dataset.drop(columns=[l],axis = 1)
                count+=1
            new_D = original_dataset[original_dataset.columns[::A_set]] 
            
        if in_place:
            self.data = new_D
        return new_D