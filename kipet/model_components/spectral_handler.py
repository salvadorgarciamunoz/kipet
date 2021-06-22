"""
Spectral Data Handling for Kipet
"""
# Third party imports
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import plot


class SpectralData:
    """This class is used to handle the spectral data used in a ReactionModel
    
    Since spectral data is different from the state data and requires different
    methods to modify the data, a separate class was designed to house all of
    the spectra specific methods.
    
    :param str name: The name for the data set
    :param pandas.DataFrame data: The spectral data (D matrix)
    :param bool remove_negatives: Option to set negative values to zero

    :Methods:

        - :func:`add_data`
        - :func:`reset`
        - :func:`plot`
        - :func:`savitzky_golay`
        - :func:`snv`
        - :func:`msc`
        - :func:`baseline_shift`
        - :func:`decrease_wavelengths`
        - :func:`decrease_times`
    
    """

    def __init__(self, name, data=None, file=None, remove_negatives=False):
        """
        Initialize a SpectralData instance
    
        :param str name: The name for the data set
        :param pandas.DataFrame data: The spectral data (D matrix)
        :param bool remove_negatives: Option to set negative values to zero

        """
        self.name = name
        self.data = data
        self.file = file
        self.data_orig = data
        self.remove_negatives = remove_negatives

        if self.remove_negatives and self.data is not None:
            self._remove_negatives()

        self._check_columns()
        
        self._decreased_wavelengths = None
        self._decreased_times = None
        self._sg = None
        self._msc = None
        self._snv = None
        self._base = None
        self._negatives_removed = None
        

    def add_data(self, data):
        """Adds a dataset to a SpectralData instance.
        
        This is used only if the SpectralData instance is created without a
        data attribute (not being None). This handles setting up the data_orig
        attribute as well and is therefore better than simply using setattr.
        
        :param pandas.DataFrame data: The spectral data (D matrix)

        :return: None
        """
        setattr(self, 'data', data)

        if self.remove_negatives:
            self._remove_negatives()
        if self.data_orig is None:
            setattr(self, 'data_orig', data)

        self._check_columns()

        return None

    def _check_columns(self):
        """Ensures that the columns in the dataframes are floats

        :return: None
        """
        if hasattr(self, 'data') and self.data is not None:
            old_columns = self.data.columns
            new_columns = [float(col) for col in old_columns]
            self.data.columns = new_columns
        if hasattr(self, 'data_orig') and self.data_orig is not None:
            old_columns = self.data.columns
            new_columns = [float(col) for col in old_columns]
            self.data.columns = new_columns

        return None

    def reset(self):
        """Resets the data back to the originally supplied data

        :return: None
        """
        self.data = self.data_orig
        return None

    def plot(self, data_set='data'):
        """ Plots spectral data in 3D plot.
        
        Plots the modified or original data sets.
        
        :param pandas.DataFrame data_set: attribute name of the spectral data
              
        :return: None
        """
        data = getattr(self, data_set)

        fig = go.Figure()
        fig.add_trace(go.Surface(x=data.columns,
                                 y=data.index,
                                 z=data.values,
                                 ))

        fig.update_layout(scene=dict(
            xaxis_title='Wavelength',
            yaxis_title='Time',
            zaxis_title='Absorbance'),
            margin=dict(r=100, b=50, l=100, t=50),
            title_text=f'{self.name}: Spectral Data',
            title_font_size=30)

        plot(fig)
        return None

    def _remove_negatives(self):
        """Simple method to set negative values to zero in a dataframe

        :return: None

        """
        self.data[self.data < 0] = 0
        self._removed_negatives = True

    def savitzky_golay(self, window=3, poly=2, deriv=0, in_place=True):
        """Implementation of the Savitzky-Golay filter for Kipet. Used for smoothing data, with
        the option to also differentiate the data. Can be used to remove high-frequency noise.
        Creates a least-squares fit of data within each time window with a high order polynomial centered
        centered at the middle of the window of points.

        :param int window: The length of the window. Must be an odd integer number
        :param int poly: Order of the polynoial used in the filter. Should be less than window_size-1
        :param int deriv: (optional) The order of the derivative to compute (default = 0 means only smoothing)
        :param bool in_place: Option to update the data in place

        :return: DataFrame containing the smoothed data
        :rtype: pandas.DataFrame

        :References:
            This code is an amalgamation of those developed in the scipy.org cookbook and that employed in Matlab 
            by WeiFeng Chen.
            Original paper: A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of Data by 
            Simplified Least Squares Procedures. Analytical Chemistry, 1964, 36 (8), pp 1627-1639.

        """
        dataFrame = self.data
        try:
            window_size = np.abs(np.int(window))
            orderPoly = np.abs(np.int(poly))
            orderDeriv = np.abs(np.int(deriv))
        except ValueError:
            raise ValueError("window_size and order have to be of type int")
        if window_size % 2 != 1 or window_size < 1:
            raise TypeError("window_size size must be a positive odd number")
        if window_size < orderPoly + 2:
            raise TypeError("window_size is too small for the polynomials order")
        if orderPoly >= window_size:
            raise ValueError("polyorder must be less than window_length.")

        if not isinstance(dataFrame, pd.DataFrame):
            raise TypeError(
                "The data must be a pandas DataFrame, try using read_spectral_data_from_txt or similar function first")
        print("Applying the Savitzky-Golay filter")
        order_range = range(orderPoly + 1)
        half_window = (window_size - 1) // 2
        # precompute coefficients
        b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
        m = np.linalg.pinv(b).A[orderDeriv]
        D = np.array(dataFrame)
        no_noise = np.array(dataFrame)
        # pad the signal at the extremes with values taken from the signal itself
        for t in range(len(dataFrame.index)):
            row = list()
            for l in range(len(dataFrame.columns)):
                row.append(D[t, l])
            firstvals = row[0] - np.abs(row[1:half_window + 1][::-1] - row[0])
            lastvals = row[-1] + np.abs(row[-half_window - 1:-1][::-1] - row[-1])
            y = np.concatenate((firstvals, row, lastvals))
            new_row = np.convolve(m, y, mode='valid')
            no_noise[t] = new_row

        if orderDeriv == 0:
            for t in range(len(dataFrame.index)):
                for l in range(len(dataFrame.columns)):
                    if no_noise[t, l] < 0:
                        no_noise[t, l] = 0

        data_frame = pd.DataFrame(data=no_noise,
                                  columns=dataFrame.columns,
                                  index=dataFrame.index)

        if in_place:
            self.data = data_frame

        self._sg = {
            'Method': 'Sovitzky-Golay',
            'Window Size': window,
            'Polygon Order': poly,
            'Derivative Order': deriv,
            }

        return data_frame

    def snv(self, offset=0, in_place=True):
        """Implementation of the Standard Normal Variate (SNV) filter for Kipet which is a weighted normalization
        method that is commonly used to remove scatter effects in spectroscopic data, this pre-processing 
        step can be applied before the SG filter or used on its own. SNV can be sensitive to noisy entries 
        in the spectra and can increase nonlinear behaviour between S and C as it is not a linear transformation.

        :param float offset: User-defined offset which can be used to avoid over-normalization for samples
                            with near-zero standard deviation. Guide for choosing this value is for something 
                            near the expected noise level to be specified. Default value is zero.
        :param bool in_place: Option to update the data in place
            
        :return: DataFrame containing pre-processed data
        :rtype: pandas.DataFrame
    
        """
        dataFrame = self.data
        # data checks
        if not isinstance(dataFrame, pd.DataFrame):
            raise TypeError(
                "The data must be a pandas DataFrame, try using read_spectral_data_from_txt or similar function first")
        print("Applying the SNV pre-processing")
        D = np.array(dataFrame)
        snv_proc = np.array(dataFrame)
        for t in range(len(dataFrame.index)):
            row = list()
            sum_spectra = 0
            for l in range(len(dataFrame.columns)):
                row.append(D[t, l])
                sum_spectra += D[t, l]
            mean_spectra = sum_spectra / (len(dataFrame.columns))
            std = 0
            for l in range(len(dataFrame.columns)):
                std += (mean_spectra - D[t, l]) ** 2
            new_row = list()
            for l in range(len(dataFrame.columns)):
                if offset == 0:
                    w = (D[t, l] - mean_spectra) * (std / (len(dataFrame.columns) - 1)) ** 0.5
                else:
                    w = (D[t, l] - mean_spectra) * (std / (len(dataFrame.columns) - 1)) ** 0.5 + 1 / offset
                new_row.append(w)

            snv_proc[t] = new_row

        data_frame = pd.DataFrame(data=snv_proc,
                                  columns=dataFrame.columns,
                                  index=dataFrame.index)
        if in_place:
            self.data = data_frame
            
        self._snv = {
            'Method': 'Standard Normal Variate Filter',
            'Offset': offset,
            }
            
        return data_frame

    def msc(self, reference_spectra=None, in_place=True):
        """Implementation of the Multiplicative Scatter Correction (MSC) filter for Kipet which is simple pre-processing
        method that attempts to remove scaling effects and offset effects in spectroscopic data. This pre-processing 
        step can be applied before the SG filter or used on its own. This approach requires a reference spectrum which
        must be determined beforehand. In this implementation, the default reference spectrum is the average spectrum 
        of the dataset provided, however an optional argument exists for user-defined reference spectra to be provided.    
        
        :param pandas.DataFrame reference_spectra: Optional user-provided reference spectra argument. Default is to
            automatically determine this using the average spectra values
        :param bool in_place: Option to update the data in place

        :return: DataFrame pre-processed data
        :rtype: pandas.DataFrame

        """
        dataFrame = self.data
        if not isinstance(dataFrame, pd.DataFrame):
            raise TypeError(
                "The data must be a pandas DataFrame, try using read_spectral_data_from_txt or similar function first")
        print("Applying the MSC pre-processing")
        if reference_spectra != None:
            if not isinstance(reference_spectra, pd.DataFrame):
                raise TypeError(
                    "The data must be a pandas DataFrame, try using read_spectral_data_from_txt or similar function first")

            if len(dataFrame.columns) != len(reference_spectra.columns) and len(dataFrame.rows) != len(
                    reference_spectra.rows):
                raise NotImplementedError("the reference spectra must have the same number of entries as the data")

        D = np.array(dataFrame)
        ref = np.array(dataFrame)
        msc_proc = np.array(dataFrame)

        # the average spectrum is calculated as reference spectra for MSC when none is given by user
        if reference_spectra is None:
            sum_spectra = 0

            for t in range(len(dataFrame.index)):
                sum_spectra = 0
                for l in range(len(dataFrame.columns)):
                    sum_spectra += D[t, l]
                mean_spectra = sum_spectra / (len(dataFrame.columns))
                for l in range(len(dataFrame.columns)):
                    ref[t, l] = mean_spectra
        else:
            # should add in some checks and additional ways to formulate these depending on what input the user provides
            # need to find out the type of data usually inputted here in order to do this
            ref = reference_spectra
        for t in range(len(dataFrame.index)):
            row = list()
            fit = np.polyfit(ref[t, :], D[t, :], 1, full=True)
            row[:] = (D[t, :] - fit[0][1]) / fit[0][0]
            msc_proc[t, :] = row

        data_frame = pd.DataFrame(data=msc_proc,
                                  columns=dataFrame.columns,
                                  index=dataFrame.index)
        if in_place:
            self.data = data_frame
            
        self._msc = {
            'Method': 'Multiplicative Scatter Correction Filter',
            'Offset': 'Provided DataFrame of reference spectra',
            }
            
        return data_frame

    def baseline_shift(self, shift=None, in_place=True):
        """Implementation of basic baseline shift. 2 modes are avaliable: 1. Automatic mode that requires no
        user arguments. The method identifies the lowest value (NOTE THAT THIS ONLY WORKS IF LOWEST VALUE
        IS NEGATIVE) and shifts the spectra up until this value is at zero. 2. Baseline shift provided by
        user. User provides the number that is added to every wavelength value in the full spectral dataset.
        
        :param float shift: user-defined baseline shift
        :param bool in_place: Option to update the data in place
            
        :return: DataFrame containing pre-processed data
        :rtype: pandas.DataFrame

        """
        dataFrame = self.data
        # if dataFrame is None:
        #     dataFrame = self.data_orig
        # data checks
        if not isinstance(dataFrame, pd.DataFrame):
            raise TypeError(
                "The data must be a pandas DataFrame, try using read_spectral_data_from_txt or similar function first")
        print("Applying the baseline shift pre-processing")
        if shift is None:
            shift = float(dataFrame.min().min()) * (-1)

        print("shifting dataset by: ", shift)
        D = np.array(dataFrame)
        for t in range(len(dataFrame.index)):
            for l in range(len(dataFrame.columns)):
                D[t, l] = D[t, l] + shift

        data_frame = pd.DataFrame(data=D, columns=dataFrame.columns, index=dataFrame.index)

        if in_place:
            self.data = data_frame
            
        self._base = {
            'Method': 'Basic Baseline Shift',
            'Shift': shift,
            }
            
        return data_frame

    def decrease_wavelengths(self, A_set=2, specific_subset=None, in_place=True):
        """
        Takes in the original, full dataset and removes specific wavelengths, or only keeps every
        multiple of A_set. Returns a new, smaller dataset that should be easier to solve

        :param int A_set:  optional user-provided multiple of wavelengths to keep. i.e. if
                                        3, every third value is kept. Default is 2.
        :param array-like specific_subset: (optional) If the user already knows which wavelengths they would like to
                                        remove, then a list containing these can be included.
        param bool in_place: Option to update the data in place

        :return: DataFrame with the smaller dataset
        :rtype:

        """
        original_dataset = self.data
        if specific_subset is not None:
            if not isinstance(specific_subset, (list, dict)):
                raise RuntimeError("subset must be of type list or dict!")

            if isinstance(specific_subset, dict):
                lists1 = sorted(specific_subset.items())
                x1, y1 = zip(*lists1)
                specific_subset = list(x1)

            if isinstance(specific_subset, dict):
                subset = specific_subset.keys()
            else:
                subset = specific_subset
            new_D = original_dataset.loc[:, subset]
        else:
            count = 0
            for l in original_dataset.columns.values:
                remcount = count % A_set
                if remcount == 0:
                    original_dataset.drop(columns=[l], axis=1)
                count += 1
            new_D = original_dataset[original_dataset.columns[::A_set]]

        if in_place:
            self.data = new_D
            
        self._decreased_wavelengths = {
            'Method': 'Basic removal of specifc wavelength intervals',
            'Frequency': A_set if specific_subset is None else 'None',
            'Subset Used': 'Yes' if specific_subset is not None else 'No'
            }
        return new_D

    def decrease_times(self, A_set=2, in_place=True):
        """
        Takes in the original, full dataset and removes specific wavelengths, or only keeps every
        multiple of A_set. Returns a new, smaller dataset that should be easier to solve

        :param array-like A_set:  optional user-provided multiple of wavelengths to keep. i.e. if
                                        3, every third value is kept. Default is 2.
        param bool in_place: Option to update the data in place

        :return: DataFrame with the smaller dataset
        :rtype: pandas.DataFrame

        """
        original_dataset = self.data
        new_D = original_dataset[original_dataset.columns[::A_set]]
        if in_place:
            self.data = new_D
            
        self._decreased_times = {
            'Method': 'Basic removal of specifc time intervals',
            'Frequency': A_set,
            }
        return new_D
