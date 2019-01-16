# from kipet.model.TemplateBuilder import *
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import matplotlib as cm
import six

#=============================================================================
#-----------------------DATA READING AND WRITING TOOLS------------------------
#=============================================================================

def write_spectral_data_to_csv(filename,dataframe):
    """ Write spectral data Dij to csv file.
    
        Args:
            filename (str): name of output file
          
            dataframe (DataFrame): pandas DataFrame
        
        Returns:
            None

    """
    dataframe.to_csv(filename)

def write_spectral_data_to_txt(filename,dataframe):
    """ Write spectral data Dij to txt file.
    
        Args:
            filename (str): name of output file
          
            dataframe (DataFrame): pandas DataFrame
        
        Returns:
            None
    
    """
    f = open(filename,'w')
    for i in dataframe.index:
        for j in dataframe.columns:
            f.write("{0} {1} {2}\n".format(i,j,dataframe[j][i]))
    f.close()

def write_absorption_data_to_csv(filename,dataframe):
    """ Write absorption data Sij to csv file.
    
        Args:
            filename (str): name of output file
          
            dataframe (DataFrame): pandas DataFrame
        
        Returns:
            None

    """
    dataframe.to_csv(filename)

def write_absorption_data_to_txt(filename,dataframe):
    """ Write absorption data Sij to txt file.
    
        Args:
            filename (str): name of output file
          
            dataframe (DataFrame): pandas DataFrame
        
        Returns:
            None

    """
    f = open(filename,'w')
    for i in dataframe.index:
        for j in dataframe.columns:
            f.write("{0} {1} {2}\n".format(i,j,dataframe[j][i]))
    f.close()

def write_concentration_data_to_csv(filename,dataframe):
    """ Write concentration data Cij to csv file.
    
        Args:
            filename (str): name of output file
          
            dataframe (DataFrame): pandas DataFrame
        
        Returns:
            None

    """
    dataframe.to_csv(filename)

def write_concentration_data_to_txt(filename,dataframe):
    """ Write concentration data Cij to txt file.
    
        Args:
            filename (str): name of output file
          
            dataframe (DataFrame): pandas DataFrame
        
        Returns:
            None

    """
    f = open(filename,'w')
    for i in dataframe.index:
        for j in dataframe.columns:
            f.write("{0} {1} {2}\n".format(i,j,dataframe[j][i]))
    f.close()


def read_concentration_data_from_txt(filename):
    """ Reads txt with concentration data
    
        Args:
            filename (str): name of input file
          
        Returns:
            DataFrame

    """

    f = open(filename,'r')
    data_dict = dict()
    set_index = set()
    set_columns = set()

    for line in f:
        if line not in ['','\n','\t','\t\n']:
            l=line.split()
            i = float(l[0])
            j = l[1]
            k = float(l[2])
            set_index.add(i)
            set_columns.add(j)
            data_dict[i,j] = k
    f.close()
    
    data_array = np.zeros((len(set_index),len(set_columns)))
    sorted_index = sorted(set_index)
    sorted_columns = set_columns

    for i,idx in enumerate(sorted_index):
        for j,jdx in enumerate(sorted_columns):
            data_array[i,j] = data_dict[idx,jdx]

    return pd.DataFrame(data=data_array,columns=sorted_columns,index=sorted_index)
 
def read_concentration_data_from_csv(filename):
    """ Reads csv with concentration data
    
        Args:
            filename (str): name of input file
         
        Returns:
            DataFrame

    """
    data = pd.read_csv(filename,index_col=0)
    data.columns = [n for n in data.columns]
    return data    

def read_spectral_data_from_csv(filename):
    """ Reads csv with spectral data
    
        Args:
            filename (str): name of input file
         
        Returns:
            DataFrame

    """
    data = pd.read_csv(filename,index_col=0)
    data.columns = [float(n) for n in data.columns]
    return data

def read_absorption_data_from_csv(filename):
    """ Reads csv with spectral data
    
        Args:
            filename (str): name of input file
          
        Returns:
            DataFrame

    """
    data = pd.read_csv(filename,index_col=0)
    return data

def read_spectral_data_from_txt(filename):
    """ Reads txt with spectral data
    
        Args:
            filename (str): name of input file
          
        Returns:
            DataFrame

    """

    f = open(filename,'r')
    data_dict = dict()
    set_index = set()
    set_columns = set()

    for line in f:
        if line not in ['','\n','\t','\t\n']:
            l=line.split()
            i = float(l[0])
            j = float(l[1])
            k = float(l[2])
            set_index.add(i)
            set_columns.add(j)
            data_dict[i,j] = k
    f.close()
    
    data_array = np.zeros((len(set_index),len(set_columns)))
    sorted_index = sorted(set_index)
    sorted_columns = sorted(set_columns)

    for i,idx in enumerate(sorted_index):
        for j,jdx in enumerate(sorted_columns):
            data_array[i,j] = data_dict[idx,jdx]

    return pd.DataFrame(data=data_array,columns=sorted_columns,index=sorted_index)

def read_absorption_data_from_txt(filename):
    """ Reads txt with absorption data
    
        Args:
            filename (str): name of input file
          
        Returns:
            DataFrame

    """

    f = open(filename,'r')
    data_dict = dict()
    set_index = set()
    set_columns = set()

    for line in f:
        if line not in ['','\n','\t','\t\n']:
            l=line.split()
            i = float(l[0])
            j = l[1]
            k = float(l[2])
            set_index.add(i)
            set_columns.add(j)
            data_dict[i,j] = k
    f.close()
    
    data_array = np.zeros((len(set_index),len(set_columns)))
    sorted_index = sorted(set_index)
    sorted_columns = set_columns

    for i,idx in enumerate(sorted_index):
        for j,jdx in enumerate(sorted_columns):
            data_array[i,j] = data_dict[idx,jdx]

    return pd.DataFrame(data=data_array,columns=sorted_columns,index=sorted_index)


def plot_spectral_data(dataFrame,dimension='2D'):
    """ Plots spectral data
    
        Args:
            dataFrame (DataFrame): spectral data
          
        Returns:
            None

    """
    if dimension=='3D':
        lambdas = dataFrame.columns
        times = dataFrame.index
        D = np.array(dataFrame)
        L, T = np.meshgrid(lambdas, times)
        fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')
        #ax.plot_wireframe(L, T, D, rstride=10, cstride=10)
        ax = fig.gca(projection='3d')
        ax.plot_surface(L, T, D, rstride=10, cstride=10, alpha=0.2)
        #cset = ax.contour(L, T, D, zdir='z',offset=-10)
        cset = ax.contour(L, T, D, zdir='x',offset=-20,cmap='coolwarm')
        cset = ax.contour(L, T, D, zdir='y',offset=times[-1]*1.1,cmap='coolwarm')
        
        ax.set_xlabel('Wavelength')
        ax.set_xlim(lambdas[0]-20, lambdas[-1])
        ax.set_ylabel('time')
        ax.set_ylim(0, times[-1]*1.1)
        ax.set_zlabel('Spectra')
        #ax.set_zlim(-10, )


    else:
        plt.figure()
        plt.plot(dataFrame)

#=============================================================================
#--------------------------- DIAGNOSTIC TOOLS ------------------------
#=============================================================================
        
def basic_pca(dataFrame,n=None):
    """ Runs basic component analysis based on SVD
    
        Args:
            dataFrame (DataFrame): spectral data
            
            n (int): number of largest singular-values
            to plot

        Returns:
            None

    """    
    times = np.array(dataFrame.index)
    lambdas = np.array(dataFrame.columns)
    D = np.array(dataFrame)
    #print("D shape: ", D.shape)
    U, s, V = np.linalg.svd(D, full_matrices=True)
    #print("U shape: ", U.shape)
    #print("s shape: ", s.shape)
    #print("V shape: ", V.shape)
    #print("sigma/singular values", s)
    if n == None:
        print("WARNING: since no number of components is specified, all components are printed")
        print("It is advised to select the number of components for n")
        n_shape = s.shape
        n = n_shape[0]
        
    u_shape = U.shape
    #print("u_shape[0]",u_shape[0])
    n_l_vector = n if u_shape[0]>=n else u_shape[0]
    for i in range(n_l_vector):
        plt.plot(times,U[:,i])
    plt.xlabel("time")
    plt.ylabel("Components U[:,i]")
    plt.show()
    
    n_singular = n if len(s)>=n else len(s)
    idxs = range(n_singular)
    vals = [s[i] for i in idxs]
    plt.semilogy(idxs,vals,'o')
    plt.xlabel("i")
    plt.ylabel("singular values")
    plt.show()
    
    v_shape = V.shape
    n_r_vector = n if v_shape[0]>=n else v_shape[0]
    for i in range(n_r_vector):
        plt.plot(lambdas,V[i,:])
    plt.xlabel("wavelength")
    plt.ylabel("Components V[i,:]")
    plt.show
     
#=============================================================================
#---------------------------PROBLEM GENERATION TOOLS------------------------
#============================================================================= 
    
def gaussian_single_peak(wl,alpha,beta,gamma):
    """
    helper function to generate absorption data based on 
    lorentzian parameters
    """
    return alpha*np.exp(-(wl-beta)**2/gamma)

def absorbance(wl,alphas,betas,gammas):
    """
    helper function to generate absorption data based on 
    lorentzian parameters
    """
    return sum(gaussian_single_peak(wl,alphas[i],betas[i],gammas[i]) for i in range(len(alphas)))

def generate_absorbance_data(wl_span,parameters_dict):
    """
    helper function to generate absorption data based on 
    lorentzian parameters
    """
    components = parameters_dict.keys()
    n_components = len(components)
    n_lambdas = len(wl_span)
    array = np.zeros((n_lambdas,n_components))
    for i,l in enumerate(wl_span):
        j = 0
        for k,p in six.iteritems(parameters_dict):
            alphas = p['alphas']
            betas  = p['betas']
            gammas = p['gammas']
            array[i,j] = absorbance(l,alphas,betas,gammas)
            j+=1

    data_frame = pd.DataFrame(data=array,
                              columns = components,
                              index=wl_span)
    return data_frame


def generate_random_absorbance_data(wl_span,component_peaks,component_widths=None,seed=None):

    np.random.seed(seed)
    parameters_dict = dict()
    min_l = min(wl_span)
    max_l = max(wl_span)
    #mean=1000.0
    #sigma=1.5*mean
    for k,n_peaks in component_peaks.items():
        params = dict()
        if component_widths:
            width = component_widths[k]
        else:
            width = 1000.0
        params['alphas'] = np.random.uniform(0.1,1.0,n_peaks)
        params['betas'] = np.random.uniform(min_l,max_l,n_peaks)
        params['gammas'] = np.random.uniform(1.0,width,n_peaks)
        parameters_dict[k] = params

    return generate_absorbance_data(wl_span,parameters_dict)

def add_noise_to_signal(signal, size):
    """
    Adds a random normally distributed noise to a clean signal. Used mostly in Kipet
    To noise absorbances or concentration profiles obtained from simulations. All
    values that are negative after the noise is added are set to zero
    Args:
        signal (data): the Z or S matrix to have noise added to it
        size (scalar): sigma (or size of distribution)
    Returns:
        pandas dataframe
    """
    clean_sig = signal    
    noise = np.random.normal(0,size,clean_sig.shape)
    sig = clean_sig+noise    
    df= pd.DataFrame(data=sig)
    df[df<0]=0
    return df

#=============================================================================
#---------------------------PRE-PROCESSING TOOLS------------------------
#=============================================================================
    
def savitzky_golay(dataFrame, window_size, orderPoly, orderDeriv=0):
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

    data_frame = pd.DataFrame(data=no_noise,
                              columns = dataFrame.columns,
                              index=dataFrame.index)
    
    return data_frame

def snv(dataFrame, offset=0):
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
    return data_frame

def msc(dataFrame, reference_spectra=None):
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
    return data_frame