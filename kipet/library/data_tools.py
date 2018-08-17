# from kipet.model.TemplateBuilder import *
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import matplotlib as cm
import six


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
    """ Reads txt with concnetration data
    
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
        cset = ax.contour(L, T, D, zdir='y',offset=times[-1]+20,cmap='coolwarm')
        
        ax.set_xlabel('Wavelength')
        ax.set_xlim(lambdas[0]-20, lambdas[-1])
        ax.set_ylabel('time')
        ax.set_ylim(0, times[-1]+20)
        ax.set_zlabel('Spectra')
        #ax.set_zlim(-10, )


    else:
        plt.figure()
        plt.plot(dataFrame)


def basic_pca(dataFrame,n=4):
    """ Runs basic component analysis based on SVD
    
        Args:
            dataFrame (DataFrame): spectral data
            
            n (int, optional): number of largest singular-values
            to plot

        Returns:
            None

    """
    times = np.array(dataFrame.index)
    lambdas = np.array(dataFrame.columns)
    D = np.array(dataFrame)
    U, s, V = np.linalg.svd(D, full_matrices=True)
    plt.subplot(1,2,1)
    u_shape = U.shape
    n_l_vector = n if u_shape[0]>=n else u_shape[0]
    for i in range(n_l_vector):
        plt.plot(times,U[:,i])
    plt.xlabel("time")
    plt.ylabel("Components U[:,i]")

    plt.subplot(1,2,2)
    n_singular = n if len(s)>=n else len(s)
    idxs = range(n_singular)
    vals = [s[i] for i in idxs]
    plt.semilogy(idxs,vals,'o')
    plt.xlabel("i")
    plt.ylabel("singular values")
    """
    plt.subplot(1,3,3)
    v_shape = V.shape
    n_r_vector = n if v_shape[0]>=n else v_shape[0]
    for i in range(n_r_vector):
        plt.plot(lambdas,V[i,:])
    plt.xlabel("wavelength")
    plt.ylabel("Components V[i,:]")
    """
        

    
def gausian_single_peak(wl,alpha,beta,gamma):
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
    return sum(gausian_single_peak(wl,alphas[i],betas[i],gammas[i]) for i in range(len(alphas)))

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
        orderDeriv (int): the order of the derivative to compute (default = 0 means only smoothing)
        
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
        
    order_range = range(orderPoly+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[orderDeriv]
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