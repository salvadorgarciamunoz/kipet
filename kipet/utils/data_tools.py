from kipet.model.TemplateBuilder import *
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib as cm

def write_spectral_data_to_csv(filename,dataframe):
    dataframe.to_csv(filename)

def write_spectral_data_to_txt(filename,dataframe):
    f = open(filename,'w')
    for i in dataframe.index:
        for j in dataframe.columns:
            f.write("{0} {1} {2}\n".format(i,j,dataframe[j][i]))
    f.close()

def write_absorption_data_to_csv(filename,dataframe):
    dataframe.to_csv(filename)

def write_absorption_data_to_txt(filename,dataframe):
    f = open(filename,'w')
    for i in dataframe.index:
        for j in dataframe.columns:
            f.write("{0} {1} {2}\n".format(i,j,dataframe[j][i]))
    f.close()
    
def read_spectral_data_from_csv(filename):
    data = pd.read_csv(filename,index_col=0)
    data.columns = [float(n) for n in data.columns]
    return data

def read_absorption_data_from_csv(filename):
    data = pd.read_csv(filename,index_col=0)
    return data

def read_spectral_data_from_txt(filename):
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
        cset = ax.contour(L, T, D, zdir='x',offset=-40,cmap='coolwarm')
        cset = ax.contour(L, T, D, zdir='y',offset=times[-1]+40,cmap='coolwarm')
        
        ax.set_xlabel('Wavelength')
        ax.set_xlim(-40, lambdas[-1])
        ax.set_ylabel('time')
        ax.set_ylim(0, times[-1]+40)
        ax.set_zlabel('Spectra')
        #ax.set_zlim(-10, )


    else:
        plt.figure()
        plt.plot(dataFrame)


def basic_pca(dataFrame,n=4):
    times = np.array(dataFrame.index)
    lambdas = np.array(dataFrame.columns)
    D = np.array(dataFrame)
    U, s, V = np.linalg.svd(D, full_matrices=True)
    plt.subplot(1,2,1)
    u_shape = U.shape
    n_l_vector = n if u_shape[0]>=n else u_shape[0]
    for i in xrange(n_l_vector):
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
    for i in xrange(n_r_vector):
        plt.plot(lambdas,V[i,:])
    plt.xlabel("wavelength")
    plt.ylabel("Components V[i,:]")
    """
        

    
