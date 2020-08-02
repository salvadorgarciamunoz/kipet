"""
Plotting functions for KIPET

"""
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
import numpy as np
import pandas as pd


COLOR_SET = [
            [0, 0.4470, 0.7410], 	          
          	[0.8500, 0.3250, 0.0980], 	          	
          	[0.9290, 0.6940, 0.1250], 	          	
          	[0.4940, 0.1840, 0.5560], 	          	
          	[0.4660, 0.6740, 0.1880], 	          	
          	[0.3010, 0.7450, 0.9330],
          	[0.6350, 0.0780, 0.1840], 	          	
    ]
    
def make_plot(results_object, data, *args, **kwargs):
    """Make a plot of the results
    
    Parameters:
    
        results_object: A kipet ResultsObject
            The results of the parameter fitting
        
        data: str
            The variable name of the data (concentration = C, etc.)
            
            
    Returns:
        None
        
    """    
    # Figure options
    xlabel = kwargs.get('xlabel', 'Time (Unit)')    
    ylabel = kwargs.get('ylabel', 'Concentration (Unit)')
    title = kwargs.get('title', 'Concentration Profiles')
    figsize = kwargs.get('figsize', (30, 10))    
    fontsize = kwargs.get('fontsize', 14)

    # Data variable
    data_type = {
        'C' : ['C', 'Z'],
        #'S' : ['S', 'D'],
        }
        
    # Make plot
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, col in enumerate(getattr(results_object, data_type[data][1]).columns):

        i = divmod(i, 7)[1]
        # Plot the experimental data
        if col in getattr(results_object, data_type[data][0]).columns:
            plot_data = getattr(results_object, data_type[data][0])[col].dropna()
            ax.scatter(plot_data.index.values, plot_data.values, marker='o', color=COLOR_SET[i], label=col + f' ({data_type[data][0]})')
        
        # Plot the predicted data
        plot_data = getattr(results_object, data_type[data][1])[col].dropna()
        ax.plot(plot_data.index.values, plot_data.values, color=COLOR_SET[i], label=col + f' ({data_type[data][1]})')
        
    # Plot annotations
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    plt.legend()
    plt.show()
    
    return None


def plot_spectral_data(dataFrame, dimension='2D'):
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