"""
Plotting functions for KIPET

"""
import matplotlib.pyplot as plt
import pandas as pd


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
    xlabel = kwargs.get('xlabel', 'time(s)')    
    ylabel = kwargs.get('ylabel', 'Concentration (mol/L)')
    title = kwargs.get('title', 'Concentration Profile')
    figsize = kwargs.get('figsize', (30, 10))    

    # Data variable
    data_type = {
        'C' : ['C', 'Z'],
        #'S' : ['S', 'D'],
        }
        
    # Make plot
    fig, ax = plt.subplots(figsize=figsize)
    
    for col in getattr(results_object, data_type[data][0]).columns:

        # Plot the experimental data
        plot_data = getattr(results_object, data_type[data][0])[col].dropna()
        ax.scatter(plot_data.index.values, plot_data.values, marker='o')
        
        # Plot the predicted data
        plot_data = getattr(results_object, data_type[data][1])[col].dropna()
        ax.plot(plot_data.index.values, plot_data.values, label=col)
        
    # Plot annotations
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
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