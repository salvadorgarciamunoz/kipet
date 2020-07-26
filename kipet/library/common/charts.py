"""
Plotting functions for KIPET

"""
import matplotlib.pyplot as plt


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