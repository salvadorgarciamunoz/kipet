"""
Model diagnostics
"""
import numpy as np

from kipet.model_tools.pyomo_model_tools import convert


def diagnostic_terms(exp, pred, num_params):
    """Calculates various model fit values using the experimental and
    predicted values
    
    :param pd.DataFrame exp: The experimental data
    :param pd.DataFrame pred: The predicted data from the model
    :param int num_params: The number of model parameters
    
    :return: Dict of various model fit values
    :rtype: dict
    
    """
    output = {}
    e = pred - exp
    n = exp.size
    
    # Sum of squared errors (also SSres)
    SSQ = (e ** 2).sum().sum()
    output['SSQ'] = (SSQ, 'Sum of squared errors')
    
    # Sum of absolute errors
    SAE = abs(e).sum().sum()
    output['SAE'] = (SAE, 'Sum of absolute errors')
    
    # Mean of SAE
    MAE = SAE/n
    output['MAE'] = (MAE, 'Mean absolute error')

    # Mean squared error
    MSE = SSQ/n
    output['MSE'] = (MSE, 'Mean squared error')

    # Root mean squared error
    RMSE = np.sqrt(MSE)
    output['RMSE'] = (RMSE, 'Root mean square error')
    
    # Relative absolute error
    y_bar = exp.sum().sum()/n
    y_bar_sum = abs(exp - y_bar).sum().sum()
    RAE = SAE/y_bar_sum
    output['RAE'] = (RAE, 'Relative absolute error')

    # Relative squared error
    SStot = ((exp - y_bar)**2).sum().sum()
    RSE = np.sqrt(SSQ/SStot)
    output['RSE'] = (RSE, 'Relative squared error')
    
    # R2
    R2 = 1 - SSQ/SStot
    output['R-sqaured'] = (R2, 'R squared')
    
    # Adj-R2
    R2_adj = 1 - (1 - R2)*(n - 1)/(n - num_params - 1)
    output['R-squared (adj)'] = (R2_adj, 'Adjusted R sqaured')
    
    # Residual Standard Error
    S = np.sqrt(SSQ/(n - num_params))
    output['S'] = (S, 'Residual standard error')
    
    return output


def model_fit(parameter_estimator):
    """ Runs basic post-processing lack of fit analysis

    :param ParameterEstimator parameter_estimator: The parameter estimator object after solving

    :return: Various model fit values
    :rtype: dict
    
    """
    model = parameter_estimator.model
    num_params = len(parameter_estimator.param_names)
    
    if hasattr(model, 'C'):
    
        C = convert(model.C)
        S = convert(model.S)
        C_red = C.loc[:, S.columns]
        exp = convert(model.D)
        pred = C_red.dot(S.T)
        
    elif hasattr(model, 'Cm'):

        exp = convert(model.Cm)
        raw_pred = convert(model.Z)
        pred = raw_pred.loc[exp.index]
        
    output = diagnostic_terms(exp, pred, num_params)
        
    return output