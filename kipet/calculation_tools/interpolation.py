"""
This module contains functions for ensuring that variables have congruent time indicies.

For many users who may have delved into older versions of KIPET, these were formerly found in the PyomoSimulator and
Simulator classes.
"""
# Standard library imports
import bisect

# Third party imports
import numpy as np
import pandas as pd


def interpolate_trajectory(t, tr):
    """Method for interpolating trajectories

    :param list t: the list of times
    :param pandas.DataFrame tr: The trajectory data

    :return float tr_val: The interpolated value

    """
    tr_val = np.zeros((len(t)))
    times = [float(ti) for ti in t]
    indx = tr.index.astype(float)

    for i, t_indx in enumerate(times):

        if i == 0:
            tr_val[i] = tr.iloc[0]
            continue

        indx_left = bisect.bisect_left(indx[1:], times[i])
        if indx_left == len(tr) - 1:
            tr_val[i] = tr.iloc[indx_left]
            continue

        dx = indx[indx_left + 1] - indx[indx_left]
        dy = tr.iloc[indx_left + 1] - tr.iloc[indx_left]
        m = dy / dx
        val = tr.iloc[indx_left] + (times[i] - indx[indx_left]) * m
        tr_val[i] = val

    return tr_val


def interpolate_trajectory2(t, data):
    """Takes some data and fills in the missing points using interpolation

    :param float t: The time point
    :param pandas.DataFrame data: The data to interpolate with

    :return df_interpolated: The data after interpolation
    :rtype: pandas.DataFrame

    """
    data = pd.DataFrame(data)
    df_interpolated = pd.DataFrame(columns=data.columns)
    
    for col in data.columns:
        
        tr = data[col]
        tr_val = np.zeros((len(t)))
        times = [float(ti) for ti in t]
        indx = tr.index.astype(float)
    
        for i, t_indx in enumerate(times):
    
            if i == 0:
                tr_val[i] = tr.iloc[0]
                continue        
    
            indx_left = bisect.bisect_left(indx[1:], times[i])
            if indx_left == len(tr) - 1:
                tr_val[i] = tr.iloc[indx_left]
                continue
            
            dx = indx[indx_left + 1] - indx[indx_left]
            dy = tr.iloc[indx_left + 1] - tr.iloc[indx_left]
            m = dy/dx
            val = tr.iloc[indx_left] +  (times[i] - indx[indx_left]) * m
            tr_val[i] = val
    
        df_interpolated[col] = tr_val
    
    return df_interpolated