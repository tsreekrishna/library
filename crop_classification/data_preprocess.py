"""Importing neccessary packages"""
from copy import deepcopy
from itertools import combinations
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import pkg_resources
from torch.utils.data import TensorDataset, DataLoader
# from torch.utils.data import DataLoader
from torch import nn
import torch
from .src.models import RNNModel
from .src.utils import _sowing_period, _harvest_avg_impute, _harvest_period, _dip_impute, _less_than_150_drop


def data_preprocess(data: np.ndarray) -> (pd.DataFrame, pd.DataFrame):
    '''
    Preprocess the input data and filter non_crop data points

    Parameters
    -----------
    data : NDVI recorded every fortnight from the 1st fortnight 
           of october to 2nd fortnight of april (Must be a 2d array)

    Returns
    -------
    2 pandas dataframes -> preprocessed crop, non-crop

    '''
    fortnight_list = ['oct_1f', 'oct_2f', 'nov_1f', 'nov_2f', 'dec_1f', 'dec_2f', 'jan_1f', 'jan_2f'
                      'feb_1f', 'feb_2f', 'mar_1f', 'mar_2f', 'apr_1f', 'apr_2f']
    data = pd.DataFrame(data=data, columns=fortnight_list)

    outliers = data[data.loc[:,'oct_1f':'dec_2f'].apply(lambda row:any((i == 1)|(i == 0) for i in row), axis=1)]

    # Rows which have 0s/1s/NaNs in the possible sowing periods for data are dropped
    data.drop(outliers.index, inplace=True)

    # Imputing the possible harvest fns with the average of its immediate neighbours
    data = data.apply(_harvest_avg_impute, axis=1)
    
    new_outliers = data[data.loc[:,'jan_2f':'apr_1f'].apply(lambda row:any((i == 1)|(i == 0) for i in row), axis=1)]
    outliers = pd.concat([outliers, new_outliers])
    
    # if 0s and 1s still exit in the possible harvest periods, those rows are dropped
    data.drop(new_outliers.index, inplace=True)

    # Imputing the dec_1f, dec_2f and jan_1f fornights with the averages if the dip is not less than 30 from the adjs 
    data = data.apply(_dip_impute, axis=1)

    # Sowing period determination
    data['sowing_period'] = data.apply(_sowing_period, axis=1)

    new_outliers = data[data.sowing_period == 'Unknown']
    outliers = pd.concat([outliers, new_outliers])
    
    # Dropping the Unknown sp labels
    data.drop(new_outliers.index, inplace=True)

    data['harvest_period'] = data.apply(_harvest_period, axis=1)

    new_outliers = data[data.harvest_period == 'Unknown']
    outliers = pd.concat([outliers, new_outliers])
    
    # Dropping the Unknown harvest labels
    data.drop(new_outliers.index, inplace=True)

    new_outliers = data[data.apply(_less_than_150_drop, axis=1) == False]
    outliers = pd.concat([outliers, new_outliers])
    
    # Dropping the rows which have max of NDVI values less than 150 for all the values between sp and hp.
    data = data.drop(new_outliers.index)

    # Dropping the duplicates
    data = data.drop_duplicates()

    return data, outliers