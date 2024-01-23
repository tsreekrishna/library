import pkg_resources
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from mapie.classification import MapieClassifier
import torch
from crop_classification.src.models import RNNModel, model_prediction
from crop_classification.src.utils.helper import _sowing_period, _harvest_period, _dip_impute, _less_than_150_drop, _generate_label_set_map
from crop_classification.src.utils.checks import data_check, algo_check, classifier_type_check, batch_size_check, alpha_check
from crop_classification.src.utils.data_loader import _load_models, _load_scalers

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
    # Checking arguments for exceptions
    data_check(data)

    fortnight_list = ['oct_1f', 'oct_2f', 'nov_1f', 'nov_2f', 'dec_1f', 'dec_2f', 'jan_1f', 'jan_2f',
                      'feb_1f', 'feb_2f', 'mar_1f', 'mar_2f', 'apr_1f', 'apr_2f']
    data = pd.DataFrame(data=data, columns=fortnight_list)
    outliers = pd.DataFrame(columns=fortnight_list)

    # Imputing the NDVI fornights with the averages if the dip is greater than 20 when compared to both adjs 
    data = data.apply(_dip_impute, axis=1)

    # Determining sowing period(S.P). If the S.P is not found, then it is regarded as a non crop. 
    data['sowing_period'] = data.apply(_sowing_period, axis=1)
    new_outliers = data[data.sowing_period == 'Unknown']
    outliers = pd.concat([outliers, new_outliers])
    data.drop(new_outliers.index, inplace=True)

    # Determining harvest period(H.P). If the H.P is not found, then it is regarded as a non crop.
    data['harvest_period'] = data.apply(_harvest_period, axis=1)
    new_outliers = data[data.harvest_period == 'Unknown']
    outliers = pd.concat([outliers, new_outliers])
    data.drop(new_outliers.index, inplace=True)

    # Dropping the rows which have max of NDVI values less than 150 for all the values between sp and hp.
    new_outliers = data[data.apply(_less_than_150_drop, axis=1) == False]
    outliers = pd.concat([outliers, new_outliers])
    data = data.drop(new_outliers.index)

    # Dropping the duplicates (if any)
    data = data.drop_duplicates()

    return data, outliers

# Initializing few common variables
rabi_season_fns = ['oct_1f', 'oct_2f', 'nov_1f', 'nov_2f', 'dec_1f', 'dec_2f', 
                    'jan_1f', 'jan_2f', 'feb_1f', 'feb_2f', 'mar_1f', 'mar_2f', 'apr_1f', 'apr_2f']
crop_maps = {'0':'Mustard', '1':'Wheat', '2':'Potato'}

def get_croptype_predictions(data: np.ndarray, algorithm: str='xgb', classifier_type : str='wmp', batch_size: int=8) -> (np.ndarray, np.ndarray):
    '''
    Classifies the crop signatures into wheat, mustard and potato
    Parameters
    -----------
    data : ndarray 
        NDVI recorded every fortnight from the 1st fortnight of october to 2nd fortnight of april.
    algorithm : {'xgb', 'rnn'}, default 'xgb'
        Algorithm to be used for prediction. 'xgb' stands for XGBoost, 'rnn' stands for SimpleRNN
    classifier_type : {'mw','mwp'}, default'mwp'
        Type of classifier to be used. 'mw' stands for mustard/wheat, 'mwp' stands for mustard/wheat/potato
    batch_size : int, default 8
        Batch size for prediction (Applicable only for RNN)
    Returns
    -------
    pred_prob : ndarray
        Probability of each crop type for each data point in the data.
    labels : ndarray
        Crop type label based on max probability for each data point in the data.
    '''
    # Checking arguments for exceptions
    algorithm, classifier_type = algorithm.lower(), classifier_type.lower()

    algo_check(algorithm), classifier_type_check(classifier_type), batch_size_check(algorithm, batch_size, classifier_type), data_check(data)
    data = pd.DataFrame(data, columns=rabi_season_fns)
    classifier = _load_models(algorithm=algorithm, classifier_type=classifier_type)
    scaler = _load_scalers(classifier_type=classifier_type)
    data = scaler.transform(data)
    pred_prob, point_pred = model_prediction(data, algorithm, classifier, batch_size)
    labels = list(map(lambda label: crop_maps[str(label)], point_pred))
    return pred_prob, labels
    
def get_conformal_predictions(data: np.ndarray, classifier_type : str='mwp', alpha: float=0.1) -> (np.ndarray, np.ndarray):
    '''
    
    Parameters
    -----------
    data : ndarray
        NDVI recorded every fortnight from the 1st fortnight of october to 2nd fortnight of april.
    classifier_type : {'mw','mwp'}, default'mwp'
        Type of classifier to be used. 'mw' stands for mustard/wheat, 'mwp' stands for mustard/wheat/potato
    alpha:  float, default 0.1
        Significance level which determines the coverage probability of the prediction set.
    
    Returns
    -------
    set_pred : ndarray
        Conformal prediction set for each data point in the data.
    labels : ndarray
        Crop type label based on max probability from model for each data point in the data.

    Note: Currently, this function only works for XGBoost algorithm. Planning to add support for RNN in the future.
    '''
    
    classifier_type = classifier_type.lower()
    # Checking arguments for exceptions
    classifier_type_check(classifier_type), data_check(data), alpha_check(alpha)

    data = pd.DataFrame(data, columns=rabi_season_fns)
    classifier = _load_models(algorithm='xgb', classifier_type=classifier_type, conformal=True)
    scaler = _load_scalers(classifier_type=classifier_type)
    data = scaler.transform(data)
    point_pred, raw_set = classifier.predict(data, alpha=alpha)
    raw_set = raw_set[:,:len(crop_maps),0]
    set_pred = list(map(lambda row: ' '.join(map(str, np.nonzero(row)[0])), raw_set))
    label_set_map = _generate_label_set_map(crop_maps)
    set_pred = list(map(lambda label: label_set_map[label] if type(label) == str else label, set_pred))
    labels = list(map(lambda label: crop_maps[str(label)], point_pred))
    return set_pred, labels