import pkg_resources
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from mapie.classification import MapieClassifier
import torch
from .src.models import RNNModel
from .src.models import model_prediction
from .src.utils.helper import _sowing_period, _harvest_period, _dip_impute, _less_than_150_drop, _generate_label_set_map
from .src.utils.checks import data_check, algo_check, classifier_type_check, batch_size_check, alphas_check

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

# importing necessary models

def _load_models(algorithm: str='xgb', classifier_type : str='wmp'):
    # Loading the models
    mw_xgb_model_path = pkg_resources.resource_filename('crop_classification', 'trained_models/mw_xgb_model.pkl')
    mw_xgb_model = pickle.load(open(mw_xgb_model_path, 'rb'))
    mwp_xgb_model_path = pkg_resources.resource_filename('crop_classification', 'trained_models/mwp_xgb_model.pkl')
    mwp_xgb_model = pickle.load(open(mwp_xgb_model_path, 'rb'))
    mwp_xgb_mapie_path = pkg_resources.resource_filename('crop_classification', 'trained_models/mwp_xgb_mapie.pkl')
    mwp_xgb_mapie = pickle.load(open(mwp_xgb_mapie_path, 'rb'))
    mw_rnn_model = RNNModel(**{'hidden_layers': 1, 'hidden_size': 16, 'input_size': 14, 'output_size': 2})
    mw_rnn_weights_path = pkg_resources.resource_filename('crop_classification', 'trained_models/mw_rnn_model.pth')
    mw_rnn_model.load_state_dict(torch.load(mw_rnn_weights_path))
    mwp_rnn_model = RNNModel(**{'hidden_layers': 1, 'hidden_size': 64, 'input_size': 14, 'output_size': 3})
    mwp_rnn_weights_path = pkg_resources.resource_filename('crop_classification', 'trained_models/mwp_rnn_model.pth')
    mwp_rnn_model.load_state_dict(torch.load(mwp_rnn_weights_path))

mw_xgb_model_path = pkg_resources.resource_filename('crop_classification', 'trained_models/mw_xgb_model.pkl')
mw_xgb_model = pickle.load(open(mw_xgb_model_path, 'rb'))
mwp_xgb_model_path = pkg_resources.resource_filename('crop_classification', 'trained_models/mwp_xgb_model.pkl')
mwp_xgb_model = pickle.load(open(mwp_xgb_model_path, 'rb'))
mwp_xgb_mapie_path = pkg_resources.resource_filename('crop_classification', 'trained_models/mwp_xgb_mapie.pkl')
mwp_xgb_mapie = pickle.load(open(mwp_xgb_mapie_path, 'rb'))
mw_rnn_model = RNNModel(**{'hidden_layers': 1, 'hidden_size': 16, 'input_size': 14, 'output_size': 2})
mw_rnn_weights_path = pkg_resources.resource_filename('crop_classification', 'trained_models/mw_rnn_model.pth')
mw_rnn_model.load_state_dict(torch.load(mw_rnn_weights_path))
mwp_rnn_model = RNNModel(**{'hidden_layers': 1, 'hidden_size': 64, 'input_size': 14, 'output_size': 3})
mwp_rnn_weights_path = pkg_resources.resource_filename('crop_classification', 'trained_models/mwp_rnn_model.pth')
mwp_rnn_model.load_state_dict(torch.load(mwp_rnn_weights_path))

# importing necessary scalers
mw_scaler_path = pkg_resources.resource_filename('crop_classification', 'trained_models/mw_standard_scaler.pkl')
mw_scaler = pickle.load(open(mw_scaler_path, 'rb'))
mwp_scaler_path = pkg_resources.resource_filename('crop_classification', 'trained_models/mwp_standard_scaler.pkl')
mwp_scaler = pickle.load(open(mwp_scaler_path, 'rb'))

# Initializing few common variables
rabi_season_fns = ['oct_1f', 'oct_2f', 'nov_1f', 'nov_2f', 'dec_1f', 'dec_2f', 
                    'jan_1f', 'jan_2f', 'feb_1f', 'feb_2f', 'mar_1f', 'mar_2f', 'apr_1f', 'apr_2f']
model_maps = {'xgb':{'mw':mw_xgb_model, 'mwp': mwp_xgb_model}, 'rnn':{'mw':mw_rnn_model, 'mwp': mwp_rnn_model}}
scaler_maps = {'mw':mw_scaler, 'mwp': mwp_scaler}
crop_maps = {'0':'Mustard', '1':'Wheat', '2':'Potato'}

def get_croptype_prediction(data: np.ndarray, algorithm: str='xgb', classifier_type : str='wmp', batch_size: int=8) -> (np.ndarray, np.ndarray):
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
    algo_check(algorithm), classifier_type_check(classifier_type), batch_size_check(batch_size), data_check(data)

    algorithm, classifier_type = algorithm.lower(), classifier_type.lower()
    data = pd.DataFrame(data, columns=rabi_season_fns)
    classifier = model_maps[algorithm][classifier_type]
    scaler = scaler_maps[classifier_type]
    data = scaler.transform(data)
    model = model_prediction(algorithm, classifier)
    pred_prob, point_pred = model.fit_predict(data, batch_size=batch_size)
    labels = list(map(lambda label: crop_maps[str(label)], point_pred))
    return pred_prob, labels
    
def get_conformal_prediction(data: np.ndarray, classifier_type : str='mwp', alphas: np.array=[0.05]) -> (np.ndarray, np.ndarray):
    '''
    
    Parameters
    -----------
    data : ndarray
        NDVI recorded every fortnight from the 1st fortnight of october to 2nd fortnight of april.
    classifier_type : {'mw','mwp'}, default'mwp'
        Type of classifier to be used. 'mw' stands for mustard/wheat, 'mwp' stands for mustard/wheat/potato
    alpha:  array, default [0.05]
        Significance level which determines the coverage probability of the prediction set.
    
    Returns
    -------

    Note: Currently, this function only works for XGBoost algorithm. Planning to add support for RNN in the future.
    '''
    # Checking arguments for exceptions
    classifier_type_check(classifier_type), data_check(data), alphas_check(alphas)

    classifier_type = classifier_type.lower()
    data = pd.DataFrame(data, columns=rabi_season_fns)
    classifier = model_maps[algorithm][classifier_type]
    scaler = scaler_maps[classifier_type]
    data = scaler.transform(data)
    conf_pred = MapieClassifier(estimator=classifier, cv="prefit", method="aps")
    val_path = pkg_resources.resource_filename('crop_classification', 'data_files/val.csv')
    val = pd.read_csv(open(val_path, 'rb'))
    X_val = val.drop('crop_name', axis=1)
    y_val = val['crop_name']
    conf_pred.fit(X_val, y_val)
    py_pred, y_pis = mapie_classifier.predict(data, alpha=alphas)
    label_set_map = _generate_label_set_map(crop_maps)
    labels = list(map(lambda label: label_set_map[label] if type(label) == str else label, set_pred))
    return pred_proba, labels