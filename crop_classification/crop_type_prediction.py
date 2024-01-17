import pkg_resources
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import torch
from .src.models import RNNModel
from .src.modules import conformal_prediction, model_prediction
from .src.utils import _sowing_period, _harvest_period, _dip_impute, _less_than_150_drop, _generate_label_set_map

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

# importing necessary model/data files
mw_xgb_model_path = pkg_resources.resource_filename('crop_classification', 'trained_models/mw_xgb_model.pkl')
# print('aa')
mw_xgb_model = pickle.load(open(mw_xgb_model_path, 'rb'))
mw_xgb_scaler_path = pkg_resources.resource_filename('crop_classification', 'trained_models/mw_xgb_scaler.pkl')
mw_xgb_scaler = pickle.load(open(mw_xgb_scaler_path, 'rb'))
mwp_xgb_model_path = pkg_resources.resource_filename('crop_classification', 'trained_models/mwp_xgb_model.pkl')
mwp_xgb_model = pickle.load(open(mwp_xgb_model_path, 'rb'))
mwp_xgb_scaler_path = pkg_resources.resource_filename('crop_classification', 'trained_models/mwp_xgb_scaler.pkl')
mwp_xgb_scaler = pickle.load(open(mwp_xgb_scaler_path, 'rb')) 
mw_rnn_model = RNNModel(**{'hidden_layers': 1, 'hidden_size': 16, 'input_size': 14, 'output_size': 2})
mw_rnn_weights_path = pkg_resources.resource_filename('crop_classification', 'trained_models/mw_rnn_model.pth')
mw_rnn_model.load_state_dict(torch.load(mw_rnn_weights_path))
mw_rnn_scaler_path = pkg_resources.resource_filename('crop_classification', 'trained_models/mw_rnn_scaler.pkl')
mw_rnn_scaler = pickle.load(open(mw_rnn_scaler_path, 'rb'))
mwp_rnn_model = RNNModel(**{'hidden_layers': 1, 'hidden_size': 64, 'input_size': 14, 'output_size': 3})
mwp_rnn_weights_path = pkg_resources.resource_filename('crop_classification', 'trained_models/mwp_rnn_model.pth')
mwp_rnn_model.load_state_dict(torch.load(mwp_rnn_weights_path))
mwp_rnn_scaler_path = pkg_resources.resource_filename('crop_classification', 'trained_models/mwp_rnn_scaler.pkl')
mwp_rnn_scaler = pickle.load(open(mwp_rnn_scaler_path, 'rb'))

# Initializing few common variables
rabi_season_fns = ['oct_1f', 'oct_2f', 'nov_1f', 'nov_2f', 'dec_1f', 'dec_2f', 
                    'jan_1f', 'jan_2f', 'feb_1f', 'feb_2f', 'mar_1f', 'mar_2f', 'apr_1f', 'apr_2f']
model_maps = {'XGB':{'classifier_types':{'mw':mw_xgb_model, 'mwp': mwp_xgb_model}, 'scaler':{'mw':mw_xgb_scaler, 'mwp': mwp_xgb_scaler}}, 
            'RNN':{'classifier_types':{'mw':mw_rnn_model, 'mwp': mwp_rnn_model}, 'scaler':{'mw':mw_rnn_scaler, 'mwp': mwp_rnn_scaler}}}
crop_maps = {'0':'Mustard', '1':'Wheat', '2':'Potato'}

def get_croptype_prediction(data: np.ndarray, algorithm: str='XGB', classifier_type : str='MWP', batch_size: int=8) -> (np.ndarray, np.ndarray):
    '''
    Classifies the crop signatures into wheat, mustard and potato
    Parameters
    -----------
    data : ndarray 
        NDVI recorded every fortnight from the 1st fortnight of october to 2nd fortnight of april.
    algorithm : {'XGB', 'RNN'}, default 'XGB'
        Algorithm to be used for prediction. 'XGB' stands for XGBoost, 'RNN' stands for SimpleRNN
    classifier_type : {'mw','mwp'}, default'mwp'
        Type of classifier to be used. 'mw' stands for mustard/wheat, 'mwp' stands for mustard/wheat/potato
    batch_size : int, default 8
        Batch size for prediction (Applicable only for RNN)
    Returns
    -------
    2 pandas dataframes -> preprocessed crop, non-crop
    '''
    data = pd.DataFrame(data, columns=rabi_season_fns)
    try:
        classifier = model_maps[algorithm]['classifier_types'][classifier_type]
    except KeyError:
        raise ValueError('Invalid algorithm or classifier type')
    scaler = model_maps[algorithm]['scaler'][classifier_type]
    data = pd.DataFrame(scaler.transform(data), columns=data.columns)
    model = model_prediction(algorithm, classifier)
    pred_prob, point_pred = model.fit_predict(data, batch_size=batch_size)
    labels = list(map(lambda label: crop_maps[str(label)], point_pred))
    return pred_prob, labels
    
def get_conformal_prediction(data: np.ndarray, algorithm: str='XGB', classifier_type : str='MWP', batch_size: int=8, alpha: float=0.1) -> (np.ndarray, np.ndarray):
    '''
    
    Parameters
    -----------
    data : ndarray
        NDVI recorded every fortnight from the 1st fortnight of october to 2nd fortnight of april.
    algorithm : str
        Algorithm to be used for prediction (XGBoost or SimpleRNN)
    classifier_type : str
        Type of classifier to be used. 'MW' stands for mustard/wheat, 'MWP' stands for mustard/wheat/potato
    batch_size : int, default 8
        Batch size for prediction (Applicable only for RNN)
    alpha: float, default 0.1
        Significance level which determines the coverage probability of the prediction set.
    '''
    data = pd.DataFrame(data, columns=rabi_season_fns)
    try:
        classifier = model_maps[algorithm]['classifier_types'][classifier_type]
    except KeyError:
        raise ValueError('Invalid algorithm or classifier type')    
    cp = conformal_prediction(algorithm, estimator=classifier)
    val_path = pkg_resources.resource_filename('crop_classification', 'data_files/val.csv')
    val = pd.read_csv(open(val_path, 'rb'))
    X_val = val.drop('crop_name', axis=1)
    y_val = val['crop_name']
    cp.fit(X_cal=X_val, y_cal=y_val, batch_size=batch_size)
    pred_proba, set_pred = cp.predict(data, alpha=alpha, batch_size=batch_size)
    label_set_map = _generate_label_set_map(crop_maps)
    labels = list(map(lambda label: label_set_map[label] if type(label) == str else label, set_pred))
    return pred_proba, labels