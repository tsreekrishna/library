from src.models import RNNModel
from xgboost import XGBClassifier
from src.modules import conformal_prediction, model_prediction
import pkg_resources
import pickle
import numpy as np
import pandas as pd
import torch
from src.utils import _batch_prediction_prob

# Pickled model imports
mw_xgb_model_path = pkg_resources.resource_filename('crop_classification', 'trained_models/mw_xgb_model.pkl')
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
    rabi_season_fns = ['oct_1f', 'oct_2f', 'nov_1f', 'nov_2f', 'dec_1f', 'dec_2f', 
                       'jan_1f', 'jan_2f', 'feb_1f', 'feb_2f', 'mar_1f', 'mar_2f', 'apr_1f', 'apr_2f']
    data = pd.DataFrame(data, columns=rabi_season_fns)
    model_maps = {'XGB':{'classifier_types':{'mw':mw_xgb_model, 'mwp': mwp_xgb_model}, 'scaler':{'mw':mw_xgb_scaler, 'mwp': mwp_xgb_scaler}}, 
                          'RNN':{'classifier_types':{'mw':mw_rnn_model, 'mwp': None}, 'scaler':{'mw':mw_rnn_scaler, 'mwp': None}}}
    crop_maps = {'mw':{'0':'Mustard', '1':'Wheat'},'mwp':{'0':'Mustard', '1':'Wheat', '2':'Potato'}}
    try:
        classifier = model_maps[algorithm]['classifier_types'][classifier_type]
    except KeyError:
        raise ValueError('Invalid algorithm or classifier type')
    scaler = model_maps[algorithm]['scaler'][classifier_type]
    data = pd.DataFrame(scaler.transform(data), columns=data.columns)
    model = model_prediction(algorithm, classifier)
    pred_prob, point_pred = model.fit_predict(data)
    crop_map = crop_maps[classifier_type]
    labels = list(map(lambda label: crop_map[str(label)], point_pred))
    return pred_prob, labels
    
# def get_conformal_prediction(data: np.ndarray, algorithm: str='XGB', classifier_type : str='MWP', batch_size: int=8, alpha: float=0.1) -> (np.ndarray, np.ndarray):
#     '''
#     Classifies the data into wheat, mustard, wheat/mustard or NONE.
#     Parameters
#     -----------
#     data : ndarray
#         NDVI recorded every fortnight from the 1st fortnight of october to 2nd fortnight of april.
#     algorithm : str
#         Algorithm to be used for prediction (XGBoost or SimpleRNN)
#     classifier_type : str
#         Type of classifier to be used. 'MW' stands for mustard/wheat, 'MWP' stands for mustard/wheat/potato
#     batch_size : int, default 8
#         Batch size for prediction (Applicable only for RNN)
#     '''
#     pred_proba, _ = get_croptype_prediction(data, algorithm, classifier_type, batch_size)
#     cp = conformal_prediction(algorithm, estimator)
