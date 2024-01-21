import pkg_resources
import pickle
import torch
from crop_classification.src.models import RNNModel

def _load_models(algorithm: str = 'xgb', classifier_type: str = 'mwp', conformal: bool = False):
    
    model = None
    # Loading the XGBoost models
    if algorithm == 'xgb':
        if classifier_type == 'mw':
            model_path = pkg_resources.resource_filename('crop_classification', 'trained_models/mw_xgb_model.pkl')
            model = pickle.load(open(model_path, 'rb'))
        elif classifier_type == 'wmp':    
            model_path = pkg_resources.resource_filename('crop_classification', 'trained_models/mwp_xgb_model.pkl')
            model = pickle.load(open(model_path, 'rb'))

    # Loading the RNN models
    elif algorithm == 'rnn':
        if classifier_type == 'mw':
            model = RNNModel(**{'hidden_layers': 1, 'hidden_size': 16, 'input_size': 14, 'output_size': 2})
            model_path = pkg_resources.resource_filename('crop_classification', 'trained_models/mw_rnn_model.pth')
            model.load_state_dict(torch.load(model_path))
        elif classifier_type == 'mwp':
            model = RNNModel(**{'hidden_layers': 1, 'hidden_size': 64, 'input_size': 14, 'output_size': 3})
            model_path = pkg_resources.resource_filename('crop_classification', 'trained_models/mwp_rnn_model.pth')
            model.load_state_dict(torch.load(model_path))

    return model

def _load_scalers(classifier_type: str = 'mwp'):

    scaler = None
    if classifier_type == 'mw':
        scaler_path = pkg_resources.resource_filename('crop_classification', 'trained_models/mw_standard_scaler.pkl')
        scaler = pickle.load(open(scaler_path, 'rb'))
    elif classifier_type == 'wmp':    
        scaler_path = pkg_resources.resource_filename('crop_classification', 'trained_models/mwp_standard_scalerl.pkl')
        scaler = pickle.load(open(scaler_path, 'rb'))

    return scaler

