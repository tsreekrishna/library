import pkg_resources
import pickle
import torch
from crop_classification.src.models import RNNModel
from huggingface_hub import hf_hub_download
import xgboost
import sklearn

def load_models(season:str, end_fn: str) -> xgboost.sklearn.XGBClassifier:
    """
    Loads the classifier from the hugging face hub.
    """
    if season == 'rabi':
        filename=f'oct_1f-{end_fn}.pkl'
    elif season == 'kharif':
        filename=f'may_1f-{end_fn}.pkl'
    path = hf_hub_download(repo_id='krishnamlexp/gt_curation_models', filename=filename,  repo_type="model")
    model = pickle.load(open(path, 'rb'))
    return model
    
def load_scalers(season:str) -> sklearn.preprocessing._data.StandardScaler:
    """
    Loads the scaler from the hugging face hub.
    """
    if season == 'rabi':
        filename='rabi_scaler.pkl'
    elif season == 'kharif':
        filename='kharif_scaler.pkl'
    path = hf_hub_download(repo_id='krishnamlexp/gt_curation_models', filename=filename,  repo_type="model")
    scaler = pickle.load(open(path, 'rb'))
    return scaler

