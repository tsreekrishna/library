import pandas as pd
import pickle
from copy import deepcopy
import numpy as np
from xgboost import XGBClassifier
import pkg_resources
from .data_preprocess import get_model_data, conformal_prediction, generate_label_set_map

# Creating scaler and classifier objects
scaler_path = pkg_resources.resource_filename('crop_classification', 'src/models/XGB_standard_scaler.pkl')
scaler = pickle.load(open(scaler_path, 'rb'))
classifier_path = pkg_resources.resource_filename('crop_classification', 'src/models/XGB_oct_2f-feb_1f.pkl')
classifier = pickle.load(open(classifier_path, 'rb'))

def point_predictions(data):
    '''
    Classifies the data into Wheat and Mustard

    Parameters
    -----------
    data : Data which has to fed to the classifier (Has to be in the form 
    of Pandas Dataframe)

    Returns
    -------
    point predictions based on max prob among all the classes

    '''
    scaled_data = pd.DataFrame(scaler.transform(data), columns=data.columns)
    scaled_data = scaled_data.loc[:, 'oct_2f':'feb_1f']
    point_pred = classifier.predict(scaled_data)
    crop_map = {'0':'Mustard', '1':'Wheat'}
    labels = list(map(lambda label: crop_map[str(label)], point_pred))
    pred_prob = classifier.predict_proba(scaled_data)
    return pred_prob, labels

def conformal_predictions(data, alpha=0.05):
    '''
    Classifies the data into wheat, mustard, wheat/mustard or NONE. 

    Parameters
    -----------
    data : Data which has to fed to the classifier (Has to be in the form 
    of Pandas Dataframe)

    Returns 
    -------
    set predictions using conformal_predictions algorithm

    '''
    scaled_data = pd.DataFrame(scaler.transform(data), columns=data.columns)
    scaled_data = scaled_data.loc[:, 'oct_2f':'feb_1f']
    _, val, _ = get_model_data()
    X_cal, y_cal = val.drop('crop_name', axis=1), val['crop_name']
    scaled_X_cal = pd.DataFrame(scaler.transform(X_cal), columns=X_cal.columns)
    scaled_X_cal = scaled_X_cal.loc[:, 'oct_2f':'feb_1f']    
    cp = conformal_prediction(classifier, 'XGB')
    cp.fit(scaled_X_cal, y_cal)
    _, sets = cp.predict(scaled_data, alpha)
    crop_map = {'0':'Mustard', '1':'Wheat'}
    crop_set_map = generate_label_set_map(crop_map)
    labels = list(map(lambda label: crop_set_map[label] if type(label) == str else label, sets))
    return labels


