from typing import Dict, List
import numpy as np
import pandas as pd
from crop_classification.src.utils.helper import generate_label_set_map
from crop_classification.src.utils.checks import data_check, alpha_check
from crop_classification.src.utils.model_loader import load_models, load_scalers
from crop_classification.src.utils.constants import crop_label_map, season_fn_map

class CropTypeClassifier:
    def __init__(self, season:str, data: np.ndarray|List[List]) -> None:
        """
        Initialize the CropTypeClassifier instance.

        Parameters
        ----------
        season (str): The season for which the classifier is to be initialized. Should be either 'rabi' or 'kharif'.
        data (np.ndarray|List[List]): Predictor set 
        """
        self.season = season
        # Checking data for exceptions
        data_check(self.data)
        self.data = pd.DataFrame(data, columns=season_fn_map[season])

    def get_croptype_predictions(self, end_fn: str) -> (np.ndarray, np.ndarray):
        '''
        Classifies the crop signatures into wheat, mustard and potato
        
        Parameters
        -----------
        end_fn (str): The last fortnight(inclusive) you want to use for classification.
        
        Returns
        -------
        pred_prob : ndarray
            Probability of each crop type for each data point in the data.
        labels : ndarray
            Crop type label based on max probability for each data point in the data.
        '''
        classifier = load_models(self.season, end_fn)
        scaler = load_scalers(self.season)
        scaled_data = scaler.transform(self.data.loc[:,:end_fn])
        pred_prob, point_pred = classifier.predict_proba(scaled_data), classifier.predict(scaled_data)
        labels = list(map(lambda label: crop_label_map[str(label)], point_pred))
        return pred_prob, labels

    def get_conformal_predictions(self, end_fn: str, alpha: float = 0.1) -> (np.ndarray, np.ndarray):
        '''
        Parameters
        -----------
        data : ndarray
            NDVI recorded every fortnight from the 1st fortnight of october to 2nd fortnight of april.
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
        alpha_check(alpha)

        classifier = load_models(self.season, end_fn)
        scaler = load_scalers(self.season)
        scaled_data = scaler.transform(self.data.loc[:,:end_fn])
        _, raw_set = classifier.predict(scaled_data, alpha=alpha)
        raw_set = raw_set[:, :len(crop_label_map), 0]
        set_pred = list(map(lambda row: ' '.join(map(str, np.nonzero(row)[0])), raw_set))
        label_set_map = generate_label_set_map(crop_label_map)
        set_pred = list(map(lambda label: label_set_map[label] if type(label) == str else label, set_pred))
        return set_pred