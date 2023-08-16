import pandas as pd
import pickle
from copy import deepcopy
import numpy as np
from xgboost import XGBClassifier
from importlib import resources
import io
import pkg_resources

def get_raw_data():
    '''
    Raw data

    Returns
    -------
    raw data in pandas dataframe format

    '''
    raw_data = pkg_resources.resource_filename('crop_classification', 'data_files/combined_for_ingestion.csv')
    raw_df = pd.read_csv(raw_data)
    return raw_df

def get_model_data():
    '''
    Data used to build, tune and test the XGboost Model 

    Returns
    -------
    3 Dataframes -> train, validation and test sets used for modelling. 

    '''
    val_data = pkg_resources.resource_filename('crop_classification', 'data_files/val-4.csv')
    val_df = pd.read_csv(val_data)
    test_data = pkg_resources.resource_filename('crop_classification', 'data_files/test-4.csv')
    test_df = pd.read_csv(test_data)
    train_data = pkg_resources.resource_filename('crop_classification', 'data_files/train-4.csv')
    train_df = pd.read_csv(train_data)
    return train_df, val_df, test_df

def _harvest_avg_impute(row):
    lst = deepcopy(row)
    hrvst_strt_idx = lst.index.get_loc('jan_2f')
    for i in range(len(lst.loc['jan_2f':'mar_2f'])):
        actual_idx = i + hrvst_strt_idx
        if (lst[actual_idx] == 1) or (lst[actual_idx] == 0):
            if lst[actual_idx-1] < 140:
                lst[actual_idx] = (lst[actual_idx-1] + lst[actual_idx+1])/2
    return lst

def _dip_impute(row):
    lst = deepcopy(row)
    act_strt_idx = lst.index.get_loc('jan_1f')
    for i in range(len(lst.loc['jan_1f':'jan_1f'])):
        actual_idx = i + act_strt_idx
        if (lst[actual_idx-1] - lst[actual_idx]) >= 20:
            lst[actual_idx] = (lst[actual_idx-1] + lst[actual_idx+1])/2
    return lst

def _sowing_period(row):
    sowing_periods = row.loc['oct_2f':'dec_2f'].index
    sowing_periods_NDVI = row.loc['oct_2f':'dec_2f']
    minima = np.argmin(sowing_periods_NDVI)
    ndvi_values = row.loc['oct_2f':'apr_1f']
    i = minima
    while i < len(sowing_periods):
        if ndvi_values[i] in set(np.arange(110, 141)):
            if (ndvi_values[i+1] - ndvi_values[i]) > 5:
                if ((ndvi_values[i+1] - ndvi_values[i+4]) < 30):
                    return sowing_periods[i]
        i += 1
        
    return 'Unknown'

def _harvest_period(row):
    sowing_period_idx = row.index.get_loc(row['sowing_period'])
    i = sowing_period_idx + 6
    while i < len(row.loc[:'apr_1f']):
        if row[i] < 140:
            return row.index[i-1]
        i += 1
    return 'Unknown'

def _less_than_150_drop(row):
    sp_loc = row.index.get_loc(row['sowing_period'])
    hp_loc = row.index.get_loc(row['harvest_period'])
    if max(row.iloc[sp_loc+1:hp_loc]) < 150:
        return False
    return True

def data_preprocess(data):
    '''
    Cleans the data 

    Parameters
    -----------
    data : Data which has to preprocessed before feeding 
    to scaler for scaling (Has to be in the form of Pandas
    Dataframe)

    Returns
    -------
    2 Dataframes -> Cleaned and Outliers
    Ouliers my be either Non-Wheat/Non-Mustard or Non-crop

    '''
    
    outliers = data[data.loc[:,'oct_2f':'nov_2f'].apply(lambda row:any((i == 1)|(i == 0) for i in row), axis=1)]

    # Rows which have 0s or 1s in the 3 possible sowing periods for data are dropped
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
    data = data.drop_duplicates(ignore_index=True)

    return data, outliers

# Creating scaler and classifier objects
scaler_path = pkg_resources.resource_filename('crop_classification', 'models/standard_scaler.pkl')
scaler = pickle.load(open(scaler_path, 'rb'))
classifier_path = pkg_resources.resource_filename('crop_classification', 'models/oct_2f-feb_1f.pkl')
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
    labels = classifier.predict(scaled_data)
    labels = list(map(lambda label: 'Wheat' if label == 1 else 'Mustard', labels))
    pred_prob = pd.DataFrame(classifier.predict_proba(scaled_data), columns=['Mustard', 'wheat'])
    return labels, pred_prob


class _conformal_prediction:
    def __init__(self, estimator):
        self.estimator = estimator
        self.quantile = None
        self.coverage = None
    def fit(self, X_cal, y_cal, alpha=0.05):
        cal_pred_proba = self.estimator.predict_proba(X_cal)
        true_class_prob = np.array(list(map(lambda row, idx:row[idx], cal_pred_proba, y_cal)))
        true_class_scores = 1 - true_class_prob
        n = X_cal.shape[0]
        self.coverage = (n+1)*(1 - alpha)/n
        self.quantile = np.quantile(true_class_scores, self.coverage)
    def predict(self, X_test):
        test_pred_proba = self.estimator.predict_proba(X_test)
        scores = 1 - test_pred_proba
        def func(crop):
            crop_set = (crop <= self.quantile).nonzero()[0]
            if len(crop_set) == 0:
                return np.nan
            else:
                return ' '.join(list(map(str, crop_set)))
        pred_set = list(map(func, scores))
        return test_pred_proba, pred_set

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
    cp = _conformal_prediction(classifier)
    cp.fit(scaled_X_cal, y_cal, alpha)
    _, labels = cp.predict(scaled_data)
    labels = list(map(lambda label: 'Wheat' if label == '1' else ('Mustard' if label=='0' else label), labels))
    return labels


