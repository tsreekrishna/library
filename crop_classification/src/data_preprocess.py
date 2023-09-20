import pandas as pd
from copy import deepcopy
import numpy as np
from xgboost import XGBClassifier
import pkg_resources
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import torch
from itertools import combinations

def get_raw_data():
    '''
    Raw data

    Returns
    -------
    raw data in pandas dataframe format

    '''
    raw_data = pkg_resources.resource_filename('crop_classification', 'src/data_files/combined_for_ingestion.csv')
    raw_df = pd.read_csv(raw_data)
    return raw_df

def get_model_data():
    '''
    Data used to build, tune and test the XGboost Model 

    Returns
    -------
    3 Dataframes -> train, validation and test sets used for modelling. 

    '''
    val_data = pkg_resources.resource_filename('crop_classification', 'src/data_files/val-4.csv')
    val_df = pd.read_csv(val_data)
    test_data = pkg_resources.resource_filename('crop_classification', 'src/data_files/test-4.csv')
    test_df = pd.read_csv(test_data)
    train_data = pkg_resources.resource_filename('crop_classification', 'src/data_files/train-4.csv')
    train_df = pd.read_csv(train_data)
    return train_df, val_df, test_df

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_layers, output_size):
        super(RNNModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers

        # RNN layers
        self.rnn = nn.RNN(
            input_size, hidden_size, hidden_layers, batch_first=True
        )
        # Fully connected layer
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        
        # device = 'cpu'

        # x.to(device)
        
        batch_size = x.size(0)
        
        # h0 = torch.zeros(self.hidden_layers, batch_size, self.hidden_size).to(device)
        h0 = torch.zeros(self.hidden_layers, batch_size, self.hidden_size)
        
        # Forward propagation by passing in the input and hidden state into the model
        out, h0 = self.rnn(x, h0)
        
        #Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        out = out[:,-1,:]
        out = self.relu(out)
        out = self.fc(out)
        out = self.softmax(out)

        return out

def batch_prediction_prob(data, n_features, batch_size, trained_classifier):
    tensor = torch.Tensor(data.values)
    data_loader = DataLoader(tensor, batch_size=batch_size)
    with torch.no_grad():
        pred_prob = []
        for batch in data_loader:
            batch = batch.view([batch.shape[0], -1, n_features])
            trained_classifier.eval()
            pred_prob.append(trained_classifier.forward(batch))
    pred_prob = np.vstack([np.array(pred_prob[:-1]).reshape(-1,2), pred_prob[-1]])
    return pred_prob

class conformal_prediction:
    def __init__(self, estimator, estimator_type):
        self.estimator = estimator
        self.quantile = None
        self.coverage = None
        self.true_class_scores = None
        self.estimator_type = estimator_type
        self.n = None
    def fit(self, X_cal, y_cal):
        if self.estimator_type == 'XGB':
            cal_pred_proba = self.estimator.predict_proba(X_cal)
        elif self.estimator_type == 'RNN':
            cal_pred_proba = batch_prediction_prob(X_cal, X_cal.shape[1], 8, self.estimator)
        true_class_prob = np.array(list(map(lambda row, idx:row[idx], cal_pred_proba, y_cal)))
        self.true_class_scores = 1 - true_class_prob
        self.n = X_cal.shape[0]
    def predict(self, X_test, alpha=0.05):
        self.coverage = (self.n+1)*(1 - alpha)/self.n
        self.quantile = np.quantile(self.true_class_scores, self.coverage)
        if self.estimator_type == 'XGB':
            test_pred_proba = self.estimator.predict_proba(X_test)
        elif self.estimator_type == 'RNN':
            test_pred_proba = batch_prediction_prob(X_test, X_test.shape[1], 8, self.estimator)
        scores = 1 - test_pred_proba
        def func(crop):
            crop_set = (crop <= self.quantile).nonzero()[0]
            if len(crop_set) == 0:
                return np.nan
            else:
                return ' '.join(list(map(str, crop_set)))
        pred_set = list(map(func, scores))
        return test_pred_proba, pred_set
    
def generate_label_set_map(label_map):
    set_map = {}
    for i in range(1,len(label_map)+1):
        for comb in combinations(label_map.keys(), i):
            key_comb = ''
            value_comb = ''
            for items in comb:
                key_comb =  key_comb + ' ' + items
                value_comb =  value_comb + ' ' + label_map[items]
            key_comb, value_comb = key_comb.lstrip(), value_comb.lstrip()
            set_map[key_comb] = value_comb
    return set_map

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